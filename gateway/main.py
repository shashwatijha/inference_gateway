"""OpenAI-compatible chat gateway.

Request path:
  auth -> rate limit -> cache lookup -> backpressure check -> enqueue (Redis Stream)
  -> worker runs the model -> tokens come back on this replica's pub/sub channel
  -> streamed to the client as server-sent events.
"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import redis.asyncio as redis
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import metrics
from .cache import cache_key
from .config import settings
from .ratelimit import TokenBucket

log = logging.getLogger("gateway")

# Each replica listens on its own channel; workers publish to the channel named
# in the request's reply_to field. One subscription per replica, not per request.
REPLICA_ID = uuid.uuid4().hex[:12]
REPLY_CHANNEL = f"replies:{REPLICA_ID}"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False


class State:
    redis: redis.Redis
    limiter: TokenBucket
    pending: dict[str, asyncio.Queue] = {}
    listener: asyncio.Task


state = State()


async def _listen_replies() -> None:
    pubsub = state.redis.pubsub()
    await pubsub.subscribe(REPLY_CHANNEL)
    async for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        event = json.loads(msg["data"])
        q = state.pending.get(event["id"])
        if q is not None:  # client may have disconnected already
            q.put_nowait(event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.redis = redis.from_url(settings.redis_url, decode_responses=True)
    state.limiter = TokenBucket(state.redis, settings.rate_per_sec, settings.burst)
    state.listener = asyncio.create_task(_listen_replies())
    yield
    state.listener.cancel()
    await state.redis.aclose()


app = FastAPI(title="inference-gateway", lifespan=lifespan)


def _authenticate(authorization: str | None) -> str:
    if not settings.api_keys:
        return "anonymous"
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in settings.api_keys:
        metrics.REQUESTS.labels("unauthorized").inc()
        raise HTTPException(401, "invalid API key")
    return token


def _chunk(rid: str, created: int, model: str, delta: dict, finish: str | None) -> str:
    body = {
        "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def _completion(rid: str, created: int, model: str, text: str, finish: str, usage: dict) -> dict:
    return {
        "id": rid, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish}],
        "usage": usage,
    }


async def _generate(rid: str, req: ChatRequest, ckey: str | None) -> AsyncIterator[dict]:
    """Enqueue the request and yield worker events: token*, then done or error."""
    q: asyncio.Queue = asyncio.Queue()
    state.pending[rid] = q
    start = time.perf_counter()
    first_token_at = None
    n_tokens = 0
    parts: list[str] = []
    try:
        await state.redis.xadd(settings.stream, {
            "id": rid,
            "reply_to": REPLY_CHANNEL,
            "enqueued_at": f"{time.time():.6f}",
            "deadline": f"{time.time() + settings.request_timeout_s:.6f}",
            "payload": req.model_dump_json(exclude={"stream", "model"}),
        })
        while True:
            event = await asyncio.wait_for(q.get(), timeout=settings.request_timeout_s)
            if event["type"] == "token":
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    metrics.TTFT.observe(first_token_at - start)
                n_tokens += 1
                parts.append(event["text"])
                metrics.OUTPUT_TOKENS.inc()
            elif event["type"] == "done":
                end = time.perf_counter()
                metrics.E2E.observe(end - start)
                if first_token_at and n_tokens > 1:
                    metrics.TPOT.observe((end - first_token_at) / (n_tokens - 1))
                metrics.REQUESTS.labels("ok").inc()
                if ckey and event.get("finish_reason") in ("stop", "length"):
                    await state.redis.set(ckey, json.dumps({
                        "text": "".join(parts), "finish_reason": event["finish_reason"],
                        "usage": event.get("usage", {}),
                    }), ex=settings.cache_ttl_s)
            elif event["type"] == "error":
                metrics.REQUESTS.labels("upstream_error").inc()
            yield event
            if event["type"] in ("done", "error"):
                return
    except asyncio.TimeoutError:
        metrics.REQUESTS.labels("timeout").inc()
        yield {"type": "error", "message": "timed out waiting for the model"}
    finally:
        state.pending.pop(rid, None)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)):
    api_key = _authenticate(authorization)

    allowed, retry_after = await state.limiter.acquire(api_key)
    if not allowed:
        metrics.REQUESTS.labels("rate_limited").inc()
        return JSONResponse({"error": "rate limit exceeded"}, status_code=429,
                            headers={"Retry-After": f"{max(1, round(retry_after))}"})

    rid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = req.model or settings.model_name
    messages = [m.model_dump() for m in req.messages]

    ckey = cache_key(model, messages, req.max_tokens) if req.temperature == 0 else None
    if ckey:
        cached = await state.redis.get(ckey)
        metrics.CACHE.labels("hit" if cached else "miss").inc()
        if cached:
            metrics.REQUESTS.labels("cache_hit").inc()
            hit = json.loads(cached)
            if not req.stream:
                return _completion(rid, created, model, hit["text"], hit["finish_reason"], hit["usage"])

            async def replay():
                yield _chunk(rid, created, model, {"role": "assistant", "content": hit["text"]}, None)
                yield _chunk(rid, created, model, {}, hit["finish_reason"])
                yield "data: [DONE]\n\n"
            return StreamingResponse(replay(), media_type="text/event-stream")

    depth = await state.redis.xlen(settings.stream)
    metrics.QUEUE_DEPTH.set(depth)
    if depth >= settings.max_queue:
        metrics.REQUESTS.labels("overloaded").inc()
        return JSONResponse({"error": "server overloaded, retry later"}, status_code=503,
                            headers={"Retry-After": "2"})

    if req.stream:
        async def sse():
            yield _chunk(rid, created, model, {"role": "assistant"}, None)
            async for ev in _generate(rid, req, ckey):
                if ev["type"] == "token":
                    yield _chunk(rid, created, model, {"content": ev["text"]}, None)
                elif ev["type"] == "done":
                    yield _chunk(rid, created, model, {}, ev.get("finish_reason", "stop"))
                else:
                    yield f"data: {json.dumps({'error': ev['message']})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    parts: list[str] = []
    async for ev in _generate(rid, req, ckey):
        if ev["type"] == "token":
            parts.append(ev["text"])
        elif ev["type"] == "done":
            return _completion(rid, created, model, "".join(parts),
                               ev.get("finish_reason", "stop"), ev.get("usage", {}))
        else:
            return JSONResponse({"error": ev["message"]}, status_code=502)


@app.get("/healthz")
async def healthz():
    await state.redis.ping()
    return {"ok": True, "replica": REPLICA_ID}


@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
