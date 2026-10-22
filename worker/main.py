"""Queue worker: pulls requests from the Redis Stream, runs them on the model
server, and publishes tokens back to the gateway replica that owns the request.

BATCH_MODE=static      collect up to BATCH_MAX requests (waiting at most
                       BATCH_WAIT_MS), run them together, and wait for the whole
                       batch to finish before taking more. Simple, but one long
                       generation holds up every slot (head-of-line blocking).
                       BATCH_MAX=1 gives one-request-at-a-time serving.
BATCH_MODE=continuous  keep up to MAX_INFLIGHT requests running and admit a new
                       one whenever any slot frees up. This is the scheduling
                       idea behind vLLM / Orca, applied at request level; the
                       model server does the token-level batching itself.
"""
import asyncio
import json
import logging
import os
import socket
import time

import httpx
import redis.asyncio as redis
from prometheus_client import start_http_server
from redis.exceptions import ResponseError

from . import metrics
from .batching import collect

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM = os.getenv("REQUEST_STREAM", "requests")
GROUP = os.getenv("CONSUMER_GROUP", "workers")
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8080").rstrip("/")
BATCH_MODE = os.getenv("BATCH_MODE", "continuous")
BATCH_MAX = int(os.getenv("BATCH_MAX", "4"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "20"))
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "4"))
# A message delivered to a consumer that died is re-claimed after this long.
CLAIM_IDLE_MS = int(os.getenv("CLAIM_IDLE_MS", "60000"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))

log = logging.getLogger("worker")
CONSUMER = f"{socket.gethostname()}-{os.getpid()}"

Entry = tuple[str, dict]


class Worker:
    def __init__(self, r: redis.Redis, http: httpx.AsyncClient):
        self.r = r
        self.http = http
        self.inflight = 0
        self.slot_freed = asyncio.Event()
        self.tasks: set[asyncio.Task] = set()
        self.last_reclaim = 0.0

    async def setup(self) -> None:
        try:
            await self.r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def read(self, n: int, wait_ms: int) -> list[Entry]:
        # At-least-once delivery: first pick up messages that a crashed worker
        # was holding, then new ones.
        if time.monotonic() - self.last_reclaim > CLAIM_IDLE_MS / 1000:
            self.last_reclaim = time.monotonic()
            _, claimed, *_ = await self.r.xautoclaim(
                STREAM, GROUP, CONSUMER, min_idle_time=CLAIM_IDLE_MS, start_id="0-0", count=n)
            if claimed:
                log.warning("reclaimed %d stale messages", len(claimed))
                return claimed
        res = await self.r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=n, block=max(1, wait_ms))
        return [entry for _, entries in res for entry in entries] if res else []

    async def publish(self, channel: str, event: dict) -> None:
        await self.r.publish(channel, json.dumps(event))

    async def handle(self, entry: Entry) -> None:
        msg_id, f = entry
        rid, reply_to = f["id"], f["reply_to"]
        now = time.time()
        metrics.QUEUE_WAIT.observe(now - float(f["enqueued_at"]))
        metrics.INFLIGHT.inc()
        try:
            if now > float(f["deadline"]):
                # The client has already given up; don't spend model time on it.
                metrics.RESULTS.labels("expired").inc()
                await self.publish(reply_to, {"id": rid, "type": "error", "message": "expired in queue"})
                return
            req = json.loads(f["payload"])
            finish, usage = await self.generate(rid, reply_to, req)
            metrics.RESULTS.labels("ok").inc()
            await self.publish(reply_to, {"id": rid, "type": "done", "finish_reason": finish, "usage": usage})
        except Exception as e:  # noqa: BLE001 - report any failure to the client
            log.exception("request %s failed", rid)
            metrics.RESULTS.labels("error").inc()
            await self.publish(reply_to, {"id": rid, "type": "error", "message": f"model error: {e}"})
        finally:
            metrics.INFLIGHT.dec()
            # Ack and delete so XLEN = waiting + in-flight, the autoscaling signal.
            async with self.r.pipeline(transaction=False) as p:
                p.xack(STREAM, GROUP, msg_id)
                p.xdel(STREAM, msg_id)
                await p.execute()

    async def generate(self, rid: str, reply_to: str, req: dict) -> tuple[str, dict]:
        body = {**req, "stream": True, "stream_options": {"include_usage": True}}
        finish, usage, n = "stop", {}, 0
        async with self.http.stream("POST", f"{MODEL_URL}/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        n += 1
                        await self.publish(reply_to, {"id": rid, "type": "token", "text": text})
                    finish = choice.get("finish_reason") or finish
        metrics.TOKENS.inc(n)
        return finish, usage or {"completion_tokens": n}

    def spawn(self, entry: Entry) -> None:
        task = asyncio.create_task(self.handle(entry))
        self.tasks.add(task)
        self.inflight += 1
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        self.inflight -= 1
        self.slot_freed.set()

    async def run(self) -> None:
        await self.setup()
        log.info("worker %s mode=%s batch_max=%d wait_ms=%d max_inflight=%d",
                 CONSUMER, BATCH_MODE, BATCH_MAX, BATCH_WAIT_MS, MAX_INFLIGHT)
        while True:
            if BATCH_MODE == "static":
                batch = await collect(self.read, BATCH_MAX, BATCH_WAIT_MS)
                metrics.BATCH_SIZE.observe(len(batch))
                await asyncio.gather(*(self.handle(e) for e in batch))
            else:
                # Wait for a free slot, then admit up to as many requests as there are free slots.
                while self.inflight >= MAX_INFLIGHT:
                    self.slot_freed.clear()
                    await self.slot_freed.wait()
                batch = await collect(self.read, MAX_INFLIGHT - self.inflight, BATCH_WAIT_MS)
                metrics.BATCH_SIZE.observe(len(batch))
                for entry in batch:
                    self.spawn(entry)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    start_http_server(METRICS_PORT)
    r = redis.from_url(REDIS_URL, decode_responses=True)
    timeout = httpx.Timeout(connect=5, read=300, write=10, pool=None)
    limits = httpx.Limits(max_connections=max(MAX_INFLIGHT, BATCH_MAX) + 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as http:
        await Worker(r, http).run()


if __name__ == "__main__":
    asyncio.run(main())
