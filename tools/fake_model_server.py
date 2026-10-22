"""A stand-in for llama.cpp's OpenAI-compatible server, for tests and for
measuring gateway/queue overhead without a model in the way.

It mimics the shape of real inference: a prefill delay proportional to prompt
length, then one token every DECODE_MS, with at most SLOTS requests running at
once (like llama.cpp's --parallel). Each extra running request makes a decode
step a little slower, as a real batched forward pass does.

    uvicorn tools.fake_model_server:app --port 8080
"""
import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

PREFILL_MS_PER_TOKEN = float(os.getenv("FAKE_PREFILL_MS_PER_TOKEN", "0.5"))
DECODE_MS = float(os.getenv("FAKE_DECODE_MS", "20"))
BATCH_PENALTY = float(os.getenv("FAKE_BATCH_PENALTY", "0.1"))  # +10% step time per extra request
SLOTS = int(os.getenv("FAKE_SLOTS", "4"))

app = FastAPI()
slots = asyncio.Semaphore(SLOTS)
running = 0
WORDS = "the quick brown fox jumps over the lazy dog while the model streams tokens".split()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    max_tokens = int(body.get("max_tokens", 64))
    prompt_tokens = sum(len(m["content"].split()) for m in body["messages"])

    async def stream():
        global running
        async with slots:
            running += 1
            try:
                await asyncio.sleep(prompt_tokens * PREFILL_MS_PER_TOKEN / 1000)
                for i in range(max_tokens):
                    await asyncio.sleep(DECODE_MS * (1 + BATCH_PENALTY * (running - 1)) / 1000)
                    chunk = {"choices": [{"index": 0, "delta": {"content": WORDS[i % len(WORDS)] + " "},
                                          "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                                  "total_tokens": prompt_tokens + max_tokens}}
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                running -= 1

    return StreamingResponse(stream(), media_type="text/event-stream")
