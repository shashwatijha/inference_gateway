"""Streaming load generator that measures what matters for LLM serving.

Sends requests with Poisson arrivals at --rate requests/s (or as fast as
--concurrency allows when --rate is 0), streams each response, and reports
TTFT, TPOT, end-to-end latency percentiles, and output tokens/s.

    python loadtest/bench.py --rate 2 --requests 100 --max-tokens 128
    python loadtest/bench.py --rate 0 --concurrency 8 --min-tokens 16 --max-tokens 256 --label continuous

Results are appended to results/bench.jsonl so runs can be compared and plotted.
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Write a haiku about GPUs.",
    "Summarize the difference between TCP and UDP.",
    "List three uses of Redis and why each fits.",
    "What is the capital of Australia? Answer in one word.",
    "Describe how a hash map handles collisions, with an example.",
    "Write a Python function that reverses a linked list and explain it step by step.",
    "Give a short history of the transformer architecture.",
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


async def one_request(client: httpx.AsyncClient, args) -> dict:
    body = {
        "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
        # Mixed output lengths are what make scheduling matter: with identical
        # lengths, static and continuous batching look the same.
        "max_tokens": random.randint(args.min_tokens or args.max_tokens, args.max_tokens),
        "temperature": args.temperature,
        "stream": True,
    }
    start = time.perf_counter()
    first = None
    tokens = 0
    status = 0
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            status = r.status_code
            if status != 200:
                await r.aread()
                return {"status": status}
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    tokens += 1
                    if first is None:
                        first = time.perf_counter()
    except httpx.HTTPError as e:
        return {"status": -1, "error": str(e)}
    end = time.perf_counter()
    out = {"status": status, "e2e": end - start, "tokens": tokens}
    if first is not None:
        out["ttft"] = first - start
        if tokens > 1:
            out["tpot"] = (end - first) / (tokens - 1)
    return out


async def run(args) -> dict:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.url, headers=headers, limits=limits,
                                 timeout=httpx.Timeout(300)) as client:
        async def guarded():
            async with sem:
                return await one_request(client, args)

        t0 = time.perf_counter()
        tasks = []
        for _ in range(args.requests):
            tasks.append(asyncio.create_task(guarded()))
            if args.rate > 0:
                await asyncio.sleep(random.expovariate(args.rate))
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["status"] == 200]
    codes: dict[str, int] = {}
    for r in results:
        codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1
    ttft = [r["ttft"] for r in ok if "ttft" in r]
    tpot = [r["tpot"] for r in ok if "tpot" in r]
    e2e = [r["e2e"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)
    return {
        "label": args.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rate": args.rate, "concurrency": args.concurrency, "requests": args.requests,
        "max_tokens": args.max_tokens, "min_tokens": args.min_tokens, "status_codes": codes, "wall_s": round(wall, 2),
        "throughput_req_s": round(len(ok) / wall, 3),
        "throughput_tok_s": round(total_tokens / wall, 1),
        **{f"ttft_p{int(p * 100)}": round(pct(ttft, p), 4) for p in (0.5, 0.95, 0.99)},
        **{f"tpot_p{int(p * 100)}": round(pct(tpot, p), 4) for p in (0.5, 0.95)},
        **{f"e2e_p{int(p * 100)}": round(pct(e2e, p), 3) for p in (0.5, 0.95, 0.99)},
        "tpot_mean": round(statistics.fmean(tpot), 4) if tpot else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="dev-key")
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--rate", type=float, default=1.0, help="Poisson arrival rate (req/s); 0 = closed loop")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--min-tokens", type=int, default=0, help="if set, max_tokens is uniform in [min, max]")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="results/bench.jsonl")
    args = ap.parse_args()

    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
