# inference-gateway

A self-hosted LLM serving stack built to learn how inference works and to measure what each design choice costs.

```
client ──► gateway (FastAPI) ──► Redis Stream ──► workers ──► model server (llama.cpp / vLLM)
              │  auth, token-bucket rate limit,          │  static or continuous batching
              │  response cache, backpressure            │
              ◄──────── tokens via Redis pub/sub ────────┘
Prometheus scrapes all of it; Kubernetes runs it; KEDA scales workers on queue length.
```

## Quickstart

Python 3.12 is recommended (it's what Docker and CI use); 3.10 to 3.14 are supported. `make install` checks the version.

```bash
brew install python@3.12                # if you don't have it
python3.12 -m venv .venv && source .venv/bin/activate
make install
brew install redis && make test         # 16 tests: rate limiter, batcher, full stack with a fake model

make up-fake                            # whole stack, fake model, no download
curl -s localhost:8000/v1/chat/completions -H 'Authorization: Bearer dev-key' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":20}'

make down && make up                    # same stack with a real model (Qwen2.5-0.5B, 4-bit) on CPU
make bench ARGS="--rate 0 --concurrency 8 --min-tokens 16 --max-tokens 256 --label baseline"
open http://localhost:3000              # Grafana: TTFT, TPOT, tokens/s, queue depth
```

On a Mac, Docker cannot use the GPU. Run `make model-native` in one terminal (llama.cpp on Metal, `brew install llama.cpp`) and `make up-native` in another.

## Layout

| Path | What |
|---|---|
| `gateway/` | OpenAI-compatible `/v1/chat/completions`, SSE streaming, Lua token bucket, exact-match cache, queue backpressure |
| `worker/` | Redis Streams consumer group, static or continuous batching, at-least-once delivery with `XAUTOCLAIM`, deadline-based load shedding |
| `tools/fake_model_server.py` | Model stand-in with realistic prefill/decode timing and parallel slots |
| `experiments/` | One script per concept: generate loop, prefill vs decode, quantization, batching simulator, speculative decoding |
| `loadtest/bench.py` | Poisson-arrival streaming load test; reports TTFT, TPOT and tokens/s, appends to `results/bench.jsonl` |
| `deploy/k8s/` | Kustomize manifests for kind, gateway HPA, KEDA ScaledObject for workers |
| `observability/` | Prometheus config and a provisioned Grafana dashboard |
| `docs/` | Design doc and notes (fill these in as you go) |

## Three-week plan (about 25 h/week)

Each item ends with a number you measured and one sentence in `docs/notes.md` explaining it.

**Week 1: how a model generates text, and serving it**
- [ ] Setup, then `experiments/01_generate_loop.py`: tokens, logits, temperature, top-p (4 h)
- [ ] `experiments/02_prefill_vs_decode.py`: prefill vs decode speed, KV cache on/off, KV memory math (5 h)
- [ ] Docker basics (official "Get started" guide), then `make up-fake`, `make up`, read `gateway/main.py` end to end (6 h)
- [ ] Quantization with llama.cpp: fp16 vs q8_0 vs q4_K_M speed and perplexity (`experiments/README.md`) (5 h)
- [ ] First baseline: `make bench` at 1, 2 and 4 req/s (3 h)

**Week 2: Redis, batching, measurement**
- [ ] Redis basics in `redis-cli`: strings, hashes, streams + consumer groups, pub/sub, `EVAL`. Then read `gateway/ratelimit.py` and `worker/main.py` (5 h)
- [ ] Batching: `BATCH_MODE=static` vs `continuous`, `MAX_INFLIGHT` 1, 2, 4, 8 against the real model. Plot throughput vs p95 latency (6 h)
- [ ] `experiments/04_batching_sim.py --plot`: does the simulation match your measured curve? Why or why not? (3 h)
- [ ] Build one "your turn" feature below, with a before/after measurement (8 h)

**Week 3: Kubernetes, autoscaling, write-up**
- [ ] Kubernetes basics (pods, deployments, services, probes), then `make kind-up k8s-deploy` (6 h)
- [ ] `make keda`, then a traffic spike with `make bench ARGS="--rate 6 --requests 300"`. Watch workers scale and measure model cold start time (5 h)
- [ ] One rented-GPU session: vLLM vs llama.cpp under the same benchmark, optionally speculative decoding (5 h, about $10)
- [ ] Write `docs/design.md` and fill in the results table below; record a 3-minute demo (6 h)

## Your turn (pick at least one; this is what interviewers ask about)

1. **Cancellation.** Stop generating when the client disconnects. Today the worker keeps going and wastes model time. Measure wasted tokens under a load test that drops 20% of clients.
2. **Token coalescing.** Publish tokens every 20 ms instead of one Redis message per token. Measure Redis ops/s and TPOT.
3. **Priority tiers.** Paid keys go to a separate stream that workers drain first. Show free-tier p95 under load with and without it.
4. **Retries and a dead-letter stream.** Use the delivery count from `XPENDING`, retry twice, then move the message to `requests:dead`.
5. **Prefix-aware routing.** With 2+ model replicas, route requests that share a system prompt to the same replica so llama.cpp reuses its prompt cache. Measure the TTFT change.

## Results

Fill in from `results/bench.jsonl`. Example from the fake model server (4 slots, 10 ms/token, 8 concurrent clients, outputs of 8 to 200 tokens):

| Config | tokens/s | TTFT p50 | TTFT p95 | E2E p95 |
|---|---|---|---|---|
| static batching, batch 4 | 215 | 2.41 s | 3.72 s | 5.18 s |
| continuous batching, 4 slots | 287 | 1.36 s | 2.12 s | 4.23 s |

Continuous batching gave 33% more throughput and 44% lower median TTFT at the same slot count. Static batching leaves slots idle until the longest request in the batch finishes.

| Config (real model) | Hardware | tokens/s | TTFT p50 | TPOT p50 | E2E p95 |
|---|---|---|---|---|---|
| llama.cpp q4_K_M, sequential | M3 Pro | | | | |
| llama.cpp q4_K_M, continuous ×4 | M3 Pro | | | | |
| vLLM fp16, continuous | L4 / A10 | | | | |

## Further reading

- kipply, "Transformer Inference Arithmetic": the memory-bound vs compute-bound math
- Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM)
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (continuous batching)
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
