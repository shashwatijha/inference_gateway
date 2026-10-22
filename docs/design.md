# Design doc (fill in during week 3)

## Requirements
- Functional: OpenAI-compatible chat API, streaming, per-key rate limits.
- Targets you measured: e.g. p95 TTFT < X s at Y req/s on Z hardware.

## Capacity math
- Decode tokens/s per replica (measured): ...
- Average output tokens per request: ...
- So requests/s per replica ≈ tokens/s ÷ tokens per request = ...
- Replicas for N req/s with 30% headroom: ...
- KV cache memory per request at max context: ... (from experiments/02)

## Key decisions and alternatives
| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Queue | Redis Streams | Kafka, RabbitMQ, direct HTTP | consumer groups + ack + reclaim, already have Redis |
| Token return path | pub/sub per gateway replica | per-request stream, direct worker→gateway HTTP | |
| Rate limiting | token bucket in Lua | fixed window | |
| Worker autoscaling | KEDA on stream length | HPA on CPU | |
| Batching | continuous | static | (your measured numbers) |

## Failure modes
- Worker crashes mid-request: ...
- Redis goes down: ...
- Model server cold start: ... (measured: ... s)
- Overload: queue cap → 503, deadlines drop expired work.

## What I would do next
