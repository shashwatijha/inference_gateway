from prometheus_client import Counter, Gauge, Histogram

# Buckets tuned for LLM latencies: TTFT is ms to seconds, full requests can take a minute.
TTFT = Histogram(
    "gateway_ttft_seconds", "Time to first token",
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
TPOT = Histogram(
    "gateway_tpot_seconds", "Mean time per output token after the first",
    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64),
)
E2E = Histogram(
    "gateway_request_seconds", "End-to-end request latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128),
)
REQUESTS = Counter("gateway_requests_total", "Requests by outcome", ["status"])
OUTPUT_TOKENS = Counter("gateway_output_tokens_total", "Generated tokens streamed to clients")
CACHE = Counter("gateway_cache_total", "Response cache lookups", ["result"])
QUEUE_DEPTH = Gauge("gateway_queue_depth", "Requests waiting or in flight (stream length)")
