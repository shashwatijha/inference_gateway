from prometheus_client import Counter, Gauge, Histogram

BATCH_SIZE = Histogram("worker_batch_size", "Requests per collected batch",
                       buckets=(1, 2, 4, 8, 16, 32, 64))
QUEUE_WAIT = Histogram("worker_queue_wait_seconds", "Time a request spent in the queue",
                       buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
INFLIGHT = Gauge("worker_inflight_requests", "Requests currently running on the model server")
TOKENS = Counter("worker_output_tokens_total", "Tokens generated")
RESULTS = Counter("worker_requests_total", "Requests by outcome", ["status"])
