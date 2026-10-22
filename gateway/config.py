import os
from dataclasses import dataclass, field


def _list(name: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    stream: str = os.getenv("REQUEST_STREAM", "requests")
    # Empty list disables auth (local dev).
    api_keys: list[str] = field(default_factory=lambda: _list("API_KEYS"))
    # Token bucket: requests/second refill and burst size, per API key.
    rate_per_sec: float = float(os.getenv("RATE_PER_SEC", "5"))
    burst: int = int(os.getenv("RATE_BURST", "10"))
    # Reject new work when this many requests are already queued (backpressure).
    max_queue: int = int(os.getenv("MAX_QUEUE", "200"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "120"))
    cache_ttl_s: int = int(os.getenv("CACHE_TTL_S", "3600"))
    model_name: str = os.getenv("MODEL_NAME", "qwen2.5-0.5b-instruct")


settings = Settings()
