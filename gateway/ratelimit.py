"""Token-bucket rate limiter that runs atomically inside Redis.

Doing read-modify-write in a Lua script means many gateway replicas can share
one bucket per key without races. Time comes from Redis (TIME), so clock skew
between gateway pods does not matter.
"""
import redis.asyncio as redis

TOKEN_BUCKET_LUA = """
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local cost     = tonumber(ARGV[3])
local t        = redis.call('TIME')
local now      = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local state  = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts     = tonumber(state[2]) or now

tokens = math.min(capacity, tokens + (now - ts) / 1000 * rate)
local allowed, retry_after = 0, 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / rate
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, math.ceil(capacity / rate * 1000) + 1000)
return {allowed, tostring(retry_after)}
"""


class TokenBucket:
    def __init__(self, r: redis.Redis, rate_per_sec: float, capacity: int):
        self._script = r.register_script(TOKEN_BUCKET_LUA)
        self.rate = rate_per_sec
        self.capacity = capacity

    async def acquire(self, key: str, cost: float = 1) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        allowed, retry = await self._script(
            keys=[f"ratelimit:{key}"], args=[self.rate, self.capacity, cost]
        )
        return bool(int(allowed)), float(retry)
