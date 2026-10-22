import asyncio

import redis.asyncio as redis

from gateway.ratelimit import TokenBucket


async def test_burst_then_refill(redis_url):
    r = redis.from_url(redis_url)
    bucket = TokenBucket(r, rate_per_sec=10, capacity=3)
    results = [await bucket.acquire("k1") for _ in range(4)]
    assert [ok for ok, _ in results] == [True, True, True, False]
    assert 0 < results[-1][1] <= 0.1  # one token refills in 100 ms
    await asyncio.sleep(0.12)
    assert (await bucket.acquire("k1"))[0]
    await r.aclose()


async def test_keys_are_independent(redis_url):
    r = redis.from_url(redis_url)
    bucket = TokenBucket(r, rate_per_sec=1, capacity=1)
    assert (await bucket.acquire("a"))[0]
    assert not (await bucket.acquire("a"))[0]
    assert (await bucket.acquire("b"))[0]
    await r.aclose()


async def test_concurrent_callers_never_overspend(redis_url):
    r = redis.from_url(redis_url)
    bucket = TokenBucket(r, rate_per_sec=0.001, capacity=5)
    results = await asyncio.gather(*(bucket.acquire("shared") for _ in range(50)))
    assert sum(ok for ok, _ in results) == 5
    await r.aclose()
