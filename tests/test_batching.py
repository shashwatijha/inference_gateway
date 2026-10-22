import asyncio
import time

from worker.batching import collect


class FakeQueue:
    """Items become readable at given times (seconds after start)."""

    def __init__(self, arrivals: list[float]):
        self.start = time.monotonic()
        self.arrivals = sorted(arrivals)

    async def read(self, n: int, wait_ms: int) -> list[float]:
        end = time.monotonic() + wait_ms / 1000
        while True:
            now = time.monotonic() - self.start
            ready = [a for a in self.arrivals if a <= now][:n]
            if ready:
                for a in ready:
                    self.arrivals.remove(a)
                return ready
            if time.monotonic() >= end:
                return []
            await asyncio.sleep(0.001)


async def test_returns_full_batch_without_waiting():
    q = FakeQueue([0, 0, 0, 0, 0])
    t = time.monotonic()
    batch = await collect(q.read, max_size=4, max_wait_ms=500)
    assert len(batch) == 4
    assert time.monotonic() - t < 0.1


async def test_stops_at_max_wait_with_partial_batch():
    q = FakeQueue([0, 0.02, 1.0])
    t = time.monotonic()
    batch = await collect(q.read, max_size=4, max_wait_ms=100)
    elapsed = time.monotonic() - t
    assert len(batch) == 2
    assert 0.09 < elapsed < 0.3


async def test_blocks_until_first_item():
    q = FakeQueue([0.15])
    t = time.monotonic()
    batch = await collect(q.read, max_size=4, max_wait_ms=0, idle_block_ms=50)
    assert batch == [0.15]
    assert time.monotonic() - t >= 0.14
