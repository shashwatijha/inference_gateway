"""Batch collection policy, kept free of Redis so it can be unit tested.

`collect` keeps calling `read(n, wait_ms)` until it has `max_size` items or
`max_wait_ms` has passed since the first item arrived. That is the core
latency/throughput knob of dynamic batching: waiting longer fills bigger
batches (more throughput) but every request in the batch waits too.
"""
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")
ReadFn = Callable[[int, int], Awaitable[list[T]]]


async def collect(read: ReadFn, max_size: int, max_wait_ms: int, idle_block_ms: int = 1000) -> list[T]:
    batch: list[T] = []
    while not batch:  # block until at least one item arrives
        batch = await read(max_size, idle_block_ms)
    deadline = time.monotonic() + max_wait_ms / 1000
    while len(batch) < max_size:
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            break
        batch.extend(await read(max_size - len(batch), remaining_ms))
    return batch
