"""Bounded scheduling for resumable, deadline-limited generation stages."""

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def completed(work, items, workers, deadline=None):
    """Yield completed (item, result) pairs; drain in-flight work at the deadline."""
    if workers < 1:
        raise ValueError("workers must be positive")
    items = iter(items)
    skipped = object()

    def start(item):
        if deadline is not None and time.time() >= deadline:
            return skipped
        return work(item)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                if deadline is not None and time.time() >= deadline:
                    exhausted = True
                    break
                try:
                    item = next(items)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(start, item)] = item
            if not pending:
                break
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                item = pending.pop(future)
                result = future.result()
                if result is not skipped:
                    yield item, result
