"""In-memory pub/sub per run feeding WebSockets [doc 03 §2.4].

Single event loop, no persistence — events.jsonl is the durable copy
[doc 03 §2.4]. Bounded queues with oldest-first eviction implement the
throttling backpressure from doc 05 §8.
"""

import asyncio
from typing import Any

DEFAULT_QUEUE_MAX = 1000


class EventBroadcaster:
    """Fan-out of run events to connected subscribers."""

    def __init__(self, queue_max: int = DEFAULT_QUEUE_MAX) -> None:
        self._queue_max = queue_max
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_max)
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        subscribers = self._subscribers.get(run_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(run_id, None)

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers.get(run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: evict the oldest event, keep the newest [doc 05 §8].
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass
