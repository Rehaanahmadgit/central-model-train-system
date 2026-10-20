"""Event broadcaster tests [doc 03 §2.4, doc 05 §8 throttling]."""

import asyncio

import pytest

from app.realtime.broadcaster import EventBroadcaster


def test_publish_delivers_to_subscriber() -> None:
    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe("run_1")
    broadcaster.publish("run_1", {"seq": 1, "type": "metric", "data": {}})
    assert queue.qsize() == 1
    broadcaster.unsubscribe("run_1", queue)


def test_publish_ignores_other_runs() -> None:
    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe("run_1")
    broadcaster.publish("run_2", {"seq": 1})
    assert queue.empty()
    broadcaster.unsubscribe("run_1", queue)


def test_unsubscribe_stops_delivery() -> None:
    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe("run_1")
    broadcaster.unsubscribe("run_1", queue)
    broadcaster.publish("run_1", {"seq": 1})
    assert queue.empty()


def test_slow_consumer_evicts_oldest() -> None:
    broadcaster = EventBroadcaster(queue_max=2)
    queue = broadcaster.subscribe("run_1")
    for seq in (1, 2, 3):
        broadcaster.publish("run_1", {"seq": seq})
    assert queue.qsize() == 2
    first = queue.get_nowait()
    second = queue.get_nowait()
    assert first["seq"] == 2
    assert second["seq"] == 3
    broadcaster.unsubscribe("run_1", queue)


@pytest.mark.asyncio
async def test_async_delivery() -> None:
    broadcaster = EventBroadcaster()
    queue = broadcaster.subscribe("run_1")
    broadcaster.publish("run_1", {"seq": 7, "type": "log", "data": {"message": "x"}})
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["seq"] == 7
    broadcaster.unsubscribe("run_1", queue)
