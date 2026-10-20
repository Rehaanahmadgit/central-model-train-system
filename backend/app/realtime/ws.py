"""WebSocket run stream [doc 05 §8].

Flow: accept → send ``snapshot`` → wait briefly for ``{"type":"resume","last_seq":N}``
→ replay events.jsonl from N → live queue → throttled sends, heartbeat every
``ws_heartbeat_seconds``. If the run is finished: replay, then close [doc 05 §8].
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import Settings
from app.realtime.broadcaster import EventBroadcaster
from app.realtime.events import RunEventLog
from app.storage.atomic import read_json

router = APIRouter()

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,100}$")
TERMINAL_STATES = frozenset({"completed", "failed", "stopped"})


def _run_state(home: Path, run_id: str) -> str | None:
    path = home / "runs" / run_id / "run.json"
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    state = data.get("state")
    return state if isinstance(state, str) else None


def _envelope(run_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "run_id": run_id}


async def _send_metrics(
    websocket: WebSocket, run_id: str, records: list[dict[str, Any]]
) -> None:
    if len(records) == 1:
        await websocket.send_json(_envelope(run_id, records[0]))
        return
    # Coalesce consecutive metrics into one envelope; data.batch is an
    # extension of the single-metric data shape (D40) [doc 05 §8 "batching metrics"].
    last = records[-1]
    await websocket.send_json(
        {
            "type": "metric",
            "run_id": run_id,
            "ts": last["ts"],
            "seq": last["seq"],
            "data": {"batch": [record["data"] for record in records]},
        }
    )


async def _stream_live(
    websocket: WebSocket,
    run_id: str,
    settings: Settings,
    broadcaster: EventBroadcaster,
    queue: asyncio.Queue[dict[str, Any]],
) -> None:
    del broadcaster  # publishing happens outside; this consumer only drains
    loop = asyncio.get_running_loop()
    last_heartbeat = loop.time()
    while True:
        # Timeout so heartbeats fire even when idle [doc 05 §8].
        try:
            first = await asyncio.wait_for(queue.get(), timeout=1.0)
        except TimeoutError:
            first = None
        now = loop.time()
        if now - last_heartbeat >= settings.ws_heartbeat_seconds:
            await websocket.send_json(
                {
                    "type": "heartbeat",  # extension of the envelope types (D41)
                    "run_id": run_id,
                    "ts": datetime.now().isoformat(timespec="milliseconds"),
                    "seq": None,
                    "data": {},
                }
            )
            last_heartbeat = now
        if first is None:
            continue
        # Drain everything pending and batch metrics [doc 05 §8].
        pending = [first]
        while True:
            try:
                pending.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        metric_batch: list[dict[str, Any]] = []
        for event in pending:
            if event["type"] == "metric":
                metric_batch.append(event)
                continue
            if metric_batch:
                await _send_metrics(websocket, run_id, metric_batch)
                metric_batch = []
                await asyncio.sleep(settings.ws_throttle_interval)
            await websocket.send_json(_envelope(run_id, event))
            await asyncio.sleep(settings.ws_throttle_interval)
        if metric_batch:
            await _send_metrics(websocket, run_id, metric_batch)
            await asyncio.sleep(settings.ws_throttle_interval)


@router.websocket("/ws/runs/{run_id}")
async def run_stream(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    if not RUN_ID_PATTERN.match(run_id):
        await websocket.close(code=4400)
        return
    settings = cast(Settings, websocket.app.state.settings)
    broadcaster = cast(EventBroadcaster, websocket.app.state.broadcaster)
    log = RunEventLog(settings.home, run_id)
    state = _run_state(settings.home, run_id)

    # Snapshot first, always [doc 05 §8].
    await websocket.send_json(
        {
            "type": "snapshot",
            "run_id": run_id,
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "seq": log.last_seq(),
            "data": log.snapshot(state),
        }
    )

    last_seq = 0
    try:
        message = await asyncio.wait_for(
            websocket.receive_json(), timeout=settings.ws_resume_timeout_seconds
        )
        if isinstance(message, dict) and message.get("type") == "resume":
            try:
                last_seq = int(message.get("last_seq", 0))
            except (TypeError, ValueError):
                last_seq = 0
    except (TimeoutError, ValueError):
        pass  # no resume → the client gets the full replay

    # Replay missed events from the file, then switch to live [doc 03 §2.4].
    for record in log.read_from(last_seq):
        await websocket.send_json(_envelope(run_id, record))
    state = _run_state(settings.home, run_id)
    if state in TERMINAL_STATES:
        # Finished: replay, then close [doc 05 §8].
        await websocket.close()
        return

    queue = broadcaster.subscribe(run_id)
    try:
        await _stream_live(websocket, run_id, settings, broadcaster, queue)
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.unsubscribe(run_id, queue)
