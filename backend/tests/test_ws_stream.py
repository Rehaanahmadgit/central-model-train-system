"""WebSocket stream tests [doc 05 §8]."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.main import create_app
from app.realtime.events import RunEventLog
from app.storage.atomic import atomic_write_json

RUN_ID = "run_20261017_113400_ab12"


def _make_app(home: Path, **overrides) -> TestClient:
    settings = Settings(
        home=home,
        log_level="DEBUG",
        ws_heartbeat_seconds=overrides.pop("heartbeat", 0.3),
        ws_throttle_interval=overrides.pop("throttle", 0.01),
        ws_resume_timeout_seconds=overrides.pop("resume_timeout", 0.3),
        **overrides,
    )
    return TestClient(create_app(settings))


def _make_run(home: Path, state: str = "running") -> None:
    run_dir = home / "runs" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "run.json", {"id": RUN_ID, "state": state, "schema_version": 1})


def test_ws_snapshot_then_live_event(home: Path) -> None:
    _make_run(home)
    with _make_app(home) as client:
        with client.websocket_connect(f"/ws/runs/{RUN_ID}") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["run_id"] == RUN_ID
            # The server subscribes after the resume window (0.3 s); wait it out.
            time.sleep(0.5)
            app = client.app  # type: ignore[attr-defined]
            app.state.broadcaster.publish(
                RUN_ID,
                {
                    "seq": 1,
                    "ts": "t",
                    "type": "metric",
                    "data": {"name": "train/loss", "value": 1.0, "step": 0},
                },
            )
            received = websocket.receive_json()
            assert received["type"] == "metric"
            assert received["run_id"] == RUN_ID
            assert received["data"]["name"] == "train/loss"


def test_ws_resume_replays_missed_events(home: Path) -> None:
    _make_run(home)
    log = RunEventLog(home, RUN_ID)
    for value in (1.0, 0.9, 0.8):
        log.append("metric", {"name": "train/loss", "value": value, "step": 0})
    log.close()
    with _make_app(home) as client:
        with client.websocket_connect(f"/ws/runs/{RUN_ID}") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "snapshot"
            websocket.send_json({"type": "resume", "last_seq": 1})
            replayed = [websocket.receive_json() for _ in range(2)]
            assert [event["seq"] for event in replayed] == [2, 3]


def test_ws_metric_batching(home: Path) -> None:
    _make_run(home)
    with _make_app(home) as client:
        with client.websocket_connect(f"/ws/runs/{RUN_ID}") as websocket:
            websocket.receive_json()  # snapshot
            time.sleep(0.5)  # wait for the server to subscribe
            app = client.app  # type: ignore[attr-defined]
            for seq in (1, 2, 3):
                app.state.broadcaster.publish(
                    RUN_ID,
                    {
                        "seq": seq,
                        "ts": "t",
                        "type": "metric",
                        "data": {"name": "train/loss", "value": float(seq), "step": 0},
                    },
                )
            received = websocket.receive_json()
            assert received["type"] == "metric"
            batch = received["data"].get("batch", [received["data"]])
            assert [item["value"] for item in batch] == [1.0, 2.0, 3.0]
            assert received["seq"] == 3


def test_ws_heartbeat_sent(home: Path) -> None:
    _make_run(home)
    with _make_app(home, heartbeat=0.2) as client:
        with client.websocket_connect(f"/ws/runs/{RUN_ID}") as websocket:
            websocket.receive_json()  # snapshot
            types: set[str] = set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                message = websocket.receive_json()
                types.add(message["type"])
                if "heartbeat" in types:
                    break
            assert "heartbeat" in types


def test_ws_closes_after_terminal_run(home: Path) -> None:
    _make_run(home, state="completed")
    log = RunEventLog(home, RUN_ID)
    log.append("state", {"from": "running", "to": "completed"})
    log.close()
    with _make_app(home) as client:
        with client.websocket_connect(f"/ws/runs/{RUN_ID}") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "snapshot"
            # Replay then close [doc 05 §8].
            assert websocket.receive_json()["type"] == "state"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_json()


def test_ws_invalid_run_id_rejected(home: Path) -> None:
    with _make_app(home) as client:
        with client.websocket_connect("/ws/runs/bad!id") as websocket:
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_json()
