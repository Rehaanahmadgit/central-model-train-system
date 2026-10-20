"""RunEventLog tests [doc 03 §2.4, doc 04 §2]."""

from pathlib import Path

from app.realtime.events import RunEventLog
from app.storage.jsonl import read_jsonl


def test_append_assigns_increasing_seq(home: Path) -> None:
    log = RunEventLog(home, "run_1")
    first = log.append("metric", {"name": "train/loss", "value": 1.0})
    second = log.append("metric", {"name": "train/loss", "value": 0.9})
    log.close()
    assert first["seq"] == 1
    assert second["seq"] == 2
    records = list(read_jsonl(home / "runs" / "run_1" / "events.jsonl"))
    assert [record["seq"] for record in records] == [1, 2]
    assert records[0]["type"] == "metric"
    assert records[0]["ts"]


def test_reopen_continues_seq(home: Path) -> None:
    log = RunEventLog(home, "run_1")
    log.append("metric", {"name": "train/loss", "value": 1.0})
    log.close()
    reopened = RunEventLog(home, "run_1")
    assert reopened.last_seq() == 1
    reopened.append("log", {"level": "INFO", "message": "hi"})
    reopened.close()
    records = list(read_jsonl(home / "runs" / "run_1" / "events.jsonl"))
    assert [record["seq"] for record in records] == [1, 2]


def test_read_from_filters_by_seq(home: Path) -> None:
    log = RunEventLog(home, "run_1")
    for value in range(1, 6):
        log.append("metric", {"name": "train/loss", "value": float(value)})
    log.close()
    replayed = RunEventLog(home, "run_1").read_from(2)
    assert [record["seq"] for record in replayed] == [3, 4, 5]


def test_snapshot_shape(home: Path) -> None:
    log = RunEventLog(home, "run_1")
    log.append("state", {"from": "queued", "to": "running"})
    log.append("metric", {"name": "train/loss", "value": 1.0, "step": 0})
    log.append("event", {"name": "checkpoint_saved"})
    log.close()
    snapshot = RunEventLog(home, "run_1").snapshot("running")
    assert snapshot["state"] == "running"
    assert snapshot["last_seq"] == 3
    assert [record["type"] for record in snapshot["metrics"]] == ["metric"]
    assert [record["type"] for record in snapshot["events"]] == ["state", "event"]


def test_empty_log_snapshot(home: Path) -> None:
    log = RunEventLog(home, "run_1")
    snapshot = log.snapshot(None)
    assert snapshot["last_seq"] == 0
    assert snapshot["metrics"] == []
    assert snapshot["events"] == []
    assert not log.path.exists()
