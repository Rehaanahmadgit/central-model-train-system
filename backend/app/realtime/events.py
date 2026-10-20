"""Per-run event log: appends to runs/{id}/events.jsonl [doc 03 §2.4, doc 04 §2].

- ``seq`` strictly increasing per run; continues across app restarts
- one JSON object per line, append-only [doc 04 §3 rule 2]
- replay helpers for the WebSocket ``last_seq`` resume [doc 05 §8]
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from app.storage.jsonl import JsonlAppender, read_jsonl

SNAPSHOT_METRICS_TAIL = 200
SNAPSHOT_EVENTS_TAIL = 50


class RunEventLog:
    """Single-writer per-run event log [doc 04 §3 rule 3]."""

    def __init__(self, home: Path, run_id: str) -> None:
        self.home = home
        self.run_id = run_id
        self.path = home / "runs" / run_id / "events.jsonl"
        self._appender: JsonlAppender | None = None
        self._next_seq = self._load_last_seq() + 1

    def _load_last_seq(self) -> int:
        if not self.path.is_file():
            return 0
        last = 0
        for record in read_jsonl(self.path):
            seq = record.get("seq")
            if isinstance(seq, int) and seq > last:
                last = seq
        return last

    def append(
        self, event_type: str, data: dict[str, Any], ts: datetime | None = None
    ) -> dict[str, Any]:
        """Append one event and return the full stored record."""
        if self._appender is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._appender = JsonlAppender(self.path)
        record: dict[str, Any] = {
            "seq": self._next_seq,
            "ts": (ts or datetime.now()).isoformat(timespec="milliseconds"),
            "type": event_type,
            "data": data,
        }
        assert self._appender is not None
        self._appender.append(record)
        self._next_seq += 1
        return record

    def sync(self) -> None:
        if self._appender is not None:
            self._appender.sync()

    def close(self) -> None:
        if self._appender is not None:
            self._appender.close()
            self._appender = None

    def last_seq(self) -> int:
        return self._next_seq - 1

    def read_from(self, last_seq: int) -> list[dict[str, Any]]:
        """Replay events with seq > last_seq, oldest first [doc 05 §8]."""
        if not self.path.is_file():
            return []
        return [
            record
            for record in read_jsonl(self.path)
            if isinstance(record.get("seq"), int) and record["seq"] > last_seq
        ]

    def snapshot(self, state: str | None) -> dict[str, Any]:
        """Recent state for the initial snapshot envelope [doc 05 §8]."""
        tail = self.read_from(max(0, self.last_seq() - SNAPSHOT_METRICS_TAIL))
        metrics = [record for record in tail if record["type"] == "metric"][-SNAPSHOT_METRICS_TAIL:]
        events = [
            record
            for record in tail
            if record["type"] in ("event", "state", "epoch_end")
        ][-SNAPSHOT_EVENTS_TAIL:]
        return {
            "state": state,
            "last_seq": self.last_seq(),
            "metrics": metrics,
            "events": events,
        }
