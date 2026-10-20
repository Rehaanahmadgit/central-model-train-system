"""Audit trail tests [doc 04 §2, doc 10 §6]."""

from datetime import datetime
from pathlib import Path

import pytest

from app.storage.audit import append_audit
from app.storage.jsonl import read_jsonl


def test_append_audit_writes_date_based_jsonl(home: Path) -> None:
    now = datetime(2026, 10, 17, 12, 0, 0)
    path = append_audit(
        home,
        action="run.started",
        entity_type="run",
        entity_id="run_1",
        details={"state": "queued"},
        now=now,
    )
    assert path == home / "audit" / "audit-2026-10.jsonl"
    records = list(read_jsonl(path))
    assert len(records) == 1
    record = records[0]
    assert record["ts"] == "2026-10-17T12:00:00"
    assert record["action"] == "run.started"
    assert record["entity_type"] == "run"
    assert record["entity_id"] == "run_1"
    assert record["details"] == {"state": "queued"}


def test_audit_appends_across_calls(home: Path) -> None:
    now = datetime(2026, 10, 17, 12, 0, 0)
    append_audit(home, action="a.one", entity_type="run", entity_id="r1", now=now)
    append_audit(home, action="a.two", entity_type="run", entity_id="r1", now=now)
    records = list(read_jsonl(home / "audit" / "audit-2026-10.jsonl"))
    assert [r["action"] for r in records] == ["a.one", "a.two"]


def test_audit_rejects_forbidden_fields(home: Path) -> None:
    # Never write personal data [doc 04 §2, doc 13 §1.2].
    for forbidden in ("email", "user_agent", "ip_address", "user_id", "org_id", "started_by"):
        with pytest.raises(ValueError):
            append_audit(
                home,
                action="run.started",
                entity_type="run",
                entity_id="r1",
                details={forbidden: "x"},
            )


def test_audit_defaults_details_to_empty(home: Path) -> None:
    now = datetime(2026, 10, 17, 12, 0, 0)
    append_audit(home, action="dataset.uploaded", entity_type="dataset", entity_id="d1", now=now)
    record = list(read_jsonl(home / "audit" / "audit-2026-10.jsonl"))[0]
    assert record["details"] == {}
