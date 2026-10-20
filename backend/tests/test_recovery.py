"""Startup crash recovery tests [doc 04 §3 rule 6, doc 11 §3b]."""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.storage.atomic import atomic_write_json, read_json
from app.storage.recovery import recover_workspace

RUN_ID = "run_20261017_113400_ab12"


def _make_run(
    home: Path, run_id: str, state: str, queued_at: str | None = None
) -> Path:
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {"id": run_id, "state": state, "schema_version": 1}
    if queued_at is not None:
        data["queued_at"] = queued_at
    atomic_write_json(run_dir / "run.json", data)
    return run_dir


def test_running_run_marked_interrupted(home: Path) -> None:
    run_dir = _make_run(home, RUN_ID, "running")
    report = recover_workspace(home)
    assert report.interrupted_run_ids == [RUN_ID]
    assert read_json(run_dir / "run.json")["state"] == "interrupted"


def test_all_interruptible_states_marked(home: Path) -> None:
    made = [
        _make_run(home, f"run_{i:04d}", state)
        for i, state in enumerate(("preparing", "evaluating", "running"))
    ]
    report = recover_workspace(home)
    assert sorted(report.interrupted_run_ids) == sorted(d.name for d in made)
    for run_dir in made:
        assert read_json(run_dir / "run.json")["state"] == "interrupted"


def test_queued_runs_collected_in_queued_at_order(home: Path) -> None:
    _make_run(home, "run_later", "queued", queued_at="2026-10-17T12:00:00")
    _make_run(home, "run_earlier", "queued", queued_at="2026-10-16T09:00:00")
    report = recover_workspace(home)
    assert report.requeued_run_ids == ["run_earlier", "run_later"]
    # Queued runs keep their state; the Job Manager (slice 8) re-queues them.
    assert read_json(home / "runs" / "run_earlier" / "run.json")["state"] == "queued"


def test_terminal_states_untouched(home: Path) -> None:
    for state in ("completed", "failed", "stopped", "interrupted", "paused"):
        _make_run(home, f"run_{state}", state)
    report = recover_workspace(home)
    assert report.interrupted_run_ids == []
    assert report.requeued_run_ids == []
    for state in ("completed", "failed", "stopped", "interrupted", "paused"):
        assert read_json(home / "runs" / f"run_{state}" / "run.json")["state"] == state


def test_corrupt_run_json_quarantined(home: Path) -> None:
    _make_run(home, "run_good", "completed")
    bad_dir = home / "runs" / "run_bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "run.json").write_text("not json")
    report = recover_workspace(home)
    assert (bad_dir / "run.json.corrupt").exists()
    assert not (bad_dir / "run.json").exists()
    assert len(report.quarantined) == 1
    # Good run unaffected.
    assert read_json(home / "runs" / "run_good" / "run.json")["state"] == "completed"


def test_unknown_schema_version_quarantined(home: Path) -> None:
    # schema_version other than 1 → quarantine [doc 04 §3 rule 7].
    run_dir = _make_run(home, "run_future", "running")
    atomic_write_json(
        run_dir / "run.json",
        {"id": "run_future", "state": "running", "schema_version": 99},
    )
    recover_workspace(home)
    assert (run_dir / "run.json.corrupt").exists()


def test_stale_part_files_removed(home: Path) -> None:
    old = home / "tmp" / "uploads" / "u1" / "chunk.part"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    two_hours_ago = time.time() - 2 * 3600
    os.utime(old, (two_hours_ago, two_hours_ago))

    fresh = home / "checkpoints" / ".tmp" / "fresh.part"
    fresh.parent.mkdir(parents=True)
    fresh.write_bytes(b"fresh")

    report = recover_workspace(home)
    assert report.removed_stale_tmp == [old]
    assert not old.exists()
    assert fresh.exists()


def test_jsonl_tails_repaired_everywhere(home: Path) -> None:
    paths = [
        home / "runs" / RUN_ID / "events.jsonl",
        home / "verifications" / "ver_1" / "checks.jsonl",
        home / "audit" / "audit-2026-10.jsonl",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"seq":1}\n{"seq":2')
    report = recover_workspace(home)
    assert sorted(report.repaired_jsonl) == sorted(paths)
    for path in paths:
        assert path.read_text() == '{"seq":1}\n'


def test_recovery_runs_on_app_startup(home: Path) -> None:
    # kill -9 simulation: a `running` run exists before the API starts [doc 11 §3b].
    run_dir = _make_run(home, RUN_ID, "running")
    settings = Settings(home=home, log_level="DEBUG")
    with TestClient(create_app(settings)):
        pass  # lifespan runs recovery at startup
    assert read_json(run_dir / "run.json")["state"] == "interrupted"


def test_recovery_uses_per_run_lock(home: Path) -> None:
    # The lock file for the run is created during the interrupted-mark write [doc 04 §3 rule 3].
    _make_run(home, RUN_ID, "running")
    recover_workspace(home)
    assert (home / "locks" / f"run-{RUN_ID}.lock").exists()


def test_recovery_of_empty_workspace(home: Path) -> None:
    report = recover_workspace(home)
    assert not report.changed_anything


def test_recovery_respects_custom_stale_age(home: Path) -> None:
    path = home / "tmp" / "x.part"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    report = recover_workspace(
        home,
        now=datetime.now() + timedelta(minutes=1),
        stale_after=timedelta(seconds=1),
    )
    assert report.removed_stale_tmp == [path]
