"""Process-pool worker for verification [doc 03 §2.3, doc 13 §3].

Runs the pipeline inside a worker process, streams progress to
``progress.jsonl``, appends finished checks to ``checks.jsonl``
[doc 04 §1], then persists the normalized manifest, the derived report and
the cache pointer so training never recomputes any of it [doc 06 §6].

Progress is file-based rather than a cross-process queue so the worker can
run under any multiprocessing start method — D47.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from app.storage.atomic import atomic_write_json
from app.storage.jsonl import JsonlAppender
from app.verification import VERIFIER_VERSION
from app.verification.cache import cache_key, write_pointer
from app.verification.manifest import write_manifest
from app.verification.runner import REPORT_SCHEMA_VERSION, run_pipeline


def run_verification_worker(
    home: Path,
    verification_id: str,
    dataset_hash: str,
    model_family: str,
    task_type: str | None,
    *,
    preset: str = "auto",
    sample_cap: int,
    check_timeout_seconds: float,
) -> dict[str, Any]:
    dataset_root = home / "datasets" / dataset_hash / "raw"
    directory = home / "verifications" / verification_id
    checks_appender = JsonlAppender(directory / "checks.jsonl")
    progress_appender = JsonlAppender(directory / "progress.jsonl")
    seq = 0
    checks_seq = 0

    def progress(event: dict[str, Any]) -> None:
        nonlocal seq, checks_seq
        seq += 1
        record = {
            "seq": seq,
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": event["type"],
            "data": event["data"],
        }
        progress_appender.append(record)
        if event["type"] == "check_finished":
            # checks.jsonl stores one line per check result with its own
            # consecutive seq and type "check" [doc 04 §1].
            checks_seq += 1
            checks_appender.append({**record, "seq": checks_seq, "type": "check"})

    try:
        outcome = run_pipeline(
            dataset_root,
            dataset_hash,
            model_family,
            task_type,
            preset=preset,
            sample_cap=sample_cap,
            check_timeout_seconds=check_timeout_seconds,
            progress=progress,
            home=home,
        )
        _persist(home, directory, dataset_hash, verification_id, model_family, outcome, preset)
        seq += 1
        progress_appender.append(
            {
                "seq": seq,
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "type": "verification_finished",
                "data": {"status": outcome["status"], "can_train": outcome["can_train"]},
            }
        )
        progress_appender.sync()
        checks_appender.sync()
        return outcome
    finally:
        checks_appender.close()
        progress_appender.close()


def _persist(
    home: Path,
    directory: Path,
    dataset_hash: str,
    verification_id: str,
    model_family: str,
    outcome: dict[str, Any],
    preset: str = "auto",
) -> None:
    """Manifest + report + cache pointer, all atomic [doc 04 §3 rule 1]."""
    manifest_path = home / "datasets" / dataset_hash / "manifest.parquet"
    rows = outcome.get("manifest_rows") or []
    write_manifest(manifest_path, list(rows))

    derived = dict(outcome.get("derived") or {})
    derived["split_manifest_path"] = manifest_path.relative_to(home).as_posix()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "verification_id": verification_id,
        "dataset_hash": dataset_hash,
        "model_family": model_family,
        "verifier_version": VERIFIER_VERSION,
        "status": outcome["status"],
        "can_train": outcome["can_train"],
        "derived": derived,
        "checks": outcome["checks"],
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(directory / "report.json", report)
    write_pointer(
        home,
        cache_key(dataset_hash, model_family, VERIFIER_VERSION),
        verification_id,
        preset=preset,
    )
