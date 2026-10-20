"""Process-pool worker for verification [doc 03 §2.3, doc 13 §3].

Runs the pipeline inside a worker process and streams progress to
``progress.jsonl``; finished checks are appended to ``checks.jsonl``
[doc 04 §1: "one line per check result (append as they finish)"].

Progress is file-based rather than a cross-process queue so the worker can
run under the `spawn` context (a multiprocessing Queue cannot be passed as
an argument to a spawned process) — D47.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from app.storage.jsonl import JsonlAppender
from app.verification.runner import run_pipeline


def run_verification_worker(
    home: Path,
    verification_id: str,
    dataset_hash: str,
    model_family: str,
    task_type: str | None,
    *,
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
            sample_cap=sample_cap,
            check_timeout_seconds=check_timeout_seconds,
            progress=progress,
        )
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

