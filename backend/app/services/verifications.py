"""Verification API service: create, poll, results [doc 05 §3].

The pipeline runs in a ProcessPoolExecutor [doc 03 §2.3]; the API process
polls ``progress.jsonl`` (written by the worker, D47) and broadcasts live
events. The API process is the single writer of verification.json; the
worker is the single writer of checks.jsonl/progress.jsonl [doc 04 §3 rule 3].
"""

import asyncio
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import ApiError
from app.realtime.broadcaster import EventBroadcaster
from app.storage.atomic import atomic_write_json, read_json
from app.storage.audit import append_audit
from app.storage.dataset import dataset_exists
from app.storage.ids import new_verification_id
from app.storage.jsonl import read_jsonl
from app.verification import VERIFIER_VERSION
from app.verification.checks.format_supported import MODEL_TASK_MAP
from app.verification.worker import run_verification_worker

POLL_SECONDS = 0.5


def _verification_path(home: Path, verification_id: str) -> Path:
    return home / "verifications" / verification_id / "verification.json"


def _load(home: Path, verification_id: str) -> dict[str, Any]:
    path = _verification_path(home, verification_id)
    if not path.is_file():
        raise ApiError(
            404, "VERIFICATION_NOT_FOUND", f"verification {verification_id} does not exist"
        )
    data = read_json(path)
    if not isinstance(data, dict):
        raise ApiError(500, "VERIFICATION_CORRUPT", f"verification {verification_id} is unreadable")
    return data


def create_verification(
    home: Path,
    settings: Settings,
    executor: Any,
    broadcaster: EventBroadcaster,
    *,
    dataset_hash: str,
    model_family: str,
    task_type: str | None = None,
) -> dict[str, Any]:
    if not dataset_exists(home, dataset_hash):
        raise ApiError(404, "DATASET_NOT_FOUND", f"dataset {dataset_hash} does not exist")
    if model_family not in MODEL_TASK_MAP:
        raise ApiError(
            400,
            "UNKNOWN_MODEL_FAMILY",
            f"model family {model_family!r} is not known; supported: {sorted(MODEL_TASK_MAP)}",
        )
    verification_id = new_verification_id()
    directory = home / "verifications" / verification_id
    directory.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    record = {
        "id": verification_id,
        "dataset_hash": dataset_hash,
        "model_family": model_family,
        "task_type": task_type,
        "verifier_version": VERIFIER_VERSION,
        "status": "running",
        "started_at": now,
        "finished_at": None,
        "schema_version": 1,
    }
    atomic_write_json(directory / "verification.json", record)
    append_audit(
        home,
        action="verification.started",
        entity_type="verification",
        entity_id=verification_id,
        details={"dataset_hash": dataset_hash, "model_family": model_family},
    )

    future = executor.submit(
        run_verification_worker,
        home,
        verification_id,
        dataset_hash,
        model_family,
        task_type,
        sample_cap=settings.verify_sample_cap,
        check_timeout_seconds=settings.verify_check_timeout_seconds,
    )
    asyncio.create_task(_monitor(home, verification_id, future, broadcaster))
    return {"id": verification_id, "status": "running"}


async def _monitor(
    home: Path,
    verification_id: str,
    future: Future[dict[str, Any]],
    broadcaster: EventBroadcaster,
) -> None:
    progress_path = home / "verifications" / verification_id / "progress.jsonl"
    last_seq = 0
    while True:
        new_events = [
            record
            for record in _read_progress(progress_path)
            if isinstance(record.get("seq"), int) and record["seq"] > last_seq
        ]
        for event in new_events:
            last_seq = max(last_seq, int(event["seq"]))
            await _broadcast(broadcaster, verification_id, event)
            if event["type"] == "verification_finished":
                data = event["data"] if isinstance(event["data"], dict) else {}
                _finalize(home, verification_id, data.get("status", "failed"))
                return
        if not new_events and future.done():
            # Worker exited without a finished event (crash) → failed [doc 03 §8].
            _finalize(home, verification_id, "failed")
            return
        await asyncio.sleep(POLL_SECONDS)


def _read_progress(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return list(read_jsonl(path))
    except OSError:
        return []


async def _broadcast(
    broadcaster: EventBroadcaster, verification_id: str, event: dict[str, Any]
) -> None:
    event_type: Any = event.get("type")
    raw: Any = event.get("data")
    envelope_type: str
    envelope_data: Any
    if event_type == "check_finished":
        envelope_type = "check"
        envelope_data = raw
    elif event_type == "verification_finished":
        envelope_type = "state"
        if isinstance(raw, dict):
            envelope_data = {"from": "running", "to": raw.get("status", "failed")}
        else:
            envelope_data = {"from": "running", "to": "failed"}
    else:
        envelope_type = str(event_type)
        envelope_data = raw
    broadcaster.publish(
        verification_id,
        {
            "seq": None,
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": envelope_type,
            "data": envelope_data,
        },
    )


def _finalize(home: Path, verification_id: str, status: str) -> None:
    path = _verification_path(home, verification_id)
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    data["status"] = status
    data["finished_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(path, data)
    append_audit(
        home,
        action="verification.finished",
        entity_type="verification",
        entity_id=verification_id,
        details={"status": status},
    )


def get_verification(home: Path, verification_id: str) -> dict[str, Any]:
    data = _load(home, verification_id)
    checks = get_checks(home, verification_id)
    status = data.get("status")
    can_train = status not in ("running", "failed")
    return {**data, "cached": False, "checks": checks, "can_train": can_train}


def get_checks(home: Path, verification_id: str) -> list[dict[str, Any]]:
    path = home / "verifications" / verification_id / "checks.jsonl"
    if not path.is_file():
        return []
    return [record["data"] for record in read_jsonl(path) if record.get("type") == "check"]
