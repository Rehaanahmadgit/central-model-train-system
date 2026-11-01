"""Runs service: create, read, list [doc 05 §5].

Creation validates that the verification actually passed and is not stale
[doc 06 §9], then writes ``run.json`` (queued) and hands the id to the Job
Manager. No heavy work happens in the event loop [doc 13 §3].
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import ApiError
from app.storage.atomic import atomic_write_json, read_json
from app.storage.audit import append_audit
from app.storage.ids import new_run_id
from app.training.registry import get_plugin

RUN_ID_PATTERN = re.compile(r"^run_[0-9]{8}_[0-9]{6}_[0-9a-f]{4}$")


def _run_path(home: Path, run_id: str) -> Path:
    return home / "runs" / run_id / "run.json"


def create_run(
    home: Path,
    settings: Settings,
    *,
    verification_id: str,
    name: str,
    config: dict[str, Any] | None,
    checkpoint_catalog_id: str | None,
    from_scratch: bool,
    parent_run_id: str | None = None,
    resume_from_last: bool = False,
) -> dict[str, Any]:
    del settings
    verification = read_json(home / "verifications" / verification_id / "verification.json")
    if not isinstance(verification, dict):
        raise ApiError(
            404, "VERIFICATION_NOT_FOUND", f"verification {verification_id} does not exist"
        )
    if verification.get("status") not in ("passed", "passed_with_warnings"):
        raise ApiError(
            409, "VERIFICATION_STALE", "verification has not passed; training cannot start"
        )

    dataset_hash = str(verification.get("dataset_hash") or "")
    model_family = str(verification.get("model_family") or "")
    task_type = verification.get("task_type")
    if not task_type:
        # The wizard does not post task_type; it is detected during verification
        # and stored in report.json's derived facts [doc 06 §6].
        report = read_json(home / "verifications" / verification_id / "report.json")
        if isinstance(report, dict):
            derived = report.get("derived")
            if isinstance(derived, dict):
                task_type = derived.get("task_type")
    plugin = get_plugin(model_family)
    if plugin is None:
        raise ApiError(409, "UNKNOWN_MODEL_FAMILY", f"no trainer for model family {model_family!r}")

    resolved_config = dict(plugin.default_hparams)
    resume_from: Path | None = None
    if parent_run_id:
        parent = read_json(home / "runs" / parent_run_id / "run.json")
        if not isinstance(parent, dict):
            raise ApiError(404, "RUN_NOT_FOUND", f"parent run {parent_run_id} does not exist")
        # Resume inherits the parent's resolved config unless overridden [doc 05 §5].
        resolved_config = dict(parent.get("config") or resolved_config)
        if resume_from_last:
            candidate = home / "runs" / parent_run_id / "checkpoints" / "last.ckpt"
            if candidate.is_file():
                resume_from = candidate
            else:
                raise ApiError(
                    409,
                    "RESUME_UNAVAILABLE",
                    f"parent run {parent_run_id} has no last.ckpt to resume from",
                )
    resolved_config.update(config or {})

    run_id = new_run_id()
    directory = home / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    checkpoint: dict[str, Any] = {"catalog_id": None, "sha256": None, "source": "none"}
    if not from_scratch and checkpoint_catalog_id:
        checkpoint = {"catalog_id": checkpoint_catalog_id, "sha256": None, "source": "download"}
    record: dict[str, Any] = {
        "id": run_id,
        "project_id": None,
        "name": name,
        "dataset_hash": dataset_hash,
        "verification_id": verification_id,
        "model_family": model_family,
        "task_type": task_type,
        "checkpoint": checkpoint,
        "parent_run_id": parent_run_id,
        "resume_from": str(resume_from) if resume_from else None,
        "config": resolved_config,
        "seed": int(resolved_config.get("seed", 42)),
        "env": {},
        "state": "queued",
        "failure": {"reason": None, "details": None},
        "queued_at": now,
        "started_at": None,
        "finished_at": None,
        "best": {"metric": None, "value": 0.0, "epoch": 0},
        "schema_version": 1,
    }
    atomic_write_json(_run_path(home, run_id), record)
    append_audit(
        home,
        action="run.created",
        entity_type="run",
        entity_id=run_id,
        details={"dataset_hash": dataset_hash, "model_family": model_family},
    )
    return record


def get_run(home: Path, run_id: str) -> dict[str, Any]:
    path = _run_path(home, run_id)
    if not path.is_file():
        raise ApiError(404, "RUN_NOT_FOUND", f"run {run_id} does not exist")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ApiError(500, "RUN_CORRUPT", f"run {run_id} is unreadable")
    data["queue_position"] = _queue_position(home, run_id)
    return data


def _queue_position(home: Path, run_id: str) -> int | None:
    """1-based position among queued runs (oldest first); None when not queued."""
    runs_dir = home / "runs"
    if not runs_dir.is_dir():
        return None
    queued: list[tuple[str, str]] = []
    is_queued = False
    for path in runs_dir.glob("*/run.json"):
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("state") != "queued":
            continue
        queued.append((str(data.get("queued_at") or ""), path.parent.name))
        if path.parent.name == run_id:
            is_queued = True
    if not is_queued:
        return None
    for position, (_, queued_id) in enumerate(sorted(queued), start=1):
        if queued_id == run_id:
            return position
    return None


def list_runs(home: Path) -> list[dict[str, Any]]:
    runs_dir = home / "runs"
    if not runs_dir.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*/run.json")):
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            data["queue_position"] = (
                _queue_position(home, str(path.parent.name))
                if data.get("state") == "queued"
                else None
            )
            runs.append(data)
    return runs


def delete_run(home: Path, run_id: str) -> None:
    get_run(home, run_id)
    import shutil

    shutil.rmtree(home / "runs" / run_id)
    append_audit(home, action="run.deleted", entity_type="run", entity_id=run_id)
