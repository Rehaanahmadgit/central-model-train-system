"""Runs API [doc 05 §5]."""

from __future__ import annotations

import mimetypes
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.pagination import paginate
from app.realtime.events import RunEventLog
from app.schemas.runs import RunCreate, RunOut, RunRetry
from app.services import runs as runs_service
from app.training.job_manager import JobManager, JobNotFoundError

router = APIRouter(prefix="/runs", tags=["runs"])

RUN_ID_PATTERN = re.compile(r"^run_[0-9]{8}_[0-9]{6}_[0-9a-f]{4}$")
# Only these run sub-directories are downloadable [doc 05 §5].
ARTIFACT_ROOTS: tuple[str, ...] = ("checkpoints", "artifacts")


def _check_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.match(run_id):
        raise ApiError(400, "INVALID_ID", "run id format is invalid")
    return run_id


def _manager(request: Request) -> JobManager:
    return cast(JobManager, request.app.state.job_manager)


def _home(request: Request) -> Path:
    return cast(Settings, request.app.state.settings).home


@router.post("", response_model=RunOut, status_code=202)
async def create(body: RunCreate, request: Request) -> dict[str, Any]:
    record = runs_service.create_run(
        _home(request),
        cast(Settings, request.app.state.settings),
        verification_id=body.verification_id,
        name=body.name,
        config=body.config,
        checkpoint_catalog_id=body.checkpoint_catalog_id,
        from_scratch=body.from_scratch,
    )
    await _manager(request).start(str(record["id"]))
    return record


@router.get("")
def list_runs(request: Request, page: int = 1, page_size: int = 20) -> dict[str, Any]:
    runs = sorted(
        runs_service.list_runs(_home(request)),
        key=lambda item: item.get("queued_at") or "",
        reverse=True,
    )
    return paginate(runs, page, page_size)


@router.get("/{run_id}")
def get(run_id: str, request: Request) -> dict[str, Any]:
    return runs_service.get_run(_home(request), _check_run_id(run_id))


@router.post("/{run_id}/retry", response_model=RunOut, status_code=202)
async def retry(run_id: str, body: RunRetry, request: Request) -> dict[str, Any]:
    """New run linked by parent_run_id; optionally resumes from last.ckpt [doc 05 §5]."""
    parent = runs_service.get_run(_home(request), _check_run_id(run_id))
    if parent.get("state") in ("queued", "preparing", "running", "paused", "evaluating"):
        raise ApiError(409, "RUN_ACTIVE", "cannot retry a run that is still active")
    record = runs_service.create_run(
        _home(request),
        cast(Settings, request.app.state.settings),
        verification_id=str(parent.get("verification_id") or ""),
        name=body.name or f"{parent.get('name', 'run')}-retry",
        config=body.config,
        checkpoint_catalog_id=(
            parent.get("checkpoint") or {}
        ).get("catalog_id")
        if isinstance(parent.get("checkpoint"), dict)
        else None,
        from_scratch=False,
        parent_run_id=run_id,
        resume_from_last=body.resume_from_last,
    )
    await _manager(request).start(str(record["id"]))
    return record


@router.post("/{run_id}/pause")
async def pause(run_id: str, request: Request) -> dict[str, Any]:
    try:
        await _manager(request).pause(_check_run_id(run_id))
    except JobNotFoundError as exc:
        raise ApiError(404, "RUN_NOT_ACTIVE", str(exc)) from exc
    return {"id": run_id, "status": "pause_requested"}


@router.post("/{run_id}/resume")
async def resume(run_id: str, request: Request) -> dict[str, Any]:
    try:
        await _manager(request).resume(_check_run_id(run_id))
    except JobNotFoundError as exc:
        raise ApiError(404, "RUN_NOT_ACTIVE", str(exc)) from exc
    return {"id": run_id, "status": "resume_requested"}


@router.post("/{run_id}/stop")
async def stop(run_id: str, request: Request) -> dict[str, Any]:
    try:
        await _manager(request).stop(_check_run_id(run_id))
    except JobNotFoundError as exc:
        raise ApiError(404, "RUN_NOT_ACTIVE", str(exc)) from exc
    return {"id": run_id, "status": "stop_requested"}


@router.get("/{run_id}/evaluation")
def get_evaluation(run_id: str, request: Request) -> dict[str, Any]:
    """Evaluation report from artifacts/evaluation.json [doc 05 §5]."""
    path = _home(request) / "runs" / _check_run_id(run_id) / "artifacts" / "evaluation.json"
    if not path.is_file():
        raise ApiError(404, "EVALUATION_NOT_FOUND", "this run has no evaluation report yet")
    from app.storage.atomic import read_json

    data = read_json(path)
    if not isinstance(data, dict):
        raise ApiError(500, "EVALUATION_CORRUPT", "evaluation report is unreadable")
    return data


def _predictions_path(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "predictions.json"


@router.get("/{run_id}/predictions")
def get_predictions(run_id: str, request: Request) -> dict[str, Any]:
    """Per-image GT/prediction boxes for the Predictions tab (artifacts/predictions.json)."""
    path = _predictions_path(_home(request) / "runs" / _check_run_id(run_id))
    if not path.is_file():
        raise ApiError(404, "PREDICTIONS_NOT_FOUND", "this run has no predictions yet")
    from app.storage.atomic import read_json

    data = read_json(path)
    if not isinstance(data, dict):
        raise ApiError(500, "PREDICTIONS_CORRUPT", "predictions file is unreadable")
    return data


@router.get("/{run_id}/images")
def get_prediction_image(run_id: str, request: Request, path: str = "") -> FileResponse:
    """Serve one dataset image, but only if predictions.json lists it.

    Paths are relative to the dataset root and must both resolve inside it and
    appear in predictions.json, so a crafted ``path`` can never read arbitrary
    files [doc 10 §5]. Token checks happen in the global middleware like every
    other ``/api`` route.
    """
    checked = _check_run_id(run_id)
    run_dir = _home(request) / "runs" / checked
    if not (run_dir / "run.json").is_file():
        raise ApiError(404, "RUN_NOT_FOUND", f"run {run_id} does not exist")
    if not path or "\x00" in path:
        raise ApiError(400, "INVALID_IMAGE_PATH", "a dataset-relative image path is required")
    predictions = get_predictions(checked, request)
    listed = {
        str(image.get("path"))
        for image in predictions.get("images") or []
        if isinstance(image, dict)
    }
    record = runs_service.get_run(_home(request), checked)
    root = runs_service.dataset_root(_home(request), str(record.get("dataset_hash") or ""))
    root = root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ApiError(400, "INVALID_IMAGE_PATH", "image path escapes the dataset directory")
    if path not in listed:
        raise ApiError(404, "IMAGE_NOT_FOUND", "image is not part of this run's predictions")
    if not resolved.is_file():
        raise ApiError(404, "IMAGE_NOT_FOUND", "image file is missing")
    media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return FileResponse(resolved, media_type=media_type)


def _list_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    """Files under runs/{id}/checkpoints and runs/{id}/artifacts [doc 05 §5]."""
    items: list[dict[str, Any]] = []
    for root_name in ARTIFACT_ROOTS:
        root = run_dir / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            items.append(
                {
                    "name": path.relative_to(run_dir).as_posix(),
                    "dir": root_name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, UTC
                    ).isoformat(timespec="seconds"),
                }
            )
    return items


def _resolve_artifact(run_dir: Path, name: str) -> Path | None:
    """Resolve ``name`` inside runs/{id}/checkpoints|artifacts; never escape the run [doc 10 §5].

    Accepts both ``checkpoints/best.ckpt`` (run-relative) and ``best.ckpt``
    (looked up in each allowed directory).
    """
    root = run_dir.resolve()
    allowed = tuple(root / base for base in ARTIFACT_ROOTS)
    for candidate in (root / name, *(base / name for base in allowed)):
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ApiError(400, "INVALID_ARTIFACT", "artifact path escapes the run directory")
        if any(resolved.is_relative_to(base) for base in allowed) and resolved.is_file():
            return resolved
    return None


@router.get("/{run_id}/artifacts")
def list_artifacts(run_id: str, request: Request) -> dict[str, Any]:
    """Downloadable checkpoints/artifacts for a run."""
    checked = _check_run_id(run_id)
    run_dir = _home(request) / "runs" / checked
    if not (run_dir / "run.json").is_file():
        raise ApiError(404, "RUN_NOT_FOUND", f"run {run_id} does not exist")
    return {"run_id": checked, "artifacts": _list_artifacts(run_dir)}


@router.get("/{run_id}/artifacts/{name:path}")
def download_artifact(run_id: str, name: str, request: Request) -> FileResponse:
    """Stream one artifact file; the path is confined to the allowed directories."""
    checked = _check_run_id(run_id)
    run_dir = _home(request) / "runs" / checked
    if not (run_dir / "run.json").is_file():
        raise ApiError(404, "RUN_NOT_FOUND", f"run {run_id} does not exist")
    path = _resolve_artifact(run_dir, name)
    if path is None:
        raise ApiError(404, "ARTIFACT_NOT_FOUND", f"artifact {name!r} does not exist")
    return FileResponse(path, filename=path.name)


@router.get("/{run_id}/events")
def get_events(run_id: str, request: Request, after_seq: int = 0) -> dict[str, Any]:
    log = RunEventLog(_home(request), _check_run_id(run_id))
    return {"run_id": run_id, "events": log.read_from(after_seq)}


@router.delete("/{run_id}", status_code=204)
def delete(run_id: str, request: Request) -> None:
    record = runs_service.get_run(_home(request), _check_run_id(run_id))
    if record.get("state") in ("queued", "preparing", "running", "paused", "evaluating"):
        raise ApiError(409, "RUN_ACTIVE", "cannot delete a running run")
    runs_service.delete_run(_home(request), run_id)
