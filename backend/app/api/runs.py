"""Runs API [doc 05 §5]."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Request

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.pagination import paginate
from app.realtime.events import RunEventLog
from app.schemas.runs import RunCreate, RunOut, RunRetry
from app.services import runs as runs_service
from app.training.job_manager import JobManager, JobNotFoundError

router = APIRouter(prefix="/runs", tags=["runs"])

RUN_ID_PATTERN = re.compile(r"^run_[0-9]{8}_[0-9]{6}_[0-9a-f]{4}$")


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
