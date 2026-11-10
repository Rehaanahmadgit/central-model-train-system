"""Try-it endpoints [Slice G]: POST /runs/{id}/infer, /runs/{id}/infer-samples.

Token-gated by the global ``/api`` middleware. Detection itself runs in a
subprocess (``app.inference``); this process never imports torch.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Request

from app.api.runs import _check_run_id
from app.core.config import Settings
from app.services import infer as infer_service

router = APIRouter(prefix="/runs", tags=["infer"])


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


@router.post("/{run_id}/infer")
async def infer(run_id: str, request: Request, threshold: float | None = None) -> dict[str, Any]:
    """Detect objects in up to 8 uploaded images (multipart ``images`` parts)."""
    checked = _check_run_id(run_id)
    declared = request.headers.get("content-length")
    body = await infer_service.read_body_capped(
        request.stream(), int(declared) if declared and declared.isdigit() else None
    )
    fields, files = infer_service.parse_multipart(request.headers.get("content-type", ""), body)
    limit = infer_service.check_threshold(
        threshold if threshold is not None else fields.get("threshold")
    )
    return await infer_service.infer_uploaded(_settings(request), checked, files, limit)


@router.post("/{run_id}/infer-samples")
async def infer_samples(
    run_id: str, request: Request, threshold: float | None = None
) -> dict[str, Any]:
    """Detect objects in 6 deterministic dataset images (test split, else val)."""
    checked = _check_run_id(run_id)
    limit = infer_service.check_threshold(threshold)
    return await infer_service.infer_samples(_settings(request), checked, limit)
