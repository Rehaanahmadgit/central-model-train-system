"""Checkpoint API [doc 05 §4].

Catalog + local availability + disk usage, prefetch, evict, reload.
No user data is involved anywhere here [doc 10 §1].
"""

from __future__ import annotations

import re
import shutil
from typing import Any, cast

from fastapi import APIRouter, Request

from app.checkpoints.catalog import CatalogEntry, CatalogError, ensure_catalog, load_catalog
from app.checkpoints.licences import LicenceStore
from app.checkpoints.resolver import CheckpointError, CheckpointIndex, CheckpointResolver
from app.core.config import Settings
from app.core.errors import ApiError

router = APIRouter(prefix="/checkpoints", tags=["checkpoints"])

CATALOG_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}@[A-Za-z0-9_.\-]{1,32}$")


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def catalog_id_or_400(value: str) -> str:
    if not CATALOG_ID_PATTERN.match(value):
        raise ApiError(400, "INVALID_CATALOG_ID", "catalog id must look like name@version")
    return value


_catalog_id = catalog_id_or_400


def describe_entry(
    entry: CatalogEntry, index: dict[str, dict[str, Any]], accepted: LicenceStore
) -> dict[str, Any]:
    """Catalog entry + local cache state + licence acceptance (shared with /models)."""
    record = index.get(entry.catalog_id) or {}
    local = bool(record.get("status") == "ready")
    return {
        **entry.as_dict(),
        "local": local,
        "local_size_bytes": record.get("size") if local else None,
        "last_used_at": record.get("last_used_at") if local else None,
        "licence_accepted": accepted.is_accepted(entry),
    }


def _error(exc: CheckpointError) -> ApiError:
    status = 404 if exc.code == "CHECKPOINT_NOT_FOUND" else 409
    return ApiError(status, exc.code, exc.message, exc.details)


@router.get("")
def list_checkpoints(
    request: Request,
    task: str | None = None,
    format: str | None = None,  # noqa: A002 - query parameter name
    family: str | None = None,
) -> dict[str, Any]:
    settings = _settings(request)
    try:
        entries = load_catalog(settings.home)
    except CatalogError as exc:
        raise ApiError(500, "CATALOG_INVALID", str(exc)) from exc
    index = CheckpointIndex(settings.home).read()
    accepted = LicenceStore(settings.home)
    usage = shutil.disk_usage(settings.home)
    items = [
        describe_entry(entry, index, accepted)
        for entry in entries
        if (task is None or task in entry.task_types)
        and (format is None or format in entry.formats)
        and (family is None or entry.model_family == family)
    ]
    return {
        "items": items,
        "total": len(items),
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
    }


@router.post("/reload")
def reload_catalog(request: Request) -> dict[str, Any]:
    """Re-read catalog.yaml + override [doc 05 §4]."""
    settings = _settings(request)
    try:
        ensure_catalog(settings.home)
        entries = load_catalog(settings.home)
    except CatalogError as exc:
        raise ApiError(422, "CATALOG_INVALID", str(exc)) from exc
    return {"reloaded": True, "total": len(entries)}


@router.post("/{catalog_id}/prefetch")
def prefetch(catalog_id: str, request: Request) -> dict[str, Any]:
    """Download ahead of time so the first run starts immediately [doc 07 §5]."""
    settings = _settings(request)
    resolver = CheckpointResolver(settings.home)
    try:
        resolved = resolver.prefetch(_catalog_id(catalog_id))
    except CheckpointError as exc:
        raise _error(exc) from exc
    return {
        "catalog_id": resolved.catalog_id,
        "local": True,
        "source": resolved.source,
        "path": str(resolved.path),
        "sha256": resolved.sha256,
        "size_bytes": resolved.size_bytes,
    }


@router.delete("/{catalog_id}/local")
def evict(catalog_id: str, request: Request) -> dict[str, Any]:
    """Evict a cached file; never allowed while a run uses it [doc 05 §4]."""
    settings = _settings(request)
    if _in_use_by_running_job(settings.home, catalog_id):
        raise ApiError(
            409, "CHECKPOINT_IN_USE", f"{catalog_id} is used by a running job"
        )
    removed = CheckpointResolver(settings.home).evict(_catalog_id(catalog_id))
    if not removed:
        raise ApiError(404, "CHECKPOINT_NOT_CACHED", f"{catalog_id} is not cached locally")
    return {"catalog_id": catalog_id, "evicted": True}


def _in_use_by_running_job(home: Any, catalog_id: str) -> bool:
    from app.storage.atomic import read_json

    runs = home / "runs"
    if not runs.is_dir():
        return False
    active = {"preparing", "running", "paused", "evaluating"}
    for run_json in runs.glob("*/run.json"):
        try:
            data = read_json(run_json)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("state") not in active:
            continue
        checkpoint = data.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("catalog_id") == catalog_id:
            return True
    return False
