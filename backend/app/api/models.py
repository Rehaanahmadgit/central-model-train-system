"""Models/plugins API [doc 05 §2].

The catalog is generated from the plugin registry, so a new trainer appears
here automatically [doc 08 §1]. The UI's model cards and the Configure-step
form come from this payload.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Request

from app.api.checkpoints import describe_entry
from app.checkpoints.catalog import CatalogError, load_catalog
from app.checkpoints.licences import LicenceStore
from app.checkpoints.resolver import CheckpointIndex
from app.core.config import Settings
from app.core.errors import ApiError
from app.training.registry import get_plugin, plugin_catalog

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(
    request: Request,
    task: str | None = None,
    format: str | None = None,  # noqa: A002 - query parameter name
) -> dict[str, Any]:
    """Model families, each with its pretrained variants (filtered by task/format)."""
    settings = cast(Settings, request.app.state.settings)
    try:
        entries = load_catalog(settings.home)
    except CatalogError as exc:
        raise ApiError(500, "CATALOG_INVALID", str(exc)) from exc
    index = CheckpointIndex(settings.home).read()
    accepted = LicenceStore(settings.home)
    items = plugin_catalog()
    for item in items:
        item["variants"] = [
            describe_entry(entry, index, accepted)
            for entry in entries
            if entry.model_family == item.get("family")
            and (task is None or task in entry.task_types)
            and (format is None or format in entry.formats)
        ]
    return {"items": items, "total": len(items)}


@router.get("/{family}")
def get_model(family: str) -> dict[str, Any]:
    plugin = get_plugin(family)
    if plugin is None:
        raise ApiError(404, "UNKNOWN_MODEL_FAMILY", f"no trainer plugin for family {family!r}")
    return {
        "family": plugin.family,
        "supported_tasks": plugin.supported_tasks,
        "supported_formats": plugin.supported_formats,
        "description": plugin.description,
        "needs_gpu": plugin.needs_gpu,
        "hparam_schema": plugin.hparam_schema,
        "default_hparams": plugin.default_hparams,
    }
