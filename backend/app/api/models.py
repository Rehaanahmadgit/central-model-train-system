"""Models/plugins API [doc 05 §2].

The catalog is generated from the plugin registry, so a new trainer appears
here automatically [doc 08 §1]. The UI's model cards and the Configure-step
form come from this payload.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.errors import ApiError
from app.training.registry import get_plugin, plugin_catalog

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models() -> dict[str, Any]:
    items = plugin_catalog()
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
