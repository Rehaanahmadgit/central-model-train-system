"""Licence acceptance API [Slice C].

Records *what* was accepted and *when* in ``<home>/licences/accepted.json``.
Nothing about the user is stored [doc 10 §1].
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.api.checkpoints import catalog_id_or_400
from app.checkpoints.catalog import CatalogError, find_entry, load_catalog
from app.checkpoints.licences import LicenceStore
from app.core.config import Settings
from app.core.errors import ApiError

router = APIRouter(prefix="/licences", tags=["licences"])


class AcceptRequest(BaseModel):
    catalog_id: str


@router.get("")
def list_accepted(request: Request) -> dict[str, Any]:
    settings = cast(Settings, request.app.state.settings)
    records = LicenceStore(settings.home).records()
    return {"items": records, "total": len(records)}


@router.post("/accept")
def accept(body: AcceptRequest, request: Request) -> dict[str, Any]:
    settings = cast(Settings, request.app.state.settings)
    catalog_id = catalog_id_or_400(body.catalog_id)
    try:
        entry = find_entry(load_catalog(settings.home), catalog_id)
    except CatalogError as exc:
        raise ApiError(500, "CATALOG_INVALID", str(exc)) from exc
    if entry is None:
        raise ApiError(404, "CHECKPOINT_NOT_FOUND", f"{catalog_id!r} is not in the catalog")
    if not entry.offered:
        raise ApiError(
            409,
            "CHECKPOINT_NOT_OFFERED",
            f"{catalog_id!r} is not offered yet (no verified URL/sha256): nothing to accept",
        )
    record = LicenceStore(settings.home).accept(entry)
    return {**record, "accepted": True}
