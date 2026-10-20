"""Dataset service: listing, meta and deletion [doc 05 §1, doc 04 §5]."""

from pathlib import Path
from typing import Any

from app.core.errors import ApiError
from app.storage.dataset import (
    dataset_exists,
    dataset_in_use,
    delete_dataset_files,
    read_dataset_meta,
)


def _load(home: Path, dataset_hash: str) -> dict[str, Any]:
    meta = read_dataset_meta(home, dataset_hash)
    if meta is None or not dataset_exists(home, dataset_hash):
        raise ApiError(404, "DATASET_NOT_FOUND", f"dataset {dataset_hash} does not exist")
    return meta


def list_datasets(home: Path, *, project_id: str | None = None) -> list[dict[str, Any]]:
    datasets_dir = home / "datasets"
    if not datasets_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in datasets_dir.glob("*/meta.json"):
        meta = read_dataset_meta(home, path.parent.name)
        if meta is None:
            continue
        if project_id is not None and meta.get("project_id") != project_id:
            continue  # D33: optional project filter [doc 05 §1]
        records.append(meta)
    records.sort(key=lambda record: record.get("created_at", ""), reverse=True)
    return records


def get_dataset(home: Path, dataset_hash: str) -> dict[str, Any]:
    meta = _load(home, dataset_hash)
    # Verification status summary [doc 05 §1]; populated in slice 5/6.
    return {**meta, "verification_summary": None}


def delete_dataset(home: Path, dataset_hash: str) -> None:
    _load(home, dataset_hash)  # 404 if missing
    if dataset_in_use(home, dataset_hash):
        raise ApiError(
            409,
            "DATASET_IN_USE",
            "dataset is referenced by an active run and cannot be deleted",
        )
    delete_dataset_files(home, dataset_hash)
