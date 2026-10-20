"""Sample dataset (DIST-11) [doc 16 §2].

Generates a tiny dataset from ``app.samples.datasets`` (the same builders the
test suite and the future sample-dataset button share) and installs it through
the normal, audited dataset path — safe extraction, content hash, dedup, meta.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import ApiError
from app.samples.datasets import build_sample_zip
from app.storage.archive import ExtractLimits, extract_archive
from app.storage.audit import append_audit
from app.storage.dataset import (
    dataset_exists,
    dataset_hash_from_dir,
    read_dataset_meta,
    write_dataset_meta,
)

# kind → synthetic spec [doc 11 §2]; tabular is the zero-setup RandomForest demo.
KIND_TO_SPEC = {
    "tabular": "good_tabular",
    "classification": "good_cls",
    "yolo": "good_yolo",
}

KIND_TO_NAME = {
    "tabular": "sample-tabular.csv",
    "classification": "sample-classification.zip",
    "yolo": "sample-yolo.zip",
}


def create_sample_dataset(
    home: Path,
    settings: Settings,
    *,
    kind: str = "tabular",
    now: datetime | None = None,
) -> dict[str, Any]:
    if kind not in KIND_TO_SPEC:
        raise ApiError(
            400,
            "INVALID_SAMPLE_KIND",
            f"kind must be one of {sorted(KIND_TO_SPEC)}",
        )
    content = build_sample_zip(KIND_TO_SPEC[kind])
    filename = KIND_TO_NAME[kind]

    with tempfile.TemporaryDirectory(dir=home / "tmp", prefix="sample-") as tmp:
        staged = Path(tmp) / filename
        staged.write_bytes(content)
        extracted = Path(tmp) / "extracted"
        extracted.mkdir()
        limits = ExtractLimits(
            max_files=settings.max_extract_files,
            max_total_bytes=settings.max_extract_total_bytes,
            zip_bomb_ratio=settings.zip_bomb_ratio,
            zip_bomb_floor_bytes=settings.zip_bomb_floor_bytes,
        )
        extract_archive(staged, extracted, limits)

        file_count = sum(1 for path in extracted.rglob("*") if path.is_file())
        dataset_hash = dataset_hash_from_dir(extracted)
        deduplicated = dataset_exists(home, dataset_hash)
        if deduplicated:
            meta = read_dataset_meta(home, dataset_hash) or {}
        else:
            target = home / "datasets" / dataset_hash
            target.mkdir(parents=True, exist_ok=False)
            extracted.rename(target / "raw")
            meta = write_dataset_meta(
                home,
                dataset_hash=dataset_hash,
                name=filename,
                size_bytes=len(content),
                file_count=file_count,
                source_filename=filename,
            )
    append_audit(
        home,
        action="dataset.uploaded",
        entity_type="dataset",
        entity_id=dataset_hash,
        details={"name": filename, "kind": kind, "sample": True, "deduplicated": deduplicated},
    )
    return {"dataset_hash": dataset_hash, "deduplicated": deduplicated, "dataset": meta}
