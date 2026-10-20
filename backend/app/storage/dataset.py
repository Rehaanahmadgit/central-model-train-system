"""Content-addressed datasets [doc 03 §3.1, doc 04 §1/§4/§5].

- dataset id = sha256 of a sorted manifest of file hashes [doc 04 §4]
- meta.json fields exactly per doc 04 §1 (plus optional project_id, D33)
- deletion removes files and appends a person-free audit line [doc 04 §5]
"""

import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.states import INTERRUPTIBLE_RUN_STATES, QUEUED

from .atomic import atomic_write_json, read_json
from .audit import append_audit

# Runs that may still touch a dataset; deleting while one of these runs
# references it is a 409 [doc 05 §1].
ACTIVE_RUN_STATES = INTERRUPTIBLE_RUN_STATES | {QUEUED, "paused"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hash_from_dir(root: Path) -> str:
    """Content hash: sha256 over sorted ``relpath file_sha256`` lines [doc 04 §4]."""
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            lines.append(f"{relative} {sha256_file(path)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def dataset_meta_path(home: Path, dataset_hash: str) -> Path:
    return home / "datasets" / dataset_hash / "meta.json"


def read_dataset_meta(home: Path, dataset_hash: str) -> dict[str, Any] | None:
    path = dataset_meta_path(home, dataset_hash)
    if not path.is_file():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


def write_dataset_meta(
    home: Path,
    *,
    dataset_hash: str,
    name: str,
    size_bytes: int,
    file_count: int,
    source_filename: str,
    project_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    # Fields per doc 04 §1; format_detected stays null until verification (D16).
    meta: dict[str, Any] = {
        "hash": dataset_hash,
        "name": name,
        "size_bytes": size_bytes,
        "file_count": file_count,
        "format_detected": None,
        "created_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "source_filename": source_filename,
        "schema_version": 1,
    }
    if project_id is not None:
        meta["project_id"] = project_id  # D33 (additive)
    meta_path = dataset_meta_path(home, dataset_hash)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(meta_path, meta)
    return meta


def dataset_exists(home: Path, dataset_hash: str) -> bool:
    return (home / "datasets" / dataset_hash / "raw").is_dir()


def dataset_in_use(home: Path, dataset_hash: str) -> bool:
    """True if any active run references this dataset [doc 05 §1: 409 if used]."""
    runs_dir = home / "runs"
    if not runs_dir.is_dir():
        return False
    for run_json in runs_dir.glob("*/run.json"):
        try:
            data = read_json(run_json)
        except (OSError, ValueError):
            continue
        if (
            isinstance(data, dict)
            and data.get("dataset_hash") == dataset_hash
            and data.get("state") in ACTIVE_RUN_STATES
        ):
            return True
    return False


def delete_dataset_files(home: Path, dataset_hash: str) -> None:
    """Remove the dataset folder and write the audit line [doc 04 §5]."""
    folder = home / "datasets" / dataset_hash
    shutil.rmtree(folder, ignore_errors=False)
    append_audit(
        home,
        action="dataset.deleted",
        entity_type="dataset",
        entity_id=dataset_hash,
        details={},
    )
