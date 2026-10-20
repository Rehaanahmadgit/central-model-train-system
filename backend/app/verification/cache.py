"""Verification cache [doc 06 §2 step 1, §6].

Date:
``cache/verif/{sha256(dataset_hash|model_family|verifier_version)}.json``
holds a pointer ``{verification_id, created_at}``. A cache hit returns the
stored report instead of recomputing, which is what makes re-uploading the
same dataset skip verification (US-06). Files only; no database.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from app.storage.atomic import atomic_write_json, read_json

CACHE_SCHEMA_VERSION = 1


def cache_key(dataset_hash: str, model_family: str, verifier_version: str) -> str:
    """Exact key from doc 06 §6: sha256(dataset_hash|model_family|verifier_version)."""
    raw = f"{dataset_hash}|{model_family}|{verifier_version}".encode()
    return hashlib.sha256(raw).hexdigest()


def _pointer_path(home: Path, key: str) -> Path:
    return home / "cache" / "verif" / f"{key}.json"


def write_pointer(home: Path, key: str, verification_id: str, *, preset: str = "auto") -> Path:
    """Write the pointer for the documented key [doc 06 §6].

    ``preset`` is stored alongside (not inside) the key: the cached *dataset
    facts* are preset-independent, but the hardware verdict is not, so a
    different preset must not replay a stale "no GPU needed" result.
    """
    path = _pointer_path(home, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "verification_id": verification_id,
            "preset": preset,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "schema_version": CACHE_SCHEMA_VERSION,
        },
    )
    return path


def read_pointer(home: Path, key: str) -> dict[str, Any] | None:
    path = _pointer_path(home, key)
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("verification_id"), str):
        return None
    return data


def lookup_cached(home: Path, key: str, *, preset: str = "auto") -> dict[str, Any] | None:
    """Return the stored verification record for a finished cache hit."""
    pointer = read_pointer(home, key)
    if pointer is None:
        return None
    if str(pointer.get("preset") or "auto") != preset:
        return None
    verification_id = str(pointer["verification_id"])
    record_path = home / "verifications" / verification_id / "verification.json"
    if not record_path.is_file():
        return None
    try:
        record = read_json(record_path)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("status") not in ("passed", "passed_with_warnings", "failed"):
        # A crash mid-verification leaves a stale pointer: recompute instead.
        return None
    return record
