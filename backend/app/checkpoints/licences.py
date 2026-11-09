"""Local licence-acceptance record [Slice C].

``<home>/licences/accepted.json`` stores only *what* was accepted and *when*:
``catalog_id``, ``terms_sha256`` (hash of the licence terms shown), and
``accepted_at``. No user identity, no IP, no telemetry. If the terms of an entry
change, its hash changes and the user is asked again.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.checkpoints.catalog import CatalogEntry
from app.storage.atomic import atomic_write_json, read_json
from app.storage.locks import FileLock

SCHEMA_VERSION = 1


class LicenceStore:
    def __init__(self, home: Path) -> None:
        self._path = home / "licences" / "accepted.json"
        self._lock = home / "locks" / "licences.lock"

    @property
    def path(self) -> Path:
        return self._path

    def records(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        try:
            data = read_json(self._path)
        except (OSError, ValueError):
            return []
        items = data.get("accepted") if isinstance(data, dict) else None
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def is_accepted(self, entry: CatalogEntry) -> bool:
        return any(
            record.get("catalog_id") == entry.catalog_id
            and record.get("terms_sha256") == entry.terms_sha256
            for record in self.records()
        )

    def accept(self, entry: CatalogEntry) -> dict[str, Any]:
        record = {
            "catalog_id": entry.catalog_id,
            "terms_sha256": entry.terms_sha256,
            "accepted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with FileLock(self._lock):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            kept = [r for r in self.records() if r.get("catalog_id") != entry.catalog_id]
            atomic_write_json(
                self._path, {"schema_version": SCHEMA_VERSION, "accepted": [*kept, record]}
            )
        return record
