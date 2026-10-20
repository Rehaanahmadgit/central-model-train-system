"""Checkpoint catalog [doc 07 §2].

``catalog.yaml`` is hand-maintained and ships with the image as defaults; an
operator can add/replace entries with ``checkpoints/catalog.override.yaml``
inside the volume without rebuilding the image [doc 14 §5].

Entries whose ``sha256`` is still ``TODO(user)`` are **unverified**: the
resolver refuses to download them because sha256 verification is mandatory
[doc 10 §4]. Nothing here invents hashes or URLs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CATALOG = Path(__file__).with_name("default_catalog.yaml")
PLACEHOLDER_PREFIX = "TODO(user)"
ALLOWED_SOURCE_TYPES = ("url", "file")


def is_placeholder_sha(value: str | None) -> bool:
    if not value:
        return True
    text = value.strip()
    return text.startswith(PLACEHOLDER_PREFIX) or len(text) != 64


@dataclass(frozen=True)
class CatalogEntry:
    """One pretrained-weight offer. ``catalog_id`` is ``name@version``."""

    name: str
    version: str
    model_family: str
    task_types: tuple[str, ...]
    pretrained_on: str
    source_type: str
    source_uri: str
    sha256: str | None
    size_bytes: int
    license: str = ""
    min_vram_gb: float = 0.0
    filename: str = ""
    is_active: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def catalog_id(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def verified(self) -> bool:
        """A real 64-hex hash is present, so a download can be checked [doc 10 §4]."""
        return not is_placeholder_sha(self.sha256)

    @property
    def resolved_filename(self) -> str:
        if self.filename:
            return self.filename
        suffix = Path(self.source_uri).suffix or ".bin"
        return f"{self.name}{suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "name": self.name,
            "version": self.version,
            "model_family": self.model_family,
            "task_types": list(self.task_types),
            "pretrained_on": self.pretrained_on,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "license": self.license,
            "min_vram_gb": self.min_vram_gb,
            "filename": self.resolved_filename,
            "is_active": self.is_active,
            "notes": self.notes,
            "verified": self.verified,
        }


class CatalogError(ValueError):
    """Raised for malformed catalog files."""


_KNOWN_KEYS = {
    "name",
    "version",
    "model_family",
    "task_types",
    "pretrained_on",
    "source_type",
    "source_uri",
    "sha256",
    "size_bytes",
    "license",
    "min_vram_gb",
    "filename",
    "is_active",
    "notes",
}


def _parse_entry(item: dict[str, Any], origin: str) -> CatalogEntry:
    missing = [key for key in ("name", "version") if not item.get(key)]
    if missing:
        raise CatalogError(f"{origin}: entry is missing {missing}")
    task_types = item.get("task_types") or []
    if not isinstance(task_types, list):
        raise CatalogError(f"{origin}: task_types must be a list")
    source_type = str(item.get("source_type") or "url")
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise CatalogError(f"{origin}: source_type must be one of {ALLOWED_SOURCE_TYPES}")
    return CatalogEntry(
        name=str(item["name"]),
        version=str(item["version"]),
        model_family=str(item.get("model_family") or ""),
        task_types=tuple(str(value) for value in task_types),
        pretrained_on=str(item.get("pretrained_on") or ""),
        source_type=source_type,
        source_uri=str(item.get("source_uri") or ""),
        sha256=str(item["sha256"]) if item.get("sha256") else None,
        size_bytes=int(item.get("size_bytes") or 0),
        license=str(item.get("license") or ""),
        min_vram_gb=float(item.get("min_vram_gb") or 0.0),
        filename=str(item.get("filename") or ""),
        is_active=bool(item.get("is_active", True)),
        notes=str(item.get("notes") or ""),
        extra={key: value for key, value in item.items() if key not in _KNOWN_KEYS},
    )


def _read_yaml(path: Path) -> list[dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CatalogError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CatalogError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        return []
    if not isinstance(data, list):
        raise CatalogError(f"{path} must contain a YAML list of entries")
    return [item for item in data if isinstance(item, dict)]


def catalog_paths(home: Path) -> tuple[Path, Path]:
    """``(volume catalog, operator override)`` inside the workspace [doc 14 §5]."""
    root = home / "checkpoints"
    return root / "catalog.yaml", root / "catalog.override.yaml"


def ensure_catalog(home: Path) -> Path:
    """Copy the shipped defaults into the workspace once [doc 14 §5]."""
    target, _ = catalog_paths(home)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_CATALOG.is_file():
        shutil.copyfile(DEFAULT_CATALOG, target)
    else:  # pragma: no cover - shipping guard
        target.write_text("[]\n", encoding="utf-8")
    return target


def load_catalog(home: Path) -> list[CatalogEntry]:
    """Defaults + override, override wins on the same ``name@version``."""
    base_path, override_path = catalog_paths(home)
    if not base_path.is_file():
        ensure_catalog(home)
    merged: dict[str, CatalogEntry] = {}
    for path in (base_path, override_path):
        if not path.is_file():
            continue
        for item in _read_yaml(path):
            entry = _parse_entry(item, path.name)
            merged[entry.catalog_id] = entry
    return [merged[key] for key in sorted(merged)]


def find_entry(entries: list[CatalogEntry], catalog_id: str) -> CatalogEntry | None:
    for entry in entries:
        if entry.catalog_id == catalog_id:
            return entry
    return None
