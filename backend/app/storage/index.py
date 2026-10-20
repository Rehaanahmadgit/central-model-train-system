"""Rebuildable in-memory index over JSON files [doc 03 §2.5, doc 04 §3 rule 8].

The index is a cache, never the source of truth; ``rebuild()`` re-scans the
workspace at startup or on demand. Corrupt files are quarantined to
``*.corrupt`` [doc 03 §8].
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A loader returns (key, value) or None to skip a file.
Loader = Callable[[Path], tuple[str, Any] | None]


def quarantine(path: Path) -> Path:
    """Rename a corrupt file to ``*.corrupt`` [doc 03 §8]."""
    target = Path(f"{path}.corrupt")
    path.replace(target)
    logger.warning("quarantined corrupt file: %s", path)
    return target


class FileIndex:
    """In-memory mapping rebuilt from files matching glob patterns."""

    def __init__(self, home: Path, patterns: list[str], load: Loader) -> None:
        self._home = home
        self._patterns = patterns
        self._load = load
        self._entries: dict[str, Any] = {}

    def rebuild(self) -> None:
        entries: dict[str, Any] = {}
        for pattern in self._patterns:
            for path in sorted(self._home.glob(pattern)):
                if not path.is_file() or path.name.endswith(".corrupt"):
                    continue
                try:
                    loaded = self._load(path)
                except (OSError, ValueError):
                    logger.exception("corrupt file during index rebuild: %s", path)
                    quarantine(path)
                    continue
                if loaded is None:
                    continue
                key, value = loaded
                entries[key] = value
        self._entries = entries

    def upsert(self, key: str, value: Any) -> None:
        self._entries[key] = value

    def get(self, key: str) -> Any | None:
        return self._entries.get(key)

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)

    def items(self) -> list[tuple[str, Any]]:
        return list(self._entries.items())

    def __len__(self) -> int:
        return len(self._entries)
