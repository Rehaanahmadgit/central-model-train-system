"""Atomic file writes [doc 04 §3 rule 1].

Write to a temp file in the same directory, fsync, then ``os.replace``.
Never edit a file in place. On any failure the previous file stays intact
and the temp file is cleaned up.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_replace(path: Path, data: bytes) -> None:
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        # Best-effort directory fsync so the rename itself survives a crash.
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            pass
        else:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:  # noqa: B036 - clean up on KeyboardInterrupt too; always re-raised
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_replace(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    _atomic_replace(path, data)


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    text = json.dumps(obj, indent=indent, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    """Read a JSON file; decode errors carry the file path for logging."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f"{path}: {exc.msg}", exc.doc, exc.pos) from None
