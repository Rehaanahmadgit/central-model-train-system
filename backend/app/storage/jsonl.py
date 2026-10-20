"""Append-only JSONL [doc 04 §3 rule 2].

- open in append mode, one line per record
- flush on a short timer (~1 s) or every N lines
- fsync on explicit ``sync()`` (state changes) and on close
- a half-written final line is skipped by the reader and removed by
  ``repair_tail()`` [doc 04 §3 rule 6]
"""

import json
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any


class JsonlParseError(ValueError):
    """A JSONL line is not valid JSON and is not the (half-written) last line."""

    def __init__(self, path: Path, line_number: int, line: str) -> None:
        super().__init__(f"{path}: invalid JSON on line {line_number}: {line!r}")
        self.path = path
        self.line_number = line_number
        self.line = line


class JsonlAppender:
    """Single-writer append-only JSONL writer [doc 04 §3 rule 3]."""

    def __init__(
        self, path: Path, *, flush_lines: int = 1, flush_interval: float = 1.0
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._flush_lines = flush_lines
        self._flush_interval = flush_interval
        self._handle = path.open("a", encoding="utf-8")
        self._pending = 0
        self._last_flush = time.monotonic()
        self._closed = False

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise ValueError("appender is closed")
        line = json.dumps(dict(record), ensure_ascii=False) + "\n"
        self._handle.write(line)
        self._pending += 1
        now = time.monotonic()
        if self._pending >= self._flush_lines or now - self._last_flush >= self._flush_interval:
            self._handle.flush()
            self._pending = 0
            self._last_flush = now

    def sync(self) -> None:
        """Flush and fsync — at state changes and on shutdown [doc 04 §3 rule 2]."""
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending = 0
        self._last_flush = time.monotonic()

    def close(self) -> None:
        if not self._closed:
            try:
                self.sync()
            finally:
                self._handle.close()
                self._closed = True

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete records; a half-written final line is silently skipped.

    A malformed line that is *not* last raises :class:`JsonlParseError`.
    """
    pending: str | None = None
    pending_line = 0
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return
    with handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if pending is not None:
                raise JsonlParseError(path, pending_line, pending)
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                pending, pending_line = stripped, line_number
            else:
                yield record
    # `pending` here means the last line was half-written; skip it.


def repair_tail(path: Path) -> int:
    """Truncate a half-written final line; returns bytes removed [doc 04 §3 rule 6]."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return 0
    if not data or data.endswith(b"\n"):
        return 0
    last_newline = data.rfind(b"\n")
    tail = data[last_newline + 1 :] if last_newline >= 0 else data
    try:
        json.loads(tail.strip().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        keep = last_newline + 1 if last_newline >= 0 else 0
        with path.open("r+b") as handle:
            handle.truncate(keep)
        return len(data) - keep
    return 0
