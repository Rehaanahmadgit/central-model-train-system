"""Cross-process file locks via ``fcntl.flock`` [doc 04 §3 rule 4].

The kernel releases a flock automatically when the owning process dies,
which is what makes run state safe across crashes. Linux only — TrainHub
targets Linux servers [doc 14 §8].
"""

import fcntl
import time
from pathlib import Path
from types import TracebackType
from typing import TextIO


class LockTimeout(TimeoutError):
    """Raised when a lock cannot be acquired within the configured timeout."""


class FileLock:
    """Exclusive or shared advisory lock on a lock file."""

    def __init__(self, path: Path, *, timeout: float | None = None, shared: bool = False) -> None:
        self._path = path
        self._timeout = timeout
        self._shared = shared
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+", encoding="utf-8")
        operation = fcntl.LOCK_SH if self._shared else fcntl.LOCK_EX
        deadline = None if self._timeout is None else time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
            except OSError:
                if deadline is not None and time.monotonic() >= deadline:
                    handle.close()
                    raise LockTimeout(f"could not acquire lock {self._path}") from None
                time.sleep(0.05)
            else:
                self._handle = handle
                return

    def release(self) -> None:
        handle = self._handle
        if handle is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            self._handle = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
