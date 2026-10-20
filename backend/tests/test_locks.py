"""Cross-process lock tests [doc 04 §3 rule 4, doc 11 §3b]."""

import os
import sys
from pathlib import Path

import pytest

from app.storage.jsonl import JsonlAppender, read_jsonl
from app.storage.locks import FileLock, LockTimeout


def test_exclusive_lock_blocks_second_holder(tmp_path: Path) -> None:
    path = tmp_path / "locks" / "test.lock"
    first = FileLock(path)
    first.acquire()
    try:
        second = FileLock(path, timeout=0.1)
        with pytest.raises(LockTimeout):
            second.acquire()
    finally:
        first.release()


def test_lock_released_after_context(tmp_path: Path) -> None:
    path = tmp_path / "locks" / "test.lock"
    with FileLock(path):
        pass
    # Reacquiring immediately must succeed.
    with FileLock(path, timeout=0.1):
        pass


def _child_write_block(home: Path, marker: str) -> None:
    # Runs in a forked child; never returns into pytest.
    lock = FileLock(home / "locks" / "shared.lock")
    with lock:
        with JsonlAppender(home / "shared.jsonl") as appender:
            for i in range(100):
                appender.append({"marker": marker, "i": i})
    os._exit(0)


@pytest.mark.skipif(sys.platform != "linux", reason="fork-based, Linux only [doc 11 §3b]")
def test_per_run_lock_prevents_interleaving(tmp_path: Path) -> None:
    # Two processes write blocks to the same file under the same lock;
    # the final file must be two contiguous blocks, never interleaved.
    children = [os.fork() for _ in range(2)]
    if children[0] == 0:
        _child_write_block(tmp_path, "A")
    if children[1] == 0:
        _child_write_block(tmp_path, "B")
    for pid in children:
        os.waitpid(pid, 0)

    records = list(read_jsonl(tmp_path / "shared.jsonl"))
    assert len(records) == 200
    markers = [record["marker"] for record in records]
    assert markers.count("A") == 100
    assert markers.count("B") == 100
    transitions = sum(1 for prev, cur in zip(markers, markers[1:], strict=False) if prev != cur)
    assert transitions == 1  # exactly two contiguous blocks


@pytest.mark.skipif(sys.platform != "linux", reason="fork-based, Linux only [doc 11 §3b]")
def test_lock_auto_released_when_process_dies(tmp_path: Path) -> None:
    # flock is released by the kernel when the holder dies [doc 04 §3 rule 4].
    path = tmp_path / "locks" / "test.lock"
    pid = os.fork()
    if pid == 0:
        lock = FileLock(path)
        lock.acquire()
        os._exit(0)  # die while holding the lock
    os.waitpid(pid, 0)
    with FileLock(path, timeout=2.0):
        pass  # acquires => kernel released the lock
