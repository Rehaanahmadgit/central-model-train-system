"""Atomic-write tests [doc 11 §3b: "Atomic write interrupted ... old file intact"]."""

from pathlib import Path

import pytest

from app.storage.atomic import atomic_write_bytes, atomic_write_json, atomic_write_text, read_json


def test_atomic_write_text_replaces(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    atomic_write_text(path, "old")
    atomic_write_text(path, "new")
    assert path.read_text() == "new"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    atomic_write_json(path, {"a": [1, 2], "b": {"c": True}})
    assert read_json(path) == {"a": [1, 2], "b": {"c": True}}


def test_atomic_write_bytes(tmp_path: Path) -> None:
    path = tmp_path / "b.bin"
    atomic_write_bytes(path, b"\x00\x01\x02")
    assert path.read_bytes() == b"\x00\x01\x02"


def test_interrupted_replace_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a crash between the temp write and os.replace [doc 11 §3b].
    path = tmp_path / "x.json"
    atomic_write_text(path, "old")

    def crash_on_replace(src: object, dst: object) -> None:
        raise RuntimeError("crash before replace")

    monkeypatch.setattr("app.storage.atomic.os.replace", crash_on_replace)
    with pytest.raises(RuntimeError):
        atomic_write_text(path, "new")
    monkeypatch.undo()

    assert path.read_text() == "old"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_into_missing_directory_raises(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "x.json"
    with pytest.raises(FileNotFoundError):
        atomic_write_text(path, "nope")
