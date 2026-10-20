"""Dataset storage tests: content hashing, meta, in-use check, deletion [doc 04]."""

import shutil
from datetime import datetime
from pathlib import Path

from app.storage.atomic import atomic_write_json
from app.storage.dataset import (
    dataset_hash_from_dir,
    dataset_in_use,
    delete_dataset_files,
    read_dataset_meta,
    write_dataset_meta,
)
from app.storage.jsonl import read_jsonl


def _seed(root: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_hash_is_content_addressed(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    _seed(first, {"x.txt": b"hello", "sub/y.txt": b"world"})
    _seed(second, {"x.txt": b"hello", "sub/y.txt": b"world"})
    assert dataset_hash_from_dir(first) == dataset_hash_from_dir(second)


def test_hash_changes_with_content(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    _seed(first, {"x.txt": b"hello"})
    _seed(second, {"x.txt": b"hello!"})
    assert dataset_hash_from_dir(first) != dataset_hash_from_dir(second)


def test_hash_ignores_directory_order_and_names(tmp_path: Path) -> None:
    first = tmp_path / "a"
    renamed = tmp_path / "b"
    same_names = tmp_path / "c"
    _seed(first, {"z.txt": b"same"})
    _seed(renamed, {"a.txt": b"same"})
    # File names are part of the manifest, so different names hash differently.
    assert dataset_hash_from_dir(first) != dataset_hash_from_dir(renamed)
    _seed(same_names, {"z.txt": b"same"})
    # Same relative names + content → same hash regardless of directory order.
    assert dataset_hash_from_dir(first) == dataset_hash_from_dir(same_names)


def test_write_and_read_meta(home: Path) -> None:
    now = datetime(2026, 10, 17, 12, 0, 0)
    written = write_dataset_meta(
        home,
        dataset_hash="a" * 64,
        name="data.csv",
        size_bytes=123,
        file_count=1,
        source_filename="data.csv",
        project_id="proj_1",
        now=now,
    )
    read = read_dataset_meta(home, "a" * 64)
    assert read == written
    assert read["format_detected"] is None  # verification writes it later (D16)
    assert read["schema_version"] == 1
    assert read["project_id"] == "proj_1"


def test_dataset_in_use_only_for_active_runs(home: Path) -> None:
    dataset_hash = "b" * 64
    for state in ("running", "queued", "paused", "completed", "failed"):
        run_dir = home / "runs" / f"run_{state}"
        run_dir.mkdir(parents=True)
        atomic_write_json(
            run_dir / "run.json",
            {
                "id": f"run_{state}",
                "state": state,
                "dataset_hash": dataset_hash,
                "schema_version": 1,
            },
        )
    assert dataset_in_use(home, dataset_hash) is True  # running/queued/paused
    # Terminated runs no longer block deletion.
    for state in ("running", "queued", "paused"):
        shutil.rmtree(home / "runs" / f"run_{state}")
    assert dataset_in_use(home, dataset_hash) is False


def test_delete_dataset_files_and_audit(home: Path) -> None:
    dataset_hash = "c" * 64
    raw = home / "datasets" / dataset_hash / "raw"
    raw.mkdir(parents=True)
    (raw / "x.csv").write_text("a,b\n")
    write_dataset_meta(
        home, dataset_hash=dataset_hash, name="x.csv", size_bytes=4, file_count=1,
        source_filename="x.csv",
    )
    delete_dataset_files(home, dataset_hash)
    assert not (home / "datasets" / dataset_hash).exists()
    records = list(read_jsonl(next((home / "audit").glob("audit-*.jsonl"))))
    assert any(r["action"] == "dataset.deleted" and r["entity_id"] == dataset_hash for r in records)
