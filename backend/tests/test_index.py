"""In-memory index tests [doc 03 §2.5, doc 04 §3 rule 8]."""

from pathlib import Path

from app.storage.atomic import atomic_write_json, read_json
from app.storage.index import FileIndex


def _make_run(home: Path, run_id: str, *, corrupt: bool = False) -> None:
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    if corrupt:
        path.write_text("not json")
    else:
        atomic_write_json(path, {"id": run_id, "schema_version": 1})


def _loader(path: Path) -> tuple[str, object] | None:
    return path.parent.name, read_json(path)


def test_rebuild_collects_valid_files(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_a")
    _make_run(tmp_path, "run_b")
    index = FileIndex(tmp_path, ["runs/*/run.json"], _loader)
    index.rebuild()
    assert len(index) == 2
    assert index.get("run_a") == {"id": "run_a", "schema_version": 1}
    assert index.get("run_b") == {"id": "run_b", "schema_version": 1}


def test_rebuild_quarantines_corrupt_file(tmp_path: Path) -> None:
    # Corrupt run.json → *.corrupt, others unaffected [doc 03 §8].
    _make_run(tmp_path, "run_good")
    _make_run(tmp_path, "run_bad", corrupt=True)
    index = FileIndex(tmp_path, ["runs/*/run.json"], _loader)
    index.rebuild()
    assert len(index) == 1
    assert index.get("run_good") is not None
    assert not (tmp_path / "runs" / "run_bad" / "run.json").exists()
    assert (tmp_path / "runs" / "run_bad" / "run.json.corrupt").exists()


def test_rebuild_skips_quarantined_files(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_bad", corrupt=True)
    index = FileIndex(tmp_path, ["runs/*/run.json"], _loader)
    index.rebuild()  # quarantines
    index.rebuild()  # must not try to load the .corrupt file
    assert len(index) == 0


def test_upsert_get_remove(tmp_path: Path) -> None:
    index = FileIndex(tmp_path, [], _loader)
    index.upsert("k", {"v": 1})
    assert index.get("k") == {"v": 1}
    index.remove("k")
    assert index.get("k") is None
    assert index.items() == []
