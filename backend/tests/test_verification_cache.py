"""S6 tests: cache, manifest, derived facts, recommendations [doc 06 §6, doc 11 §2-§3]."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml
from fixtures.make_datasets import (
    SPECS,
    build_classification,
    build_tabular,
    build_yolo,
    materialize,
    spec_model_family,
    to_zip,
)
from helpers import upload_dataset

from app.core.device import DeviceInfo, resolve_device
from app.verification.cache import cache_key, read_pointer
from app.verification.manifest import build_manifest
from app.verification.recommend import (
    CatalogEntry,
    load_catalog,
    rank_checkpoints,
    recommend_hparams,
    size_bucket,
)
from app.verification.runner import run_pipeline
from app.verification.tasks import detect_task

CPU = DeviceInfo(
    kind="cpu",
    index=None,
    name="CPU",
    torch_installed=False,
    cuda_available=False,
    reason="test",
)


def _run(root: Path, model_family: str = "rtdetr", sample_cap: int = 500) -> dict:
    return run_pipeline(
        root,
        "a" * 64,
        model_family,
        None,
        sample_cap=sample_cap,
        check_timeout_seconds=60,
        progress=lambda event: None,
    )


# --- dataset spec table (doc 11 §2) ----------------------------------------


@pytest.mark.parametrize("name", sorted(SPECS))
def test_dataset_specs_match_expected_outcome(name: str, tmp_path: Path) -> None:
    """Every synthetic dataset has a documented, asserted outcome [doc 11 §2]."""
    builder, expected = SPECS[name]
    raw = materialize(tmp_path / "raw", builder())
    outcome = _run(raw, spec_model_family(name))
    assert outcome["status"] == expected, [
        (check["name"], check["status"], check["message"])
        for check in outcome["checks"]
        if check["status"] in ("fail", "warn")
    ]


# --- manifest ---------------------------------------------------------------


def test_manifest_parquet_has_stable_schema(tmp_path: Path) -> None:
    from app.verification.manifest import write_manifest

    raw = materialize(tmp_path / "raw", build_yolo())
    summary = build_manifest(raw, "yolo", sample_cap=500)
    assert len(summary.rows) == 24
    assert summary.class_names == ["0", "1"]
    assert summary.image_stats["max_wh"] == [16, 16]
    assert summary.object_stats["boxes"] == 24

    path = tmp_path / "manifest.parquet"
    write_manifest(path, summary.rows)
    table = pq.read_table(path)
    assert table.column_names == [
        "path",
        "split",
        "label",
        "class_id",
        "num_boxes",
        "boxes",
        "width",
        "height",
        "color_mode",
        "file_sha256",
        "annotation_path",
    ]
    assert table.num_rows == 24
    assert table.column("split").to_pylist()[0] == "unspecified"


def test_manifest_detects_splits_and_boxes(tmp_path: Path) -> None:
    raw = materialize(tmp_path / "raw", build_yolo(split=True))
    summary = build_manifest(raw, "yolo", sample_cap=500)
    assert set(summary.num_samples) == {"train", "val"}
    first_with_boxes = next(row for row in summary.rows if row["num_boxes"])
    assert json.loads(first_with_boxes["boxes"])[0] == [0.25, 0.25, 0.5, 0.5]


def test_manifest_tabular_schema(tmp_path: Path) -> None:
    raw = materialize(tmp_path / "raw", build_tabular())
    summary = build_manifest(raw, "tabular", sample_cap=500)
    assert summary.tabular_schema["target"] == "target"
    assert summary.tabular_schema["rows"] == 30
    numeric = {column["name"]: column["numeric"] for column in summary.tabular_schema["columns"]}
    assert numeric["feature_b"] is True
    assert numeric["feature_a"] is True


# --- task detection ---------------------------------------------------------


def test_task_detection_reasons_are_explainable(tmp_path: Path) -> None:
    yolo_root = materialize(tmp_path / "yolo", build_yolo())
    yolo_summary = build_manifest(yolo_root, "yolo", sample_cap=100)
    task, reason = detect_task(yolo_root, "yolo", yolo_summary)
    assert task == "object_detection"
    assert "bounding boxes" in reason

    cls_root = materialize(tmp_path / "cls", build_classification())
    cls_summary = build_manifest(cls_root, "image_folder", sample_cap=100)
    task, reason = detect_task(cls_root, "image_folder", cls_summary)
    assert task == "image_classification"
    assert "folder-per-class" in reason

    tab_root = materialize(tmp_path / "tab", build_tabular())
    tab_summary = build_manifest(tab_root, "tabular", sample_cap=100)
    task, reason = detect_task(tab_root, "tabular", tab_summary)
    assert task == "tabular_classification"
    assert "distinct values" in reason


# --- recommendations --------------------------------------------------------


def test_size_buckets() -> None:
    assert size_bucket(999) == "tiny"
    assert size_bucket(1_000) == "small"
    assert size_bucket(10_000) == "medium"
    assert size_bucket(500_000) == "large"


def test_recommend_hparams_shrinks_imgsz_for_small_images(tmp_path: Path) -> None:
    raw = materialize(tmp_path / "raw", build_yolo())
    summary = build_manifest(raw, "yolo", sample_cap=100)
    hparams = recommend_hparams(
        summary, task_type="object_detection", model_family="rtdetr", device=CPU
    )
    assert hparams["imgsz"] == 32  # rounded up from 16 px, multiple of 32
    assert hparams["freeze_backbone"] is True  # tiny dataset
    assert hparams["device"] == "cpu"
    assert hparams["strategy"].startswith("freeze backbone")


def test_recommend_hparams_uses_gpu_batch() -> None:
    from app.verification.manifest import ManifestSummary

    gpu = DeviceInfo("cuda", 0, "NVIDIA Test", True, True, "test")
    hparams = recommend_hparams(
        ManifestSummary(rows=[{}] * 20_000),
        task_type="image_classification",
        model_family="resnet",
        device=gpu,
    )
    assert hparams["batch_size"] == 16
    assert hparams["freeze_backbone"] is False
    assert hparams["device"] == "cuda"


def test_catalog_missing_returns_empty_and_no_invented_ids(tmp_path: Path) -> None:
    assert load_catalog(tmp_path / "catalog.yaml") == []
    raw = materialize(tmp_path / "raw", build_yolo())
    outcome = _run(raw)
    assert outcome["derived"]["recommended_checkpoints"] == []
    assert outcome["derived"]["checkpoint_catalog_configured"] is False


def test_catalog_ranking_follows_doc_rules(tmp_path: Path) -> None:
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "rtdetr-l",
                    "version": "1.0",
                    "model_family": "rtdetr",
                    "task_types": ["object_detection"],
                    "pretrained_on": "coco",
                    "sha256": "TODO(user)",
                    "size_bytes": 128_000_000,
                },
                {
                    "name": "resnet50",
                    "version": "imagenet-1k",
                    "model_family": "resnet",
                    "task_types": ["image_classification"],
                    "pretrained_on": "imagenet",
                },
            ]
        ),
        encoding="utf-8",
    )
    catalog = load_catalog(catalog_file)
    assert {entry.catalog_id for entry in catalog} == {"rtdetr-l@1.0", "resnet50@imagenet-1k"}
    ranked = rank_checkpoints(
        catalog,
        task_type="object_detection",
        model_family="rtdetr",
        bucket="small",
        device=CPU,
    )
    assert [item["catalog_id"] for item in ranked] == ["rtdetr-l@1.0"]
    assert "exact task match" in ranked[0]["reason"]


def test_catalog_present_flags_config_in_report(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "checkpoints").mkdir(parents=True)
    (home / "checkpoints" / "catalog.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "name": "rtdetr-l",
                    "version": "1.0",
                    "model_family": "rtdetr",
                    "task_types": ["object_detection"],
                    "pretrained_on": "coco",
                }
            ]
        ),
        encoding="utf-8",
    )
    raw = materialize(tmp_path / "raw", build_yolo())
    outcome = run_pipeline(
        raw,
        "a" * 64,
        "rtdetr",
        None,
        sample_cap=100,
        check_timeout_seconds=30,
        progress=lambda event: None,
        home=home,
    )
    assert outcome["derived"]["checkpoint_catalog_configured"] is True
    assert outcome["derived"]["recommended_checkpoints"][0]["catalog_id"] == "rtdetr-l@1.0"


def test_catalog_entry_shape() -> None:
    entry = CatalogEntry(
        catalog_id="x@1",
        model_family="resnet",
        task_types=("image_classification",),
        pretrained_on="",
    )
    assert entry.is_active is True


# --- device abstraction -----------------------------------------------------


def test_resolve_device_never_claims_cuda_without_torch() -> None:
    info = resolve_device("auto")
    if not info.cuda_available:
        assert info.kind == "cpu"
        assert "torch" in info.reason.lower() or "cuda" in info.reason.lower()
    assert resolve_device("cpu").kind == "cpu"


# --- cache ------------------------------------------------------------------


def test_cache_key_is_documented_hash() -> None:
    key = cache_key("a" * 64, "rtdetr", "1.0.0")
    assert len(key) == 64
    assert key == cache_key("a" * 64, "rtdetr", "1.0.0")
    assert key != cache_key("a" * 64, "yolo", "1.0.0")
    assert key != cache_key("a" * 64, "rtdetr", "1.0.1")


def _wait(client, verification_id: str, timeout: float = 30.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/verifications/{verification_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError("verification did not finish")


def test_second_request_hits_cache_and_skips_recompute(client, home: Path) -> None:
    """Re-verifying the same dataset + model returns the stored report [US-06]."""
    completed = upload_dataset(client, "yolo.zip", to_zip(build_yolo()))
    first = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    )
    assert first.status_code == 202
    assert first.json()["cached"] is False
    first_id = first.json()["id"]
    _wait(client, first_id)

    pointer = read_pointer(home, cache_key(completed["dataset_hash"], "rtdetr", "1.0.0"))
    assert pointer is not None
    assert pointer["verification_id"] == first_id

    second = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    )
    assert second.status_code == 202
    body = second.json()
    assert body["cached"] is True
    assert body["id"] == first_id
    assert body["status"] == "passed_with_warnings"
    # No second verification directory was created.
    assert len(list((home / "verifications").iterdir())) == 1


def test_cache_miss_for_different_model_family(client) -> None:
    completed = upload_dataset(client, "yolo.zip", to_zip(build_yolo()))
    first = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    ).json()
    _wait(client, first["id"])
    second = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "yolo"},
    ).json()
    assert second["cached"] is False
    assert second["id"] != first["id"]
