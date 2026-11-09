"""Pretrained checkpoint loading tests [FR-6, doc 07 §5, PRD §10.3].

Prove the full chain on CPU with a synthetic weight file:
resolve -> download -> sha256 verify -> cache reuse -> the CNN actually loads
the backbone weights and adapts the head. Unit tests assert real tensor
values, not paths or metadata.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from fixtures.make_datasets import build_classification, materialize, to_zip
from helpers import make_pretrained_checkpoint, upload_dataset

from app.core.device import resolve_device
from app.training.plugins.cnn import CNNTrainer
from app.verification.manifest import build_manifest, write_manifest

# --- unit-level: the trainer really consumes the weights --------------------


def _prepare(trainer: CNNTrainer, root: Path, *, checkpoint_path: Path | None) -> None:
    """Build a manifest + fake RunContext and run prepare() [mirrors worker_main]."""
    summary = build_manifest(root, "image_folder", sample_cap=100)
    write_manifest(root / "manifest.parquet", summary.rows)
    logs: list[str] = []
    events: list[tuple[str, dict]] = []
    ctx = SimpleNamespace(
        dataset_root=root,
        manifest_path=root / "manifest.parquet",
        config={"epochs": 1, "batch_size": 2, "imgsz": 32, "patience": 0, "min_delta": 0.0},
        seed=42,
        device=resolve_device("cpu"),
        output_dir=root / "artifacts",
        checkpoint_path=checkpoint_path,
        emit_log=lambda level, message: logs.append(f"{level} {message}"),
        emit_event=lambda event_type, payload: events.append((event_type, payload)),
    )
    trainer.prepare(ctx)
    trainer._logs = logs  # type: ignore[attr-defined]  # captured for assertions
    trainer._events = events  # type: ignore[attr-defined]


def _backbone_mean(trainer: CNNTrainer) -> float:
    return float(trainer._model.conv1.weight.mean().item())  # type: ignore[union-attr]


def test_cnn_loads_pretrained_backbone_and_reaadapts_head(tmp_path: Path) -> None:
    """Backbone weights are actually applied; head is re-initialized for a
    different class count. Asserts real tensor values, not paths."""
    root = tmp_path / "ds"
    materialize(
        root / "raw", build_classification(per_class=3, classes=("cat", "dog"), size=(32, 32))
    )
    checkpoint = tmp_path / "pretrained.ckpt"
    make_pretrained_checkpoint(checkpoint, num_classes=5, fill=0.5)

    trainer = CNNTrainer()
    _prepare(trainer, root / "raw", checkpoint_path=checkpoint)

    # Backbone conv weights == 0.5 (the distinctive synthetic fill).
    assert _backbone_mean(trainer) == pytest.approx(0.5, abs=1e-6)
    # Head was re-initialized for 2 classes, not reused from the 5-class checkpoint.
    assert trainer._model.fc.out_features == 2  # type: ignore[union-attr]
    assert trainer._model.fc.weight.shape[0] == 2  # type: ignore[union-attr]
    assert any("classification head re-initialized" in log for log in trainer._logs)  # type: ignore[attr-defined]


def test_cnn_reuses_head_when_class_count_matches(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    materialize(
        root / "raw", build_classification(per_class=3, classes=("cat", "dog"), size=(32, 32))
    )
    checkpoint = tmp_path / "pretrained.ckpt"
    make_pretrained_checkpoint(checkpoint, num_classes=2, fill=0.5)

    trainer = CNNTrainer()
    _prepare(trainer, root / "raw", checkpoint_path=checkpoint)

    assert _backbone_mean(trainer) == pytest.approx(0.5, abs=1e-6)
    # Head reused: fc weights also carry the distinctive fill.
    assert float(trainer._model.fc.weight.mean().item()) == pytest.approx(0.5, abs=1e-6)  # type: ignore[union-attr]
    assert any("classification head reused" in log for log in trainer._logs)  # type: ignore[attr-defined]


def test_cnn_rejects_incompatible_backbone(tmp_path: Path) -> None:
    """A channel mismatch in the backbone fails loudly, never silently skips."""
    root = tmp_path / "ds"
    materialize(
        root / "raw", build_classification(per_class=3, classes=("cat", "dog"), size=(32, 32))
    )
    checkpoint = tmp_path / "pretrained.ckpt"
    make_pretrained_checkpoint(checkpoint, num_classes=2, channels=1)

    trainer = CNNTrainer()
    with pytest.raises(ValueError, match="backbone mismatch"):
        _prepare(trainer, root / "raw", checkpoint_path=checkpoint)


def test_cnn_rejects_garbage_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    materialize(
        root / "raw", build_classification(per_class=3, classes=("cat", "dog"), size=(32, 32))
    )
    checkpoint = tmp_path / "garbage.ckpt"
    checkpoint.write_bytes(b"definitely not a torch checkpoint")

    trainer = CNNTrainer()
    with pytest.raises(ValueError, match="could not load pretrained checkpoint"):
        _prepare(trainer, root / "raw", checkpoint_path=checkpoint)


def test_cnn_trains_from_scratch_when_no_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    materialize(
        root / "raw", build_classification(per_class=3, classes=("cat", "dog"), size=(32, 32))
    )
    trainer = CNNTrainer()
    _prepare(trainer, root / "raw", checkpoint_path=None)
    # Fresh init: backbone weights are NOT the synthetic 0.5 fill.
    assert _backbone_mean(trainer) != pytest.approx(0.5, abs=1e-6)
    assert not any("loaded pretrained" in log for log in trainer._logs)  # type: ignore[attr-defined]


# --- integration: resolve -> download -> verify -> cache -> load -------------

CATALOG_ID = "tiny-cnn@1.0"


def _install_catalog(home: Path, source: Path) -> str:
    """Write a catalog entry with a REAL sha256 for the synthetic file."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (home / "checkpoints").mkdir(parents=True, exist_ok=True)
    (home / "checkpoints" / "catalog.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "name": "tiny-cnn",
                    "version": "1.0",
                    "model_family": "cnn",
                    "task_types": ["image_classification"],
                    "pretrained_on": "unit-test",
                    "source_type": "file",
                    "source_uri": f"file://{source}",
                    "sha256": digest,
                    "size_bytes": source.stat().st_size,
                    "license": "MIT",
                    "filename": "tiny-cnn.ckpt",
                }
            ]
        ),
        encoding="utf-8",
    )
    from app.checkpoints.catalog import load_catalog
    from app.checkpoints.licences import LicenceStore

    for entry in load_catalog(home):  # the user accepted the licence in the UI
        LicenceStore(home).accept(entry)
    return digest


def _wait_verification(client: TestClient, verification_id: str) -> dict:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/verifications/{verification_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError("verification did not finish")


def _wait_run(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/runs/{run_id}").json()
        if body["state"] in ("completed", "failed", "stopped"):
            return body
        time.sleep(0.1)
    raise AssertionError("run did not finish")


@pytest.mark.slow
def test_pretrained_checkpoint_end_to_end(client: TestClient, home: Path) -> None:
    """Missing checkpoint is downloaded, verified, cached, and actually loaded;
    a second run reuses the cache with no second fetch. Run.json records the
    truthful source (download then cache)."""
    source = home / "source.ckpt"
    make_pretrained_checkpoint(source, num_classes=5, fill=0.5)
    _install_catalog(home, source)

    def run_once() -> dict:
        completed = upload_dataset(
            client, "cls.zip", to_zip(build_classification(per_class=3, size=(32, 32)))
        )
        verification = client.post(
            "/api/v1/verifications",
            json={
                "dataset_hash": completed["dataset_hash"],
                "model_family": "cnn",
                "preset": "dev",
            },
        ).json()
        _wait_verification(client, verification["id"])
        started = client.post(
            "/api/v1/runs",
            json={
                "verification_id": verification["id"],
                "name": "pretrained-run",
                "config": {"epochs": 1, "batch_size": 2},
                "checkpoint_catalog_id": CATALOG_ID,
            },
        ).json()
        return _wait_run(client, started["id"])

    first = run_once()
    assert first["state"] == "completed", first
    assert first["checkpoint"]["catalog_id"] == CATALOG_ID
    assert first["checkpoint"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert first["checkpoint"]["source"] == "download"
    assert Path(first["checkpoint"]["path"]).is_file()

    # The trainer actually loaded the weights (log + event), and evaluation ran.
    events = list(_read_jsonl(home / "runs" / first["id"] / "events.jsonl"))
    assert any(
        "loaded pretrained checkpoint" in e["data"]["message"] for e in events if e["type"] == "log"
    )
    assert any(
        e["type"] == "event" and e["data"].get("name") == "pretrained_loaded" for e in events
    )
    assert (home / "runs" / first["id"] / "artifacts" / "evaluation.json").is_file()

    # Second run with the same catalog id: cache hit, no re-download.
    second = run_once()
    assert second["state"] == "completed", second
    assert second["checkpoint"]["source"] == "cache"
    assert second["checkpoint"]["sha256"] == first["checkpoint"]["sha256"]


@pytest.mark.slow
def test_incompatible_checkpoint_fails_safely(client: TestClient, home: Path) -> None:
    """A checkpoint whose backbone does not match must fail the run with a
    clear DATA_ERROR, not silently train from scratch."""
    source = home / "bad.ckpt"
    make_pretrained_checkpoint(source, num_classes=2, channels=1)  # 1-channel backbone
    _install_catalog(home, source)

    completed = upload_dataset(
        client, "cls.zip", to_zip(build_classification(per_class=3, size=(32, 32)))
    )
    verification = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "cnn", "preset": "dev"},
    ).json()
    _wait_verification(client, verification["id"])
    started = client.post(
        "/api/v1/runs",
        json={
            "verification_id": verification["id"],
            "name": "bad-pretrained-run",
            "config": {"epochs": 1, "batch_size": 2},
            "checkpoint_catalog_id": CATALOG_ID,
        },
    ).json()
    body = _wait_run(client, started["id"])
    assert body["state"] == "failed"
    assert body["failure"]["reason"] in ("DATA_ERROR", "UNKNOWN")
    events = list(_read_jsonl(home / "runs" / started["id"] / "events.jsonl"))
    assert any(
        "backbone mismatch" in str(e["data"].get("message")) for e in events if e["type"] == "log"
    )


def _read_jsonl(path: Path):
    from app.storage.jsonl import read_jsonl

    return read_jsonl(path)
