"""Golden test for the RT-DETR training path [S1 part 1].

One fixed CPU dev_preset run is recorded in ``golden/rtdetr_dev_preset.json``;
this test fails, with a diff table, when a later change moves those numbers.
Re-record deliberately (and review the diff) with::

    TRAINHUB_UPDATE_GOLDEN=1 python -m pytest tests/test_rtdetr_golden.py -m slow

Tolerances (DECISIONS D129):
  * split membership, sample counts, prediction splits, sum(error_score): exact
  * train/loss: relative 1e-3
  * mAP50: absolute 0.02   (a box near IoU 0.5 can flip tp<->fp with float noise)
  * per-class AP: absolute 0.05

The fixture is 8 train / 4 val / 3 test with a class-specific blob, fixed config
(dev_preset, imgsz 128, batch 2, 2 epochs, seed 42) and CPU only. With two epochs
from a random head mAP50 stays 0.0 — record is honest about that; the loss curve,
the split membership and the prediction counts are what guard a refactor today.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fixtures.make_datasets import materialize
from PIL import Image, ImageDraw

from app.training.plugins.rtdetr import RTDetrTrainer
from app.verification.manifest import build_manifest, write_manifest

GOLDEN_PATH = Path(__file__).with_name("golden") / "rtdetr_dev_preset.json"
UPDATE = os.environ.get("TRAINHUB_UPDATE_GOLDEN") == "1"

CONFIG: dict[str, Any] = {
    "epochs": 2,
    "batch_size": 2,
    "imgsz": 128,
    "lr": 1e-4,
    "freeze_backbone": True,
    "amp": False,
    "dev_preset": True,
    "score_threshold": 0.25,
}
SEED = 42
CLASS_NAMES = ["cat", "dog"]

TOLERANCES = {
    "train_loss_rel": 1e-3,
    "map50_abs": 0.02,
    "per_class_abs": 0.05,
}


def _blob_png(class_id: int, index: int) -> bytes:
    """Same object at a class-specific position, so the loss has real signal."""
    image = Image.new("RGB", (64, 64), (20, 20, 24))
    draw = ImageDraw.Draw(image)
    cx, cy = ((0.3, 0.3), (0.7, 0.7))[class_id]
    color = ((220, 60, 40), (40, 120, 220))[class_id]
    draw.rectangle(
        [(cx - 0.2) * 64, (cy - 0.2) * 64, (cx + 0.2) * 64, (cy + 0.2) * 64], fill=color
    )
    draw.ellipse([index, index, index + 6, index + 6], fill=(240, 240, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fixture_entries() -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    index = 0
    for split, count in (("train", 8), ("val", 4), ("test", 3)):
        for _ in range(count):
            class_id = index % 2
            cx, cy = (0.3, 0.3) if class_id == 0 else (0.7, 0.7)
            entries[f"{split}/images/img{index}.png"] = _blob_png(class_id, index)
            entries[f"{split}/labels/img{index}.txt"] = (
                f"{class_id} {cx} {cy} 0.4 0.4\n".encode()
            )
            index += 1
    entries["data.yaml"] = b"names:\n  - cat\n  - dog\n"
    return entries


def _run_once(tmp_path: Path, *, seed: int = SEED) -> dict[str, Any]:
    """Train + evaluate the fixed config and return the numbers we care about."""
    raw = materialize(tmp_path / "raw", _fixture_entries())
    summary = build_manifest(raw, "yolo", sample_cap=500)
    manifest = tmp_path / "manifest.parquet"
    write_manifest(manifest, summary.rows)

    metrics: list[tuple[str, float, int]] = []
    ctx = SimpleNamespace(
        home=tmp_path,
        run_id="golden",
        dataset_hash="golden",
        model_family="rtdetr",
        task_type="object_detection",
        dataset_root=raw,
        manifest_path=manifest,
        facts={"classes": CLASS_NAMES},
        config=dict(CONFIG),
        output_dir=tmp_path / "artifacts",
        checkpoint_path=None,
        seed=seed,
        device=SimpleNamespace(is_gpu=False, kind="cpu"),
        emit_log=lambda level, message: None,
        emit_metric=lambda name, value, step, epoch, split: metrics.append(
            (name, float(value), int(epoch))
        ),
        emit_event=lambda *args, **kwargs: None,
        emit_system=lambda *args, **kwargs: None,
        save_checkpoint=lambda *args, **kwargs: None,
        should_pause=lambda: False,
        should_stop=lambda: False,
        wait_if_paused=lambda: None,
    )
    trainer = RTDetrTrainer()
    trainer.prepare(ctx)
    result = trainer.train(ctx)
    evaluation = trainer.evaluate(ctx)

    predictions = json.loads((tmp_path / "artifacts" / "predictions.json").read_text())
    splits = {
        "train": sorted(path.relative_to(raw).as_posix() for path, _ in trainer._train_samples),
        "val": sorted(path.relative_to(raw).as_posix() for path, _ in trainer._val_samples),
        "test": sorted(path.relative_to(raw).as_posix() for path, _ in trainer._test_samples),
    }
    epochs: dict[int, dict[str, float]] = {}
    for name, value, epoch in metrics:
        record = epochs.setdefault(epoch, {})
        if name == "train/loss":
            record["train_loss"] = value
        elif name == "val/mAP50":
            record["val_map50"] = value
    return {
        "config": dict(CONFIG),
        "seed": seed,
        "splits": splits,
        "epochs": [{"epoch": epoch, **values} for epoch, values in sorted(epochs.items())],
        "evaluation": {
            "val": evaluation["val"],
            "test": evaluation["test"],
        },
        "predictions": {
            "splits": predictions["splits"],
            "sum_error_score": sum(int(image["error_score"]) for image in predictions["images"]),
        },
        "best": result.best,
    }


def _close(actual: float, expected: float, *, rel: float = 0.0, abs_: float = 0.0) -> bool:
    return abs(actual - expected) <= max(abs_, rel * abs(expected), 1e-12)


def _compare(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Return human-readable diff rows; empty means the numbers match."""
    rows: list[str] = []
    for split, expected_paths in expected["splits"].items():
        got = actual["splits"].get(split)
        if got != expected_paths:
            rows.append(f"splits.{split}: {got!r} != {expected_paths!r} (exact)")
    if actual["predictions"]["splits"] != expected["predictions"]["splits"]:
        rows.append(
            f"predictions.splits: {actual['predictions']['splits']} != "
            f"{expected['predictions']['splits']} (exact)"
        )
    if actual["predictions"]["sum_error_score"] != expected["predictions"]["sum_error_score"]:
        rows.append(
            f"sum(error_score): {actual['predictions']['sum_error_score']} != "
            f"{expected['predictions']['sum_error_score']} (exact)"
        )
    if actual["best"].get("epoch") != expected["best"].get("epoch"):
        rows.append(
            f"best.epoch: {actual['best'].get('epoch')} != "
            f"{expected['best'].get('epoch')} (exact)"
        )

    for got, want in zip(actual["epochs"], expected["epochs"], strict=False):
        if got["epoch"] != want["epoch"]:
            rows.append(f"epoch order: {got['epoch']} != {want['epoch']} (exact)")
            continue
        if not _close(got["train_loss"], want["train_loss"], rel=TOLERANCES["train_loss_rel"]):
            rows.append(
                f"epoch {got['epoch']} train/loss: {got['train_loss']:.6f} != "
                f"{want['train_loss']:.6f} (rel {TOLERANCES['train_loss_rel']})"
            )
        if not _close(got["val_map50"], want["val_map50"], abs_=TOLERANCES["map50_abs"]):
            rows.append(
                f"epoch {got['epoch']} val/mAP50: {got['val_map50']:.4f} != "
                f"{want['val_map50']:.4f} (abs {TOLERANCES['map50_abs']})"
            )
    if len(actual["epochs"]) != len(expected["epochs"]):
        rows.append(f"epoch count: {len(actual['epochs'])} != {len(expected['epochs'])} (exact)")

    for split in ("val", "test"):
        got, want = actual["evaluation"][split], expected["evaluation"][split]
        if got is None or want is None:
            if got != want:
                rows.append(f"evaluation.{split}: {got} != {want}")
            continue
        if got["samples"] != want["samples"]:
            rows.append(
                f"evaluation.{split}.samples: {got['samples']} != {want['samples']} (exact)"
            )
        if not _close(got["mAP50"], want["mAP50"], abs_=TOLERANCES["map50_abs"]):
            rows.append(
                f"evaluation.{split}.mAP50: {got['mAP50']:.4f} != {want['mAP50']:.4f} "
                f"(abs {TOLERANCES['map50_abs']})"
            )
        for name, ap in want["per_class_AP"].items():
            if name not in got["per_class_AP"]:
                rows.append(f"evaluation.{split}.per_class_AP missing class {name!r}")
            elif not _close(got["per_class_AP"][name], ap, abs_=TOLERANCES["per_class_abs"]):
                rows.append(
                    f"evaluation.{split}.per_class_AP[{name}]: "
                    f"{got['per_class_AP'][name]:.4f} != {ap:.4f} "
                    f"(abs {TOLERANCES['per_class_abs']})"
                )
    return rows


@pytest.mark.slow
def test_golden_dev_preset_matches_recorded_numbers(tmp_path: Path) -> None:
    actual = _run_once(tmp_path)
    if UPDATE:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"golden re-recorded at {GOLDEN_PATH}")

    expected = json.loads(GOLDEN_PATH.read_text())
    rows = _compare(actual, expected)
    if rows:
        table = "\n".join(f"  {row}" for row in rows)
        pytest.fail(f"golden numbers moved ({GOLDEN_PATH.name}):\n{table}")


@pytest.mark.slow
def test_same_seed_is_deterministic(tmp_path: Path) -> None:
    """Two runs with the same seed must produce identical numbers [S1 part 1]."""
    first = _run_once(tmp_path / "a")
    second = _run_once(tmp_path / "b")

    assert first["splits"] == second["splits"]
    assert first["epochs"] == second["epochs"]
    assert first["evaluation"] == second["evaluation"]
    assert first["predictions"] == second["predictions"]
    assert first["best"] == second["best"]
