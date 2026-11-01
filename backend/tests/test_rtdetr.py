"""Honest-evaluation tests for the RT-DETR plugin [doc 08 §6].

Fast (no training loop): the split loader, the evaluation-report shape and the
best-checkpoint reload are exercised directly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from fixtures.make_datasets import build_yolo, materialize, png_bytes

from app.training.plugins.rtdetr import RTDetrTrainer, build_eval_report, split_warning
from app.verification.manifest import build_manifest, write_manifest


def _ctx(tmp_path: Path, raw: Path, *, seed: int = 42, num_classes: int = 2) -> Any:
    summary = build_manifest(raw, "yolo", sample_cap=500)
    manifest = tmp_path / "manifest.parquet"
    write_manifest(manifest, summary.rows)
    logs: list[str] = []

    def emit_log(level: str, message: str) -> None:
        logs.append(f"{level} {message}")

    ctx = SimpleNamespace(
        home=tmp_path,
        run_id="run_test",
        dataset_hash="hash",
        model_family="rtdetr",
        task_type="object_detection",
        dataset_root=raw,
        manifest_path=manifest,
        facts={},
        config={},
        output_dir=tmp_path / "artifacts",
        checkpoint_path=None,
        device=SimpleNamespace(is_gpu=False, kind="cpu"),
        seed=seed,
        emit_log=emit_log,
        emit_metric=lambda *args, **kwargs: None,
        emit_event=lambda *args, **kwargs: None,
        emit_system=lambda *args, **kwargs: None,
        save_checkpoint=lambda *args, **kwargs: None,
        should_pause=lambda: False,
        should_stop=lambda: False,
        wait_if_paused=lambda: None,
    )
    trainer = RTDetrTrainer()
    trainer._num_classes = num_classes
    trainer._class_names = ["cat", "dog"][:num_classes]
    return ctx, trainer, logs


def _split_dataset(with_test: bool) -> dict[str, bytes]:
    """YOLO layout with real directory splits (``train/``, ``val/``, ``test/``)."""
    entries: dict[str, bytes] = {}
    layout = [("train", 8), ("val", 4)] + ([("test", 3)] if with_test else [])
    index = 0
    for split, count in layout:
        for _ in range(count):
            entries[f"{split}/images/img{index}.png"] = png_bytes((10 + index, 40, 90))
            entries[f"{split}/labels/img{index}.txt"] = b"0 0.25 0.25 0.5 0.5\n"
            index += 1
    entries["data.yaml"] = b"names:\n  - cat\n  - dog\n"
    return entries


def test_test_split_images_never_enter_train_or_val(tmp_path: Path) -> None:
    raw = materialize(tmp_path / "raw", _split_dataset(with_test=True))
    ctx, trainer, _logs = _ctx(tmp_path, raw)

    train, val, test = trainer._load_samples(ctx)

    assert len(train) == 8
    assert len(val) == 4
    assert len(test) == 3
    trained = {path for path, _boxes in train} | {path for path, _boxes in val}
    assert not any("/test/" in path.as_posix() for path in trained)
    assert {path.name for path, _ in test} == {"img12.png", "img13.png", "img14.png"}


def test_unspecified_split_falls_back_to_deterministic_80_20(tmp_path: Path) -> None:
    entries = {
        f"images/img{index}.png": png_bytes((index * 5, 30, 60)) for index in range(20)
    }
    entries.update(
        {f"labels/img{index}.txt": b"1 0.5 0.5 0.2 0.2\n" for index in range(20)}
    )
    entries["data.yaml"] = b"names:\n  - cat\n  - dog\n"
    raw = materialize(tmp_path / "raw", entries)

    first_ctx, first, logs = _ctx(tmp_path, raw, seed=7)
    train_a, val_a, test_a = first._load_samples(first_ctx)
    second_ctx, second, _ = _ctx(tmp_path, raw, seed=7)
    train_b, val_b, _test_b = second._load_samples(second_ctx)

    assert len(train_a) == 16 and len(val_a) == 4
    assert val_a == val_b, "same seed must give the same split"
    assert train_a == train_b
    assert test_a == []
    assert any("deterministic 80/20" in line for line in logs)


def test_eval_report_has_val_and_test_sections() -> None:
    report = build_eval_report(
        ["cat", "dog"],
        {"mAP50": 0.9, "per_class_AP": {"cat": 0.95, "dog": 0.85}, "samples": 19},
        {"mAP50": 0.5, "per_class_AP": {"cat": 0.7, "dog": 0.3}, "samples": 10},
    )

    assert report["val"]["mAP50"] == 0.9
    assert report["test"]["mAP50"] == 0.5
    assert report["test"]["samples"] == 10
    assert report["test_reason"] is None
    # Backward compatibility: the top-level keys are the val numbers.
    assert report["mAP50"] == 0.9
    assert report["per_class_AP"] == {"cat": 0.95, "dog": 0.85}
    assert report["samples"] == 19
    assert report["task"] == "object_detection"


def test_eval_report_test_is_null_without_test_split() -> None:
    report = build_eval_report(
        ["cat"],
        {"mAP50": 0.42, "per_class_AP": {"cat": 0.42}, "samples": 5},
        None,
    )

    assert report["test"] is None
    assert report["test_reason"] == "no test split"
    assert report["mAP50"] == 0.42


def test_small_split_carries_warning() -> None:
    assert split_warning(19) is not None
    assert "19 images" in str(split_warning(19))
    assert split_warning(30) is None
    assert split_warning(500) is None


def test_evaluate_reloads_best_checkpoint(tmp_path: Path) -> None:
    """Evaluation must use the kept checkpoint, not the last epoch's weights."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True)
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    torch.save({"epoch": 3, "model_state": model.state_dict()}, checkpoints / "best.ckpt")

    logs: list[str] = []
    ctx = SimpleNamespace(
        output_dir=tmp_path / "artifacts",
        emit_log=lambda level, message: logs.append(f"{level} {message}"),
    )
    trainer = RTDetrTrainer()
    trainer._model = torch.nn.Linear(2, 2, bias=False)  # random init, must be replaced
    trainer._device = torch.device("cpu")

    trainer._load_best_weights(ctx)

    assert torch.allclose(trainer._model.weight, torch.eye(2))
    assert any("best weights" in line for line in logs)
    assert not trainer._model.training


def test_evaluate_without_best_checkpoint_keeps_current_weights(tmp_path: Path) -> None:
    trainer = RTDetrTrainer()
    trainer._model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        trainer._model.weight.copy_(torch.zeros(2, 2))
    logs: list[str] = []
    ctx = SimpleNamespace(
        output_dir=tmp_path / "artifacts",
        emit_log=lambda level, message: logs.append(f"{level} {message}"),
    )

    trainer._load_best_weights(ctx)

    assert torch.allclose(trainer._model.weight, torch.zeros(2, 2))
    assert any("final epoch weights" in line for line in logs)


def test_yolo_fixture_split_still_trains_only_on_train_rows(tmp_path: Path) -> None:
    """The shared fixture (train/val prefixes) keeps working after the split change."""
    raw = materialize(tmp_path / "raw", build_yolo(per_class=4, split=True))
    ctx, trainer, _logs = _ctx(tmp_path, raw, num_classes=2)

    train, val, test = trainer._load_samples(ctx)

    assert test == []
    assert train and val
    assert all("/val/" not in path.as_posix() for path, _ in train)
