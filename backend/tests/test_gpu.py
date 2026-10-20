"""GPU smoke tests [doc 14 §3.5]. Run on the GPU server with ``make test-gpu``.

These are excluded from ``make verify``/``test-fast`` by the ``gpu`` marker and
skip automatically when no CUDA device is visible. They are the only tests that
touch a real GPU: a CUDA tensor op plus a full tiny-CNN train/evaluate pass.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available"),
]

from fixtures.make_datasets import build_classification, materialize  # noqa: E402

from app.core.device import resolve_device  # noqa: E402
from app.training.plugins.cnn import CNNTrainer  # noqa: E402
from app.verification.manifest import build_manifest, write_manifest  # noqa: E402


def test_cuda_tensor_and_cnn_training(tmp_path: Path) -> None:
    """A real CUDA tensor op succeeds, then the tiny CNN trains on the GPU."""
    # 1) Real CUDA tensor operation [Phase 3 preflight].
    left = torch.rand(64, 64, device="cuda")
    right = torch.rand(64, 64, device="cuda")
    product = left @ right
    torch.cuda.synchronize()
    assert product.device.type == "cuda"
    assert bool((product.sum().cpu() > 0).item())

    # 2) The actual training code path runs on the GPU (1 epoch, tiny preset).
    raw = tmp_path / "raw"
    materialize(raw, build_classification(per_class=4, size=(32, 32)))
    summary = build_manifest(raw, "image_folder", sample_cap=100)
    write_manifest(tmp_path / "manifest.parquet", summary.rows)

    logs: list[str] = []
    metrics: list[tuple[Any, ...]] = []

    def emit_metric(name: str, value: float, step: int, epoch: int, split: str) -> None:
        metrics.append((name, value, step, epoch, split))

    ctx = SimpleNamespace(
        home=tmp_path,
        run_id="gpu-smoke",
        dataset_hash="gpu-smoke",
        model_family="cnn",
        task_type="image_classification",
        dataset_root=raw,
        manifest_path=tmp_path / "manifest.parquet",
        facts={},
        config={"epochs": 1, "batch_size": 4, "imgsz": 32, "patience": 0, "min_delta": 0.0},
        output_dir=tmp_path / "artifacts",
        checkpoint_path=None,
        device=resolve_device("cuda"),
        seed=42,
        emit_log=lambda level, message: logs.append(f"{level} {message}"),
        emit_metric=emit_metric,
        emit_event=lambda event_type, payload: None,
        emit_system=lambda cpu, ram_mb: None,
        save_checkpoint=lambda path, is_best, metrics: None,
        should_pause=lambda: False,
        should_stop=lambda: False,
        wait_if_paused=lambda: None,
    )
    # Explicit CUDA request must resolve to a GPU, not silently fall back [Phase 3].
    assert ctx.device.is_gpu, f"expected a GPU, got: {ctx.device.reason}"

    trainer = CNNTrainer()
    trainer.prepare(ctx)
    assert any("on cuda" in log for log in logs), logs

    result = trainer.train(ctx)
    assert result.epochs == 1
    assert any(name == "val/accuracy" and split == "val" for name, *_values, split in metrics)

    evaluation = trainer.evaluate(ctx)
    assert evaluation["task"] == "image_classification"
    assert evaluation["samples"] > 0
