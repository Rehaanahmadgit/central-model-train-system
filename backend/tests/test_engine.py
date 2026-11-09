"""Unit tests for the shared training engine [A2]; no RT-DETR needed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from app.training.engine import EngineHooks, TrainEngine


def _make(scores: list[float], losses: list[float] | None = None, epochs: int = 4):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    calls = {"step": 0, "eval": 0}
    batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 1.0]])]

    def step(batch: Any) -> float:
        loss = model(batch).pow(2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        i = calls["step"]
        calls["step"] += 1
        return losses[i] if losses else float(loss)

    def evaluate() -> float:
        i = calls["eval"]
        calls["eval"] += 1
        return scores[i] if i < len(scores) else scores[-1]

    hooks = EngineHooks(
        batches=lambda: batches,
        train_step=step,
        evaluate=evaluate,
        set_train_mode=model.train,
        model_state=model.state_dict,
        load_model_state=model.load_state_dict,
        optimizer_state=opt.state_dict,
        load_optimizer_state=opt.load_state_dict,
    )
    return model, TrainEngine(hooks=hooks, epochs=epochs), calls


def _ctx(tmp_path: Path, *, stop_after_checks: int | None = None):
    metrics: list[tuple[str, float, int]] = []
    saved: list[tuple[str, bool]] = []
    checks = {"n": 0}

    def should_stop() -> bool:
        checks["n"] += 1
        return stop_after_checks is not None and checks["n"] > stop_after_checks

    ctx = SimpleNamespace(
        output_dir=tmp_path / "artifacts",
        emit_metric=lambda name, value, step, epoch, split: metrics.append((name, value, epoch)),
        emit_log=lambda level, message: None,
        save_checkpoint=lambda path, is_best, metrics: saved.append((Path(path).name, is_best)),
        should_pause=lambda: False,
        wait_if_paused=lambda: None,
        should_stop=should_stop,
    )
    return ctx, metrics, saved


def test_tie_keeps_earlier_epoch(tmp_path: Path) -> None:
    _, engine, _ = _make([0.2, 0.5, 0.5, 0.4])
    ctx, _, saved = _ctx(tmp_path)
    result = engine.run(ctx)
    assert result.best == {"metric": "val/mAP50", "value": 0.5, "epoch": 2}
    assert [name for name, best in saved if best] == ["best.ckpt", "best.ckpt"]  # epochs 1, 2 only


def test_metrics_emitted_once_per_epoch(tmp_path: Path) -> None:
    _, engine, _ = _make([0.1, 0.2, 0.3, 0.4], epochs=3)
    ctx, metrics, _ = _ctx(tmp_path)
    engine.run(ctx)
    assert [(n, e) for n, _, e in metrics if n == "train/loss"] == [
        ("train/loss", 1),
        ("train/loss", 2),
        ("train/loss", 3),
    ]
    assert len([m for m in metrics if m[0] == "val/mAP50"]) == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_inf_loss_aborts(tmp_path: Path, bad: float) -> None:
    _, engine, _ = _make([0.1], losses=[1.0, bad])
    ctx, _, _ = _ctx(tmp_path)
    with pytest.raises(ValueError, match="nan/inf loss at epoch 1"):
        engine.run(ctx)


def test_nan_score_aborts(tmp_path: Path) -> None:
    _, engine, _ = _make([float("nan")])
    ctx, _, _ = _ctx(tmp_path)
    with pytest.raises(ValueError, match="nan validation"):
        engine.run(ctx)


def test_stop_before_first_epoch_trains_nothing(tmp_path: Path) -> None:
    _, engine, calls = _make([0.1])
    ctx, _, saved = _ctx(tmp_path, stop_after_checks=0)
    result = engine.run(ctx)
    assert calls["step"] == 0 and saved == []
    assert result.best["epoch"] == 0 and result.best["value"] == 0.0


def test_resume_is_exact(tmp_path: Path) -> None:
    scores = [0.1, 0.4, 0.3, 0.6]
    straight_model, straight, _ = _make(scores, epochs=4)
    ctx1, _, _ = _ctx(tmp_path / "straight")
    full = straight.run(ctx1)

    first_model, first, _ = _make(scores, epochs=2)
    ctx2, _, _ = _ctx(tmp_path / "split")
    first.run(ctx2)
    last = TrainEngine.checkpoint_dir(ctx2) / "last.ckpt"

    # Fresh process: new model/optimizer with different init, resumed from last.ckpt.
    resumed_model, resumed, calls = _make(scores[2:], epochs=4)
    torch.nn.init.constant_(resumed_model.weight, 9.0)
    resumed.resume(ctx2, last, "cpu")
    assert resumed.start_epoch == 3
    assert (resumed.best_value, resumed.best_epoch) == (0.4, 2)  # includes the epoch just finished
    result = resumed.run(ctx2)

    assert result.best == full.best == {"metric": "val/mAP50", "value": 0.6, "epoch": 4}
    for a, b in zip(
        straight_model.state_dict().values(), resumed_model.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)


def test_load_best_without_file_returns_none(tmp_path: Path) -> None:
    _, engine, _ = _make([0.1])
    ctx, _, _ = _ctx(tmp_path)
    assert engine.load_best(ctx, "cpu") is None
