"""Shared epoch-loop engine for trainer plugins [A2].

Owns the parts of training that must behave identically for every model family:
the epoch loop, pause/stop checks, the NaN/inf guard, best-metric tracking
(strict ``>``: a tie keeps the earlier epoch), checkpoint save/load, exact
resume and the per-epoch metric/log emission.

A plugin supplies only what is model specific, through :class:`EngineHooks`.
The engine never imports torch at module level beyond ``torch.save/load`` so a
plugin without torch can still reuse the loop if it provides its own hooks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.training.base import RunContext, TrainResult

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class EngineHooks:
    """Model-specific callables a plugin hands to the engine."""

    batches: Callable[[], Sequence[Any]]  # one epoch worth of training batches
    train_step: Callable[[Any], float]  # forward/backward/step, returns the loss
    evaluate: Callable[[], float]  # validation metric (higher is better)
    set_train_mode: Callable[[], None]
    model_state: Callable[[], dict[str, Any]]
    load_model_state: Callable[[dict[str, Any]], None]
    optimizer_state: Callable[[], dict[str, Any]]
    load_optimizer_state: Callable[[dict[str, Any]], None]
    # Extra files written next to best.ckpt (e.g. an upstream-compatible export).
    export_best: Callable[[Path], None] | None = None


@dataclass
class TrainEngine:
    hooks: EngineHooks
    epochs: int
    loss_metric: str = "train/loss"
    score_metric: str = "val/mAP50"
    log_label: str = ""
    start_epoch: int = 1
    best_value: float = -1.0
    best_epoch: int = 0

    # -- loop ----------------------------------------------------------------

    def run(self, ctx: RunContext) -> TrainResult:
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.epochs + 1):
            if ctx.should_stop():
                break
            self.hooks.set_train_mode()
            running_loss = 0.0
            seen = 0
            for batch in self.hooks.batches():
                if ctx.should_pause():
                    ctx.wait_if_paused()
                if ctx.should_stop():
                    break
                loss = self.hooks.train_step(batch)
                if math.isnan(loss) or math.isinf(loss):
                    raise ValueError(f"nan/inf loss at epoch {epoch}")
                running_loss += loss
                seen += 1
            if ctx.should_stop():
                break
            # Epoch-mean train loss, one point per epoch (step == epoch) [doc 08 §6].
            train_loss = running_loss / max(seen, 1)
            ctx.emit_metric(self.loss_metric, float(train_loss), epoch, epoch, "train")
            score = self.hooks.evaluate()
            if math.isnan(score):
                raise ValueError(f"nan validation mAP at epoch {epoch}")
            ctx.emit_metric(self.score_metric, score, epoch, epoch, "val")
            improved = score > self.best_value  # strict: ties keep the earlier epoch
            ctx.emit_log(
                "INFO",
                f"epoch {epoch}/{self.epochs} train/loss={train_loss:.4f} val/mAP50={score:.4f}",
            )
            # Best is updated BEFORE saving so the checkpoint records it: a resume
            # from last.ckpt must not forget the epoch it just finished.
            if improved:
                self.best_value = score
                self.best_epoch = epoch
            self.save(ctx, epoch, is_best=improved)
        shown = max(self.best_value, 0.0)
        return TrainResult(
            best={
                "metric": self.score_metric,
                "value": round(shown, 4),
                "epoch": self.best_epoch,
            },
            epochs=epoch,
            note=f"{self.score_metric}={shown:.4f}",
        )

    # -- checkpoints ---------------------------------------------------------

    @staticmethod
    def checkpoint_dir(ctx: RunContext) -> Path:
        return ctx.output_dir.parent / "checkpoints"

    def save(self, ctx: RunContext, epoch: int, *, is_best: bool) -> None:
        checkpoints = self.checkpoint_dir(ctx)
        checkpoints.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": epoch,
            "model_state": self.hooks.model_state(),
            "optimizer_state": self.hooks.optimizer_state(),
            "best_map": self.best_value,
            "best_epoch": self.best_epoch,
        }
        last = checkpoints / "last.ckpt"
        torch.save(payload, last)
        ctx.save_checkpoint(last, is_best=False, metrics={"epoch": epoch})
        if is_best:
            best = checkpoints / "best.ckpt"
            torch.save(payload, best)
            if self.hooks.export_best is not None:
                self.hooks.export_best(checkpoints)
            ctx.save_checkpoint(best, is_best=True, metrics={"epoch": epoch})

    def resume(self, ctx: RunContext, checkpoint_path: Path, device: Any) -> None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(payload, dict) and "model_state" in payload:
            self.hooks.load_model_state(payload["model_state"])
            self.hooks.load_optimizer_state(payload["optimizer_state"])
            self.start_epoch = int(payload.get("epoch", 0)) + 1
            self.best_value = float(payload.get("best_map", -1.0))
            self.best_epoch = int(payload.get("best_epoch", 0))
            ctx.emit_log("INFO", f"resumed from {checkpoint_path.name} at epoch {self.start_epoch}")
        else:
            self.hooks.load_model_state(payload)
            ctx.emit_log("INFO", f"loaded weights from {checkpoint_path.name} (no optimizer state)")

    def load_best(self, ctx: RunContext, device: Any) -> int | None:
        return load_best_checkpoint(ctx, device, self.hooks.load_model_state)


def load_best_checkpoint(
    ctx: RunContext, device: Any, load_model_state: Callable[[dict[str, Any]], None]
) -> int | None:
    """Load best.ckpt (the kept checkpoint, not the last epoch). Returns its epoch."""
    best = TrainEngine.checkpoint_dir(ctx) / "best.ckpt"
    if not best.is_file():
        ctx.emit_log("INFO", "no best.ckpt found: evaluating the final epoch weights")
        return None
    payload = torch.load(best, map_location=device, weights_only=True)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        ctx.emit_log("WARNING", f"{best.name} has no model_state: evaluating last weights")
        return None
    load_model_state(state)
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    ctx.emit_log("INFO", f"evaluation uses best weights from {best.name} (epoch {epoch})")
    return int(epoch) if epoch is not None else 0
