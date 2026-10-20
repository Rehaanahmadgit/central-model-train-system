"""Trainer plugin interface [doc 08 §1].

A plugin is a new file under ``training/plugins/`` registered in the module
registry; the core pipeline never changes [doc 13 §3]. Plugins run inside the
training subprocess and talk to the parent only through ``RunContext``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.core.device import DeviceInfo


@dataclass
class Issue:
    """One configuration problem found by ``validate_config``."""

    path: str
    message: str
    severity: str = "error"  # "error" | "warn"


@dataclass
class TrainResult:
    """What ``train()`` reports back to the worker."""

    best: dict[str, Any] = field(default_factory=dict)
    epochs: int = 0
    note: str = ""


class RunContext(Protocol):
    """Everything a plugin needs; implemented by the training subprocess."""

    home: Path
    run_id: str
    dataset_hash: str
    model_family: str
    task_type: str | None
    dataset_root: Path
    manifest_path: Path
    facts: dict[str, Any]
    config: dict[str, Any]
    output_dir: Path
    checkpoint_path: Path | None
    device: DeviceInfo
    seed: int

    def emit_metric(self, name: str, value: float, step: int, epoch: int, split: str) -> None: ...

    def emit_log(self, level: str, message: str) -> None: ...

    def emit_event(self, event_type: str, payload: dict[str, Any]) -> None: ...

    def emit_system(self, cpu: float | None, ram_mb: float | None) -> None: ...

    def save_checkpoint(self, path: Path, is_best: bool, metrics: dict[str, Any]) -> None: ...

    def should_pause(self) -> bool: ...

    def should_stop(self) -> bool: ...

    def wait_if_paused(self) -> None: ...


class TrainerPlugin(ABC):
    """One model family: validate, prepare, train, evaluate [doc 08 §1]."""

    family: str
    supported_tasks: list[str]
    supported_formats: list[str]
    description: str
    needs_gpu: bool = False
    hparam_schema: dict[str, Any] = {}
    default_hparams: dict[str, Any] = {}

    @abstractmethod
    def validate_config(self, config: dict[str, Any], facts: dict[str, Any]) -> list[Issue]:
        """Reject impossible combinations before the subprocess starts."""

    @abstractmethod
    def prepare(self, ctx: RunContext) -> None:
        """Load data/loaders, build the model, adapt the head if needed."""

    @abstractmethod
    def train(self, ctx: RunContext) -> TrainResult:
        """Main loop; must honour ``ctx.should_pause()/should_stop()``."""

    def evaluate(self, ctx: RunContext) -> dict[str, Any]:
        """Optional final metrics; default returns what train reported."""
        return {}

    def resume(self, ctx: RunContext, checkpoint_path: Path) -> None:  # pragma: no cover
        """Optional exact resume from a checkpoint [doc 08 §3]."""
        del ctx, checkpoint_path
