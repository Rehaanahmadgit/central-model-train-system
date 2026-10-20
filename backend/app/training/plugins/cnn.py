"""Small image-classification CNN (PyTorch, CPU) [doc 08 §2, doc 14 §3.4].

Trains from scratch on the normalized manifest.parquet so the same plugin
works unchanged on the GPU image (the device comes from ``RunContext``).
``dev_preset`` = 2 epochs, 32 px images, batch 4 so a laptop demo finishes in
seconds. No pretrained checkpoint is downloaded.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.training.base import Issue, RunContext, TrainerPlugin, TrainResult


class _TinyCNN(nn.Module):
    def __init__(self, num_classes: int, channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x: Any) -> Any:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNTrainer(TrainerPlugin):
    family = "cnn"
    supported_tasks = ["image_classification"]
    supported_formats = ["image_folder", "csv_path_label"]
    description = "Tiny from-scratch CNN (CPU or GPU). dev_preset: 2 epochs, 32 px, batch 4."
    needs_gpu = False
    hparam_schema = {
        "type": "object",
        "properties": {
            "epochs": {"type": "integer", "minimum": 1, "default": 2},
            "batch_size": {"type": "integer", "minimum": 1, "default": 4},
            "imgsz": {"type": "integer", "minimum": 16, "maximum": 1024, "default": 32},
            "lr": {"type": "number", "minimum": 1e-6, "default": 0.001},
        },
    }
    default_hparams = {"epochs": 2, "batch_size": 4, "imgsz": 32, "lr": 0.001}

    # -- plugin interface ---------------------------------------------------

    def validate_config(self, config: dict[str, Any], facts: dict[str, Any]) -> list[Issue]:
        del facts
        issues: list[Issue] = []
        if int(config.get("epochs", 2)) < 1:
            issues.append(Issue("epochs", "must be at least 1"))
        if int(config.get("batch_size", 4)) < 1:
            issues.append(Issue("batch_size", "must be at least 1"))
        return issues

    def prepare(self, ctx: RunContext) -> None:
        table = pq.read_table(ctx.manifest_path)
        paths = table.column("path").to_pylist()
        labels = table.column("label").to_pylist()
        splits = table.column("split").to_pylist()

        classes = sorted({label for label in labels if label is not None})
        if len(classes) < 2:
            raise ValueError(f"classification needs at least 2 classes, got {classes}")
        self._class_to_id = {name: index for index, name in enumerate(classes)}

        samples = [
            (path, self._class_to_id[label], split)
            for path, label, split in zip(paths, labels, splits, strict=False)
            if label is not None and (ctx.dataset_root / path).is_file()
        ]
        if not samples:
            raise ValueError("no labelled images in the manifest")

        train = [sample for sample in samples if sample[2] == "train"]
        val = [sample for sample in samples if sample[2] == "val"]
        if not train and not val:
            # No explicit split: 80/20 by a deterministic permutation [doc 06 §4].
            generator = torch.Generator().manual_seed(ctx.seed)
            order = torch.randperm(len(samples), generator=generator).tolist()
            boundary = max(1, int(0.8 * len(samples)))
            train = [samples[index] for index in order[:boundary]]
            val = [samples[index] for index in order[boundary:]]
        elif not val:
            boundary = max(1, int(0.8 * len(train)))
            val = train[boundary:]
            train = train[:boundary]

        self._train_samples = train
        self._val_samples = val
        self._imgsz = int(ctx.config.get("imgsz", 32))
        self._batch_size = int(ctx.config.get("batch_size", 4))
        self._epochs = int(ctx.config.get("epochs", 2))
        self._lr = float(ctx.config.get("lr", 0.001))

        self._model = _TinyCNN(len(classes))
        self._device = torch.device("cuda" if ctx.device.is_gpu else "cpu")
        self._model.to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._criterion = nn.CrossEntropyLoss()
        ctx.emit_log(
            "INFO",
            f"prepared {len(train)} train / {len(val)} val images, {len(classes)} classes "
            f"on {self._device}",
        )

    def train(self, ctx: RunContext) -> TrainResult:
        best_accuracy = -1.0
        for epoch in range(1, self._epochs + 1):
            if ctx.should_stop():
                break
            self._model.train()
            running_loss = 0.0
            correct = 0
            seen = 0
            for batch_index, batch in enumerate(_batches(self._train_samples, self._batch_size)):
                if ctx.should_pause():
                    ctx.wait_if_paused()
                if ctx.should_stop():
                    break
                loss, hits, count = self._train_batch(ctx, batch)
                running_loss += loss * count
                correct += hits
                seen += count
                step = (epoch - 1) * len(self._train_samples) + seen
                ctx.emit_metric("train/loss", float(running_loss / seen), step, epoch, "train")
                time.sleep(float(ctx.config.get("batch_delay", 0.0)))
                del batch_index

            val_loss, val_accuracy, val_count = self._evaluate(ctx)
            train_accuracy = correct / max(seen, 1)
            ctx.emit_metric(
                "train/accuracy",
                float(train_accuracy),
                epoch * len(self._train_samples),
                epoch,
                "train",
            )
            ctx.emit_metric("val/loss", float(val_loss), epoch, epoch, "val")
            ctx.emit_metric("val/accuracy", float(val_accuracy), epoch, epoch, "val")
            ctx.emit_log(
                "INFO",
                f"epoch {epoch}/{self._epochs} train_acc={train_accuracy:.4f} "
                f"val_acc={val_accuracy:.4f} val_loss={val_loss:.4f}",
            )
            self._save(ctx, epoch, is_best=val_accuracy > best_accuracy)
            best_accuracy = max(best_accuracy, val_accuracy)
        return TrainResult(
            best={
                "metric": "val/accuracy",
                "value": round(best_accuracy, 4),
                "epoch": self._epochs,
            },
            epochs=self._epochs,
            note=f"val/accuracy={best_accuracy:.4f}",
        )

    # -- helpers ------------------------------------------------------------

    def _train_batch(
        self, ctx: RunContext, batch: list[tuple[str, int, str]]
    ) -> tuple[float, int, int]:
        tensors: list[Any] = []
        targets: list[int] = []
        for path, label, _split in batch:
            tensor = self._load(ctx.dataset_root / path)
            if tensor is None:
                continue
            tensors.append(tensor)
            targets.append(label)
        if not tensors:
            return 0.0, 0, 0
        x = torch.stack(tensors).to(self._device)
        y = torch.tensor(targets, dtype=torch.long, device=self._device)
        self._optimizer.zero_grad()
        logits = self._model(x)
        loss = self._criterion(logits, y)
        loss.backward()
        self._optimizer.step()
        predictions = logits.argmax(dim=1)
        hits = int((predictions == y).sum().item())
        return float(loss.item()), hits, len(targets)

    def _evaluate(self, ctx: RunContext) -> tuple[float, float, int]:
        self._model.eval()
        total_loss = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in _batches(self._val_samples, self._batch_size):
                tensors = []
                targets = []
                for path, label, _split in batch:
                    tensor = self._load(ctx.dataset_root / path)
                    if tensor is None:
                        continue
                    tensors.append(tensor)
                    targets.append(label)
                if not tensors:
                    continue
                x = torch.stack(tensors).to(self._device)
                y = torch.tensor(targets, dtype=torch.long, device=self._device)
                logits = self._model(x)
                total_loss += float(self._criterion(logits, y).item()) * len(targets)
                correct += int((logits.argmax(dim=1) == y).sum().item())
                count += len(targets)
        return total_loss / max(count, 1), correct / max(count, 1), count

    def _save(self, ctx: RunContext, epoch: int, *, is_best: bool) -> None:
        checkpoints = ctx.output_dir.parent / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        last = checkpoints / "last.ckpt"
        torch.save(self._model.state_dict(), last)
        ctx.save_checkpoint(last, is_best=False, metrics={"epoch": epoch})
        if is_best:
            best = checkpoints / "best.ckpt"
            torch.save(self._model.state_dict(), best)
            ctx.save_checkpoint(best, is_best=True, metrics={"epoch": epoch})

    def _load(self, path: Path) -> Any | None:
        try:
            with Image.open(path) as opened:
                rgb = opened.convert("RGB").resize((self._imgsz, self._imgsz))
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        except (OSError, ValueError):
            return None
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor


def _batches(
    samples: list[tuple[str, int, str]], batch_size: int
) -> list[list[tuple[str, int, str]]]:
    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]
