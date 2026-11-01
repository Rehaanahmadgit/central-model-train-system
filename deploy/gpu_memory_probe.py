#!/usr/bin/env python3
"""GPU memory probe for RT-DETR r18vd [doc 14 §3.5, D122].

Sweeps ``batch_size`` × ``imgsz`` with fp16 autocast and one **real training
step** (forward + SetCriterion loss + backward + AdamW step), reporting
``torch.cuda.max_memory_allocated()`` per cell. Cells that raise
``torch.cuda.OutOfMemoryError`` are marked OOM and skipped.

The model is rebuilt once per imgsz (the encoder's fixed positional
embeddings depend on imgsz); the batch sweep reuses it, exactly like training.

Run on the GPU server, inside the -gpu image or any env with torch+CUDA:

    python deploy/gpu_memory_probe.py
    python deploy/gpu_memory_probe.py --batches 1 2 --imgszs 320 512
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

# deploy/ lives one level above backend/; import the vendored RT-DETR and the
# plugin constants exactly as the trainer does.
BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.training.plugins.rtdetr import (  # noqa: E402
    LOSSES,
    MATCHER_COSTS,
    WEIGHT_DICT,
    build_r18vd,
)
from app.vendor.rtdetr import HungarianMatcher, SetCriterion  # noqa: E402

BATCHES = [1, 2, 4, 8]
IMGSZS = [320, 512, 640]
NUM_CLASSES = 80  # official COCO head (the real pretrained checkpoint)
BOXES_PER_IMAGE = 5
LEARNING_RATE = 0.0001
MIB = 1024 * 1024


def build(num_classes: int, imgsz: int) -> tuple[torch.nn.Module, torch.nn.Module]:
    model = build_r18vd(num_classes, eval_spatial_size=(imgsz, imgsz))
    matcher = HungarianMatcher(
        weight_dict=MATCHER_COSTS, use_focal_loss=True, alpha=0.25, gamma=2.0
    )
    criterion = SetCriterion(
        matcher,
        weight_dict=WEIGHT_DICT,
        losses=LOSSES,
        alpha=0.75,
        gamma=2.0,
        num_classes=num_classes,
    )
    return model, criterion


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    # Same groups/lr as the trainer: backbone 0.1×, head 1× [doc 08 §6].
    backbone = [
        p for n, p in model.named_parameters() if n.startswith("backbone") and p.requires_grad
    ]
    head = [
        p for n, p in model.named_parameters() if not n.startswith("backbone") and p.requires_grad
    ]
    groups: list[dict] = []
    if backbone:
        groups.append({"params": backbone, "lr": LEARNING_RATE * 0.1})
    if head:
        groups.append({"params": head, "lr": LEARNING_RATE})
    return torch.optim.AdamW(groups, lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)


def make_targets(batch: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
    """Synthetic detection targets: a few random normalized boxes per image."""
    targets: list[dict[str, torch.Tensor]] = []
    for _ in range(batch):
        labels = torch.randint(
            0, NUM_CLASSES, (BOXES_PER_IMAGE,), dtype=torch.int64, device=device
        )
        boxes = torch.rand(BOXES_PER_IMAGE, 4, dtype=torch.float32, device=device)
        targets.append({"labels": labels, "boxes": boxes})
    return targets


def run_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: int,
    imgsz: int,
    device: torch.device,
) -> float:
    """One real training step; returns peak memory in MiB."""
    images = torch.randn(batch, 3, imgsz, imgsz, device=device)
    targets = make_targets(batch, device)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(images, targets)
        losses = criterion(outputs, targets)
        loss = sum(losses.values())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / MIB


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--imgszs", type=int, nargs="+", default=IMGSZS)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is not available; run this probe on the GPU server (or the -gpu image).")
        return 2

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'batch':>5} {'imgsz':>5} {'max_memory_allocated_MiB':>24}  status")
    for imgsz in args.imgszs:
        model, criterion = build(NUM_CLASSES, imgsz)
        model.to(device)
        optimizer = make_optimizer(model)
        for batch in args.batches:
            try:
                mem = run_step(model, criterion, optimizer, batch, imgsz, device)
            except torch.cuda.OutOfMemoryError:
                print(f"{batch:>5} {imgsz:>5} {'—':>24}  OOM")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            print(f"{batch:>5} {imgsz:>5} {mem:>24.1f}  ok")
            gc.collect()
        # Free this imgsz's model before building the next (larger) one.
        del model, criterion, optimizer
        torch.cuda.empty_cache()
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
