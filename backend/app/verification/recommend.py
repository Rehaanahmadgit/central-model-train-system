"""Checkpoint + hyperparameter recommendations [doc 06 §6, doc 07 §3].

Rule-based v1, explainable: every recommendation carries a human-readable
reason. The catalog itself (real URLs/sha256) is S7 work; until a
``catalog.yaml`` exists this returns an empty list with a clear note rather
than inventing identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.device import DeviceInfo
from app.verification.manifest import ManifestSummary

SIZE_BUCKETS = ((1_000, "tiny"), (10_000, "small"), (100_000, "medium"))


@dataclass(frozen=True)
class CatalogEntry:
    catalog_id: str
    model_family: str
    task_types: tuple[str, ...]
    pretrained_on: str
    license: str = ""
    size_bytes: int = 0
    min_vram_gb: float = 0.0
    is_active: bool = True


def size_bucket(num_samples: int) -> str:
    for ceiling, name in SIZE_BUCKETS:
        if num_samples < ceiling:
            return name
    return "large"


def load_catalog(path: Path) -> list[CatalogEntry]:
    """Load ``checkpoints/catalog.yaml``; missing file means "not configured"."""
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    entries: list[CatalogEntry] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        version = item.get("version")
        if not name or not version:
            continue
        entries.append(
            CatalogEntry(
                catalog_id=f"{name}@{version}",
                model_family=str(item.get("model_family", "")),
                task_types=tuple(str(value) for value in item.get("task_types", []) or []),
                pretrained_on=str(item.get("pretrained_on", "")),
                license=str(item.get("license", "")),
                size_bytes=int(item.get("size_bytes", 0) or 0),
                min_vram_gb=float(item.get("min_vram_gb", 0.0) or 0.0),
                is_active=bool(item.get("is_active", True)),
            )
        )
    return entries


def rank_checkpoints(
    catalog: list[CatalogEntry],
    *,
    task_type: str | None,
    model_family: str,
    bucket: str,
    device: DeviceInfo,
) -> list[dict[str, Any]]:
    """Score catalog entries with the documented v1 rules [doc 07 §3.2]."""
    ranked: list[dict[str, Any]] = []
    for entry in catalog:
        if not entry.is_active:
            continue
        if task_type is not None and task_type not in entry.task_types:
            continue
        if entry.model_family != model_family:
            continue
        score = 0
        reasons: list[str] = []
        if task_type is not None and task_type in entry.task_types:
            score += 5
            reasons.append(f"exact task match ({task_type})")
        if entry.model_family == model_family:
            score += 3
            reasons.append(f"model family match ({model_family})")
        if entry.pretrained_on:
            score += 2
            reasons.append(f"pretrained on {entry.pretrained_on}")
        if bucket in ("tiny", "small"):
            # Smaller checkpoints are preferred for small data; size_bytes is a proxy.
            score += 1 if entry.size_bytes <= 200_000_000 else 0
            reasons.append("small-data friendly variant preferred")
        needs_gpu = entry.min_vram_gb > 0
        if needs_gpu and not device.is_gpu:
            score -= 5
            reasons.append("needs a GPU, none available")
        elif entry.min_vram_gb:
            reasons.append(f"needs ~{entry.min_vram_gb:.0f} GB VRAM")
        ranked.append(
            {
                "catalog_id": entry.catalog_id,
                "score": score,
                "reason": "; ".join(reasons),
                "local": False,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["catalog_id"]))
    return ranked


def finetune_strategy(bucket: str) -> str:
    return {
        "tiny": "freeze backbone, low LR, heavy augmentation",
        "small": "partial unfreeze, discriminative LR",
        "medium": "partial unfreeze, discriminative LR",
        "large": "full fine-tune",
    }[bucket]


def recommend_hparams(
    summary: ManifestSummary,
    *,
    task_type: str | None,
    model_family: str,
    device: DeviceInfo,
) -> dict[str, Any]:
    """Heuristic defaults shown in the Configure step (ASSUMPTION: not spec'd)."""
    total = sum(summary.class_counts.values()) or len(summary.rows)
    bucket = size_bucket(total)
    max_side = 0
    if summary.image_stats.get("max_wh"):
        max_side = max(int(value) for value in summary.image_stats["max_wh"])

    imgsz = 640
    if max_side and max_side < 640:
        imgsz = max(32, (max_side + 31) // 32 * 32)
    batch_size = 16 if device.is_gpu else 4
    lr = {
        "object_detection": 0.01,
        "image_classification": 0.001,
        "segmentation": 0.01,
        "tabular_classification": None,
        "tabular_regression": None,
    }.get(task_type or "", 0.001)

    hparams: dict[str, Any] = {
        "epochs": 50 if bucket != "tiny" else 20,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "freeze_backbone": bucket == "tiny",
        "strategy": finetune_strategy(bucket),
        "dataset_size_bucket": bucket,
        "samples": total,
        "device": "cuda" if device.is_gpu else "cpu",
        "model_family": model_family,
    }
    if lr is not None:
        hparams["lr"] = lr
    if task_type is None:
        hparams["note"] = "task type could not be detected; confirm it before training"
    return hparams
