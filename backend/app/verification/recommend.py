"""Checkpoint + hyperparameter recommendations [doc 06 §6, doc 07 §3].

Rule-based v1, explainable: every recommendation carries a human-readable
reason. The catalog itself (real URLs/sha256) is S7 work; until a
``catalog.yaml`` exists this returns an empty list with a clear note rather
than inventing identifiers.
"""

from __future__ import annotations

from typing import Any

from app.checkpoints.catalog import CatalogEntry
from app.core.device import DeviceInfo
from app.verification.manifest import ManifestSummary

SIZE_BUCKETS = ((1_000, "tiny"), (10_000, "small"), (100_000, "medium"))


def size_bucket(num_samples: int) -> str:
    for ceiling, name in SIZE_BUCKETS:
        if num_samples < ceiling:
            return name
    return "large"


def _hard_filters(
    entry: CatalogEntry,
    *,
    task_type: str | None,
    model_family: str,
    num_classes: int | None,
    device: DeviceInfo,
) -> tuple[list[str], list[str]]:
    """Return (rejections, warnings). Any rejection means "never auto-select"."""
    rejected: list[str] = []
    warnings: list[str] = []
    if task_type is not None and task_type not in entry.task_types:
        rejected.append(f"task {task_type} not supported")
    if entry.model_family != model_family:
        rejected.append(f"family {entry.model_family} is not {model_family}")
    # Class-head compatibility: the trainer truncates/pads the pretrained head to the
    # dataset's class count (rtdetr.load_pretrained), so any count >= 1 fits. The
    # dataset file format is irrelevant to the weights (COCO and YOLO both work).
    if num_classes is not None and num_classes < 1:
        rejected.append("dataset has no classes")
    if entry.min_vram_gb > 0:
        if not device.is_gpu:
            rejected.append(f"needs {entry.min_vram_gb:g} GB VRAM, no GPU available")
        elif device.memory_gb is not None and entry.min_vram_gb > device.memory_gb:
            rejected.append(f"needs {entry.min_vram_gb:g} GB VRAM, GPU has {device.memory_gb:g} GB")
    else:
        warnings.append("VRAM need not measured")
    if entry.commercial_use is False:
        rejected.append("licence forbids commercial use")
    if not entry.offered:
        rejected.append("not offered yet (no verified sha256/URL)")
    if not entry.weights_license_verified:
        warnings.append("weights licence VERIFY")
    return rejected, warnings


def rank_checkpoints(
    catalog: list[CatalogEntry],
    *,
    task_type: str | None,
    model_family: str,
    bucket: str,
    device: DeviceInfo,
    num_classes: int | None = None,
) -> list[dict[str, Any]]:
    """Score catalog entries (documented v1 rules) and flag hard-filter failures.

    Every returned item has ``eligible`` plus ``rejected`` / ``warnings`` lists.
    Entries failing a hard filter stay in the list (the UI explains why) but are
    never auto-selected [Slice D].
    """
    ranked: list[dict[str, Any]] = []
    for entry in catalog:
        if not entry.is_active:
            continue
        if task_type is not None and task_type not in entry.task_types:
            continue
        if entry.model_family != model_family:
            continue
        rejected, warnings = _hard_filters(
            entry,
            task_type=task_type,
            model_family=model_family,
            num_classes=num_classes,
            device=device,
        )
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
        if entry.min_vram_gb > 0 and device.is_gpu:
            reasons.append(f"needs ~{entry.min_vram_gb:.0f} GB VRAM")
        ranked.append(
            {
                "catalog_id": entry.catalog_id,
                "score": score,
                "reason": "; ".join(reasons),
                "eligible": not rejected,
                "rejected": rejected,
                "warnings": warnings,
                "local": False,
                "size_bytes": entry.size_bytes,
                "license": entry.license,
                "sha256_verified": entry.verified,
                "downloadable": entry.offered,
            }
        )
    # Eligible first, then downloadable, then score: a placeholder/unverified entry
    # must never outrank a real one [doc 07 §3.2, user bug 3].
    ranked.sort(
        key=lambda item: (
            -int(item["eligible"]),
            -int(item["downloadable"]),
            -item["score"],
            item["catalog_id"],
        )
    )
    return ranked


def auto_select(ranked: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Pick the top eligible entry and say why (or why nothing was picked)."""
    if not ranked:
        return None, "no catalog entry matches this task and model family"
    top = ranked[0]
    if not top["eligible"]:
        why = "; ".join(
            f"{item['catalog_id']}: {', '.join(item['rejected'])}" for item in ranked[:3]
        )
        return None, f"no checkpoint passed the hard filters ({why})"
    others = [item["catalog_id"] for item in ranked[1:] if item["eligible"]]
    parts = [
        top["reason"],
        "passed: task, family, class head, VRAM, licence, verified sha256",
    ]
    if top["warnings"]:
        parts.append("note: " + ", ".join(top["warnings"]))
    if others:
        parts.append(f"{len(others)} other eligible candidate(s) scored lower")
    return str(top["catalog_id"]), "; ".join(parts)


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
    preset: str | None = None,
) -> dict[str, Any]:
    """Heuristic defaults shown in the Configure step (ASSUMPTION: not spec'd)."""
    total = sum(summary.class_counts.values()) or len(summary.rows)
    bucket = size_bucket(total)
    max_side = 0
    if summary.image_stats.get("max_wh"):
        max_side = max(int(value) for value in summary.image_stats["max_wh"])

    is_detection = task_type == "object_detection"
    if is_detection:
        # RT-DETR encoder downsamples by 8/16/32, so the input must be a fixed
        # multiple of 32 (never shrink to the image size) [S14b].
        imgsz = 640
    else:
        imgsz = 640
        if max_side and max_side < 640:
            imgsz = max(32, (max_side + 31) // 32 * 32)
    # RT-DETR needs a conservative batch on small GPUs (4 GB) [S14b]; the
    # exact batch/imgsz ceiling is measured by deploy/gpu_memory_probe.py and
    # updated here once the table exists (D122).
    batch_size = 1 if is_detection else (16 if device.is_gpu else 4)
    lr = {
        "object_detection": 0.0001,  # RT-DETR official AdamW lr (was 0.01) [S14b]
        "image_classification": 0.001,
        "segmentation": 0.01,
        "tabular_classification": None,
        "tabular_regression": None,
    }.get(task_type or "", 0.001)

    resolved_preset = preset or ("full" if device.is_gpu else "dev")
    epochs = 50 if bucket != "tiny" else 20
    if resolved_preset == "dev":
        epochs = 1 if is_detection else 2  # doc 14 §3.4: "1–2 epochs"
    hparams: dict[str, Any] = {
        "preset": resolved_preset,
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": 512 if (is_detection and resolved_preset == "dev") else imgsz,
        "freeze_backbone": is_detection or bucket == "tiny",
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
