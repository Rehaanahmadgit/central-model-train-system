"""Task detection with an explainable reason [doc 06 §5].

Heuristic and stored in the report as ``task_detection_reason`` so the UI can
always tell the user *why* a task was chosen.
"""

from __future__ import annotations

from pathlib import Path

from app.verification.manifest import ManifestSummary

SEGMENTATION_HINTS = ("masks", "masks_train", "segmentations")


def detect_task(
    root: Path, detected_format: str | None, summary: ManifestSummary
) -> tuple[str | None, str]:
    """Return ``(task_type, reason)``; task_type is None when undetectable."""
    if detected_format is None:
        return None, "no recognized dataset format, so no task could be inferred"

    mask_dirs = [
        path.name
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in SEGMENTATION_HINTS
    ]
    if mask_dirs and summary.rows:
        return (
            "segmentation",
            f"found a mask directory {mask_dirs[0]!r} next to image files",
        )

    if detected_format == "yolo":
        boxes = sum(int(row.get("num_boxes") or 0) for row in summary.rows)
        return (
            "object_detection",
            f"YOLO label files contain {boxes} normalized bounding boxes",
        )
    if detected_format == "coco":
        boxes = sum(int(row.get("num_boxes") or 0) for row in summary.rows)
        if boxes:
            return "object_detection", f"COCO annotations contain {boxes} bounding boxes"
        return (
            "image_classification",
            "COCO json has categories but no bounding boxes, so it is "
            "treated as image-level labels",
        )
    if detected_format == "image_folder":
        folders = sorted({Path(row["path"]).parent.name for row in summary.rows})
        return (
            "image_classification",
            f"folder-per-class layout with class folders: {', '.join(folders[:5])}",
        )
    if detected_format == "csv_path_label":
        return (
            "image_classification",
            "CSV pairs image paths with a single label column",
        )
    if detected_format == "tabular":
        target = summary.tabular_schema.get("target")
        if target is None:
            return (
                None,
                "tabular CSV has no target/label column, so the task cannot be inferred",
            )
        distinct = int(summary.tabular_schema.get("target_distinct") or 0)
        rows = int(summary.tabular_schema.get("rows") or 0)
        # Documented heuristic [doc 06 §5]: decide by target dtype/cardinality.
        if distinct <= max(20, int(0.05 * max(rows, 1))):
            return (
                "tabular_classification",
                f"target column {target!r} has {distinct} distinct values over {rows} rows",
            )
        return (
            "tabular_regression",
            f"target column {target!r} is high-cardinality ({distinct} distinct values)",
        )
    return None, f"format {detected_format!r} has no task rule"
