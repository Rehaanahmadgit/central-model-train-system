"""Normalized dataset manifest [doc 06 §6, doc 04 §1].

Built once per verification and written to ``datasets/{hash}/manifest.parquet``;
training reads this instead of the raw folders, so every trainer plugin sees
one consistent format. Nothing here recomputes during training.

Rows are flat and Arrow-typed so the parquet is stable across versions.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image

from app.storage.atomic import atomic_write_bytes

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
    "eval": "test",
}

MANIFEST_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("split", pa.string()),
        ("label", pa.string()),
        ("class_id", pa.int32()),
        ("num_boxes", pa.int32()),
        ("boxes", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("color_mode", pa.string()),
        ("file_sha256", pa.string()),
        ("annotation_path", pa.string()),
    ]
)


@dataclass
class ManifestSummary:
    """Derived facts consumed by checks, the report and training [doc 06 §6]."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    num_samples: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    class_to_id: dict[str, int] = field(default_factory=dict)
    image_stats: dict[str, Any] = field(default_factory=dict)
    object_stats: dict[str, Any] = field(default_factory=dict)
    tabular_schema: dict[str, Any] = field(default_factory=dict)
    files_hashed: int = 0


def split_of(relative: str) -> str:
    """Infer the split from the first path component naming a split [doc 06 §3]."""
    for part in Path(relative).parts[:-1]:
        mapped = SPLIT_ALIASES.get(part.lower())
        if mapped is not None:
            return mapped
    return "unspecified"


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _image_meta(path: Path) -> tuple[int, int, str] | None:
    try:
        with Image.open(path) as image:
            return image.width, image.height, image.mode
    except (OSError, ValueError):
        return None


def _yolo_label_path(root: Path, image: Path) -> Path | None:
    """Resolve ``images/x.png`` → ``labels/x.txt``; None when the label is missing."""
    parts = list(image.relative_to(root).parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            mapped = root.joinpath(*parts).with_suffix(".txt")
            return mapped if mapped.is_file() else None
    sibling = image.with_suffix(".txt")
    return sibling if sibling.is_file() else None


def _yolo_rows(root: Path, images: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in images:
        relative = image.relative_to(root).as_posix()
        label_path = _yolo_label_path(root, image)
        boxes: list[list[float]] = []
        class_id: int | None = None
        label: str | None = None
        if label_path is not None and label_path.is_file():
            for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = raw.split()
                if len(parts) != 5:
                    continue
                try:
                    parsed = [float(value) for value in parts[1:]]
                    parsed_id = int(parts[0])
                except ValueError:
                    continue
                boxes.append(parsed)
                if class_id is None:
                    class_id = parsed_id
        if class_id is not None:
            label = str(class_id)
        rows.append(
            {
                "path": relative,
                "split": split_of(relative),
                "label": label,
                "class_id": class_id,
                "num_boxes": len(boxes),
                "boxes": json.dumps(boxes) if boxes else None,
                "width": None,
                "height": None,
                "color_mode": None,
                "file_sha256": None,
                "annotation_path": (
                    label_path.relative_to(root).as_posix() if label_path is not None else None
                ),
            }
        )
    return rows


def _coco_rows(root: Path, annotation_file: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    categories = {
        entry.get("id"): str(entry.get("name", entry.get("id")))
        for entry in data.get("categories", [])
        if isinstance(entry, dict)
    }
    by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        if isinstance(annotation, dict):
            by_image.setdefault(annotation.get("image_id"), []).append(annotation)
    rows: list[dict[str, Any]] = []
    for image in data.get("images", []):
        if not isinstance(image, dict):
            continue
        file_name = str(image.get("file_name", ""))
        candidate = root / file_name
        relative = candidate.relative_to(root).as_posix() if candidate.is_file() else file_name
        annotations = by_image.get(image.get("id"), [])
        boxes = [annotation.get("bbox") for annotation in annotations if annotation.get("bbox")]
        first = annotations[0].get("category_id") if annotations else None
        label = categories.get(first) if first is not None else None
        rows.append(
            {
                "path": relative,
                "split": split_of(relative),
                "label": label,
                "class_id": int(first) if isinstance(first, int) else None,
                "num_boxes": len(boxes),
                "boxes": json.dumps(boxes) if boxes else None,
                "width": image.get("width"),
                "height": image.get("height"),
                "color_mode": None,
                "file_sha256": None,
                "annotation_path": annotation_file.relative_to(root).as_posix(),
            }
        )
    return rows


def _folder_rows(root: Path, images: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in images:
        relative = image.relative_to(root).as_posix()
        label = image.parent.name
        rows.append(
            {
                "path": relative,
                "split": split_of(relative),
                "label": label,
                "class_id": None,
                "num_boxes": 0,
                "boxes": None,
                "width": None,
                "height": None,
                "color_mode": None,
                "file_sha256": None,
                "annotation_path": None,
            }
        )
    return rows


def _csv_rows(root: Path, csv_file: Path) -> list[dict[str, Any]]:
    try:
        lines = csv_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    path_field = next(
        (name for name in (reader.fieldnames or []) if name.lower() in ("path", "image", "file")),
        None,
    )
    label_field = next(
        (name for name in (reader.fieldnames or []) if name.lower() == "label"), None
    )
    if path_field is None or label_field is None:
        return []
    rows: list[dict[str, Any]] = []
    for record in reader:
        relative = (record.get(path_field) or "").strip()
        label = (record.get(label_field) or "").strip() or None
        rows.append(
            {
                "path": relative,
                "split": split_of(relative),
                "label": label,
                "class_id": None,
                "num_boxes": 0,
                "boxes": None,
                "width": None,
                "height": None,
                "color_mode": None,
                "file_sha256": None,
                "annotation_path": csv_file.relative_to(root).as_posix(),
            }
        )
    return rows


def _tabular_rows(root: Path, csv_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = csv_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    target = _target_column(reader.fieldnames or [])
    for index, record in enumerate(reader):
        rows.append(
            {
                "path": f"{csv_file.relative_to(root).as_posix()}#row={index}",
                "split": "unspecified",
                "label": (record.get(target) if target else None),
                "class_id": None,
                "num_boxes": 0,
                "boxes": None,
                "width": None,
                "height": None,
                "color_mode": None,
                "file_sha256": None,
                "annotation_path": csv_file.relative_to(root).as_posix(),
            }
        )
    return rows


TARGET_NAMES = ("target", "label", "class", "y")


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _target_column(fieldnames: Sequence[str]) -> str | None:
    for name in fieldnames:
        if name.lower() in TARGET_NAMES:
            return name
    return None


def build_manifest(
    root: Path,
    detected_format: str | None,
    *,
    sample_cap: int,
) -> ManifestSummary:
    """Build the full file list plus sampled pixel facts [doc 06 §2 step 2, §7]."""
    files = sorted(path for path in root.rglob("*") if path.is_file())
    images = sorted(
        path
        for path in files
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    summary = ManifestSummary()

    if detected_format == "yolo":
        summary.rows = _yolo_rows(root, images)
    elif detected_format == "coco":
        annotation_file = next((path for path in files if path.suffix.lower() == ".json"), None)
        summary.rows = _coco_rows(root, annotation_file) if annotation_file else []
    elif detected_format in ("image_folder", "csv_path_label"):
        summary.rows = _folder_rows(root, images)
        if detected_format == "csv_path_label":
            csv_file = next((path for path in files if path.suffix.lower() == ".csv"), None)
            if csv_file is not None:
                labelled = _csv_rows(root, csv_file)
                by_path = {row["path"]: row for row in labelled}
                for row in summary.rows:
                    override = by_path.get(row["path"])
                    if override is not None:
                        row["label"] = override["label"]
                        row["annotation_path"] = override["annotation_path"]
    elif detected_format == "tabular":
        csv_file = next((path for path in files if path.suffix.lower() == ".csv"), None)
        if csv_file is not None:
            summary.rows = _tabular_rows(root, csv_file)
            summary.tabular_schema = _tabular_schema(csv_file)

    _fill_pixel_facts(root, summary, images, sample_cap=sample_cap)
    _summarize(summary)
    return summary


def _tabular_schema(csv_file: Path) -> dict[str, Any]:
    try:
        lines = csv_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    if not lines:
        return {}
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    fieldnames = reader.fieldnames or []
    records = list(reader)
    target = _target_column(fieldnames)
    columns: list[dict[str, Any]] = []
    for name in fieldnames:
        values = [(record.get(name) or "").strip() for record in records]
        missing = sum(1 for value in values if value == "")
        distinct = len({value for value in values if value != ""})
        present = [value for value in values if value != ""]
        numeric = bool(present) and all(_is_float(value) for value in present)
        columns.append(
            {
                "name": name,
                "missing": missing,
                "missing_pct": round(100 * missing / max(len(values), 1), 2),
                "distinct": distinct,
                "constant": distinct <= 1,
                "numeric": numeric,
            }
        )
    return {
        "rows": len(records),
        "columns": columns,
        "target": target,
        "target_distinct": next(
            (column["distinct"] for column in columns if column["name"] == target), 0
        ),
    }


def _fill_pixel_facts(
    root: Path, summary: ManifestSummary, images: list[Path], *, sample_cap: int
) -> None:
    """Decode dimensions/mode for a sample; structure stays complete [doc 06 §7]."""
    if not summary.rows or not images:
        return
    paths = [root / row["path"] for row in summary.rows if (root / row["path"]).is_file()]
    sampled = paths[:sample_cap]
    with ThreadPoolExecutor(max_workers=8) as pool:
        metas = list(pool.map(_image_meta, sampled))
        digests = list(pool.map(_sha256, sampled))
    summary.files_hashed = len(sampled)
    meta_by_path: dict[str, tuple[tuple[int, int, str] | None, str | None]] = {}
    for path, meta, digest in zip(sampled, metas, digests, strict=False):
        meta_by_path[path.relative_to(root).as_posix()] = (meta, digest)
    for row in summary.rows:
        meta, digest = meta_by_path.get(row["path"], (None, None))
        if meta is not None:
            row["width"], row["height"], row["color_mode"] = meta
        row["file_sha256"] = digest


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summarize(summary: ManifestSummary) -> None:
    splits: dict[str, int] = {}
    counts: dict[str, int] = {}
    names: list[str] = []
    for row in summary.rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
        label = row.get("label")
        if label is not None:
            key = str(label)
            counts[key] = counts.get(key, 0) + 1
            if key not in names:
                names.append(key)
    summary.num_samples = {
        key: splits[key] for key in ("train", "val", "test", "unspecified") if key in splits
    }
    summary.class_counts = counts
    summary.class_names = sorted(names)
    summary.class_to_id = {name: index for index, name in enumerate(summary.class_names)}

    sizes = [(row["width"], row["height"]) for row in summary.rows if row.get("width")]
    if sizes:
        widths = [int(width) for width, _ in sizes]
        heights = [int(height) for _, height in sizes]
        modes: dict[str, int] = {}
        for row in summary.rows:
            if row.get("color_mode"):
                modes[str(row["color_mode"])] = modes.get(str(row["color_mode"]), 0) + 1
        summary.image_stats = {
            "sampled": len(sizes),
            "min_wh": [min(widths), min(heights)],
            "max_wh": [max(widths), max(heights)],
            "median_w": _percentile(widths, 0.5),
            "median_h": _percentile(heights, 0.5),
            "color_modes": modes,
            "tiny_images": sum(1 for width, height in sizes if width < 32 or height < 32),
            "odd_aspect": sum(
                1 for width, height in sizes if height and not 0.5 <= width / height <= 2.0
            ),
        }

    box_areas: list[float] = []
    for row in summary.rows:
        if not row.get("boxes"):
            continue
        try:
            for box in json.loads(str(row["boxes"])):
                if len(box) == 4:
                    box_areas.append(float(box[2]) * float(box[3]))
        except (ValueError, TypeError):
            continue
    if box_areas:
        summary.object_stats = {
            "boxes": len(box_areas),
            "median_area": round(sorted(box_areas)[len(box_areas) // 2], 5),
            "tiny_boxes": sum(1 for area in box_areas if area < 0.001),
            "empty_images": sum(1 for row in summary.rows if not row.get("num_boxes")),
        }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the manifest atomically as parquet [doc 04 §3 rule 1]."""
    columns: dict[str, list[Any]] = {name: [] for name in MANIFEST_SCHEMA.names}
    for row in rows:
        for name in MANIFEST_SCHEMA.names:
            columns[name].append(row.get(name))
    table = pa.Table.from_pydict(columns, schema=MANIFEST_SCHEMA)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, buffer.getvalue())
