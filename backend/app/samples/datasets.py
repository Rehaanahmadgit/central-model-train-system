"""Synthetic dataset builders [doc 11 §2].

Generated, never committed as binaries. Each builder returns an in-memory
``{relative_path: bytes}`` mapping so the same spec can be:

* written to a temp directory for unit tests (``materialize``), or
* zipped and pushed through the real chunked-upload API (``to_zip``), or
* served later by the DIST-11 "Try sample dataset" button without touching
  the test suite.

Only stdlib + Pillow are used so the module is safe to import from app code.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image

Entries = dict[str, bytes]


# --- primitives -------------------------------------------------------------


def png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (16, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def grayscale_png_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", size, 128).save(buffer, format="PNG")
    return buffer.getvalue()


def materialize(root: Path, entries: Entries) -> Path:
    """Write entries under ``root`` and return the directory."""
    for relative, data in entries.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def to_zip(entries: Entries) -> bytes:
    """Zip entries (sorted for reproducibility) — ready for the upload API."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            archive.writestr(name, entries[name])
    return buffer.getvalue()


def _yolo_row(class_id: int, box: tuple[float, float, float, float]) -> bytes:
    x, y, w, h = box
    return f"{class_id} {x} {y} {w} {h}\n".encode()


# --- builders ---------------------------------------------------------------


def build_yolo(
    *,
    per_class: int = 12,
    classes: tuple[str, ...] = ("cat", "dog"),
    corrupt_image: bool = False,
    missing_label: bool = False,
    bad_class_id: bool = False,
    out_of_bounds: bool = False,
    split: bool = False,
    duplicate_leakage: bool = False,
    empty_labels: bool = False,
) -> Entries:
    """YOLO detection layout: ``images/`` + ``labels/`` + ``data.yaml``."""
    entries: Entries = {}
    index = 0
    for class_id in range(len(classes)):
        for _ in range(per_class):
            prefix = ""
            if split:
                prefix = "train/" if index % 4 != 3 else "val/"
            image = f"{prefix}images/img{index}.png"
            label = f"{prefix}labels/img{index}.txt"
            entries[image] = png_bytes((20 + class_id * 40, 60, 120 + index))
            effective_id = 99 if (bad_class_id and index == 0) else class_id
            box = (0.1, 0.1, 1.5, 0.2) if (out_of_bounds and index == 1) else (0.25, 0.25, 0.5, 0.5)
            entries[label] = (
                b"" if (empty_labels and index == 0) else _yolo_row(effective_id, box)
            )
            index += 1
    if missing_label:
        del entries["labels/img0.txt"]
    if corrupt_image:
        entries["images/img0.png"] = b"definitely not a PNG"
    if duplicate_leakage:
        entries["train/images/dup.png"] = entries["images/img1.png"]
        entries["train/labels/dup.txt"] = entries["labels/img1.txt"]
        entries["val/images/dup.png"] = entries["images/img1.png"]
        entries["val/labels/dup.txt"] = entries["labels/img1.txt"]
    entries["data.yaml"] = ("names:\n" + "".join(f"  - {name}\n" for name in classes)).encode()
    return entries


def build_classification(
    *,
    per_class: int = 12,
    minority_count: int = 3,
    classes: tuple[str, ...] = ("cat", "dog"),
    corrupt_image: bool = False,
    single_class: bool = False,
    imbalance: bool = False,
    grayscale: bool = False,
    split: bool = False,
    size: tuple[int, int] = (16, 16),
) -> Entries:
    """Folder-per-class image classification layout (optionally train/val)."""
    if single_class:
        classes = (classes[0],)
    entries: Entries = {}
    for class_id, name in enumerate(classes):
        count = per_class
        if imbalance and class_id > 0:
            count = minority_count
        for index in range(count):
            data = (
                grayscale_png_bytes(size)
                if grayscale
                else png_bytes((class_id * 90 % 256, 30 + index % 200, 200), size=size)
            )
            prefix = ""
            if split:
                prefix = "train/" if index % 5 != 0 else "val/"
            entries[f"{prefix}{name}/{name}{index}.png"] = data
    if corrupt_image:
        entries[f"{classes[0]}/{classes[0]}0.png"] = b"not an image"
    return entries


def build_coco(*, per_class: int = 12, classes: tuple[str, ...] = ("cat", "dog")) -> Entries:
    """COCO detection layout: images + one annotations json."""
    entries: Entries = {}
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    index = 0
    for class_id in range(1, len(classes) + 1):
        for _ in range(per_class):
            file_name = f"images/{index}.png"
            entries[file_name] = png_bytes((class_id * 60, 80, 40 + index), size=(64, 64))
            images.append({"id": index, "file_name": f"{index}.png", "width": 64, "height": 64})
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": index,
                    "category_id": class_id,
                    "bbox": [8, 8, 32, 32],
                }
            )
            annotation_id += 1
            index += 1
    entries["annotations.json"] = json.dumps(
        {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": class_id, "name": name}
                for class_id, name in enumerate(classes, start=1)
            ],
        }
    ).encode()
    return entries


def build_image_csv(
    *, per_class: int = 12, classes: tuple[str, ...] = ("cat", "dog"), missing_file: bool = False
) -> Entries:
    """CSV (path,label) pointing at images in the same tree."""
    entries: Entries = {}
    rows = ["path,label"]
    index = 0
    for name in classes:
        for _ in range(per_class):
            relative = f"images/{index}.png"
            entries[relative] = png_bytes((index * 20 % 255, 120, 40))
            rows.append(f"{relative},{name}")
            index += 1
    if missing_file:
        rows.append("images/does-not-exist.png,cat")
    entries["labels.csv"] = ("\n".join(rows) + "\n").encode()
    return entries


def build_tabular(
    *,
    rows: int = 30,
    missing_values: bool = False,
    constant_column: bool = False,
    high_cardinality: bool = False,
    drop_target: bool = False,
) -> Entries:
    """Tabular CSV with a ``target`` column."""
    header = "feature_a,feature_b,feature_c"
    lines = [header if drop_target else f"{header},target"]
    for index in range(rows):
        feature_a = "1" if constant_column else str(index % 7)
        feature_b = "" if (missing_values and index % 5 == 0) else f"{index * 0.5:.2f}"
        feature_c = f"id-{index}" if high_cardinality else str(index % 3)
        target = "yes" if index % 2 == 0 else "no"
        if drop_target:
            lines.append(f"{feature_a},{feature_b},{feature_c}")
        else:
            lines.append(f"{feature_a},{feature_b},{feature_c},{target}")
    return {"data.csv": ("\n".join(lines) + "\n").encode()}


def build_empty() -> Entries:
    """A valid archive that contains no usable dataset files."""
    return {"README.txt": b"this archive has no dataset in it\n"}


# --- named specs (doc 11 §2 "expected verification outcome" table) ----------

SPECS: dict[str, tuple[Callable[[], Entries], str]] = {
    "good_yolo": (build_yolo, "passed_with_warnings"),
    "good_yolo_split": (lambda: build_yolo(split=True), "passed_with_warnings"),
    "good_cls": (build_classification, "passed_with_warnings"),
    "good_coco": (build_coco, "passed_with_warnings"),
    "good_cls_split": (
        lambda: build_classification(per_class=160, split=True, size=(64, 64)),
        "passed",
    ),
    "good_image_csv": (build_image_csv, "passed_with_warnings"),
    "good_tabular": (build_tabular, "passed_with_warnings"),
    "corrupt_image": (lambda: build_yolo(corrupt_image=True), "failed"),
    "missing_label": (lambda: build_yolo(missing_label=True), "failed"),
    "bad_class_id": (lambda: build_yolo(bad_class_id=True), "failed"),
    "out_of_bounds_bbox": (lambda: build_yolo(out_of_bounds=True), "failed"),
    "empty_labels": (lambda: build_yolo(empty_labels=True), "passed_with_warnings"),
    "imbalanced": (
        lambda: build_classification(per_class=70, minority_count=3, imbalance=True),
        "passed_with_warnings",
    ),
    "single_class": (lambda: build_classification(single_class=True), "failed"),
    "duplicate_leakage": (lambda: build_yolo(duplicate_leakage=True), "failed"),
    "empty_dataset": (build_empty, "failed"),
    "csv_missing_file": (lambda: build_image_csv(missing_file=True), "failed"),
    "tabular_missing_target": (lambda: build_tabular(drop_target=True), "failed"),
}

# Model family each spec is verified against (doc 06 §4 task_model_match).
SPEC_MODEL_FAMILY: dict[str, str] = {
    "good_cls": "resnet",
    "good_cls_split": "resnet",
    "single_class": "resnet",
    "imbalanced": "resnet",
    "good_image_csv": "resnet",
    "good_tabular": "random_forest",
    "tabular_missing_target": "random_forest",
}


def spec_model_family(name: str) -> str:
    return SPEC_MODEL_FAMILY.get(name, "rtdetr")


def build_sample_zip(name: str = "good_yolo") -> bytes:
    """Zip for the DIST-11 sample-dataset flow (no test fixtures involved)."""
    if name not in SPECS:
        raise KeyError(f"unknown sample dataset {name!r}; known: {sorted(SPECS)}")
    return to_zip(SPECS[name][0]())


def spec_names() -> Iterable[str]:
    return SPECS.keys()
