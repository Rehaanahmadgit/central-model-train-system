"""Synthetic verification datasets [doc 11 §2: generated, not committed binaries]."""

import io
import json
from pathlib import Path

from PIL import Image


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_yolo_dataset(
    root: Path,
    *,
    corrupt_image: bool = False,
    bad_label: str | None = None,
    imbalance: bool = False,
) -> Path:
    """YOLO-style dataset: images + labels + data.yaml with 2 classes.

    Balanced: 3x class 0, 1x class 1. With imbalance=True: 21x class 0, 1x class 1.
    """
    raw = root / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)
    count = 22 if imbalance else 4
    for index in range(count):
        class_id = 0 if index < count - 1 else 1
        (raw / "images" / f"img{index}.png").write_bytes(_png_bytes((255, 0, 0)))
        label = f"{class_id} 0.25 0.25 0.5 0.5\n"
        if bad_label is not None and index == 1:
            label = bad_label + "\n"
        (raw / "labels" / f"img{index}.txt").write_text(label)
    if corrupt_image:
        (raw / "images" / "img0.png").write_bytes(b"this is not an image")
    (raw / "data.yaml").write_text("names:\n  - cat\n  - dog\n")
    return raw


def make_folder_dataset(root: Path, *, classes: tuple[str, ...] = ("cat", "dog")) -> Path:
    raw = root / "raw"
    for name in classes:
        directory = raw / name
        directory.mkdir(parents=True)
        for index in range(3):
            (directory / f"{name}{index}.png").write_bytes(_png_bytes((0, 255, 0)))
    return raw


def make_coco_multi_split_dataset(
    root: Path,
    *,
    train: int = 69,
    valid: int = 19,
    test: int = 10,
) -> Path:
    """Roboflow-style COCO export: one ``_annotations.coco.json`` per split.

    ``categories`` has a dummy parent (id 0 "fruit", supercategory "none") that
    no annotation uses, plus apple (1) and orange (2). Every image is 64x64 with
    a single [16, 8, 32, 16] pixel bbox, so the normalized box is deterministic.
    """
    raw = root / "raw"
    categories = [
        {"id": 0, "name": "fruit", "supercategory": "none"},
        {"id": 1, "name": "apple", "supercategory": "fruit"},
        {"id": 2, "name": "orange", "supercategory": "fruit"},
    ]
    for folder, count in (("train", train), ("valid", valid), ("test", test)):
        directory = raw / folder
        directory.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        for index in range(count):
            file_name = f"{index}.png"
            (directory / file_name).write_bytes(_png_bytes((255, 0, 0)))
            category_id = 1 if index % 2 == 0 else 2
            images.append(
                {"id": index, "file_name": file_name, "width": 64, "height": 64}
            )
            annotations.append(
                {
                    "id": index + 1,
                    "image_id": index,
                    "category_id": category_id,
                    "bbox": [16, 8, 32, 16],
                }
            )
        (directory / "_annotations.coco.json").write_text(
            json.dumps(
                {"images": images, "annotations": annotations, "categories": categories}
            ),
            encoding="utf-8",
        )
    return raw
