"""Synthetic verification datasets [doc 11 §2: generated, not committed binaries]."""

import io
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
