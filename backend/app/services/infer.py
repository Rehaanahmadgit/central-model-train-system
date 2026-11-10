"""Try-it inference orchestration [Slice G].

Nothing here imports torch: detection runs in ``app.inference.detect_worker``, a
throw-away subprocess. User images live in a temp directory that is deleted when
the request ends (never kept, never logged). No multipart dependency: the body is
parsed with the stdlib ``email`` parser.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.core.errors import ApiError
from app.services import runs as runs_service
from app.storage.atomic import read_json
from app.storage.locks import FileLock, LockTimeout

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BODY_BYTES = MAX_IMAGES * MAX_IMAGE_BYTES + 1024 * 1024  # + multipart framing
ALLOWED_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
SAMPLE_COUNT = 6
JOB_TIMEOUT_SECONDS = 300
DEFAULT_THRESHOLD = 0.25
_NAME_SAFE = re.compile(r"[^A-Za-z0-9._ -]+")

# One inference at a time: a second request gets a clear 409 instead of piling up.
_busy = asyncio.Lock()


def check_threshold(raw: object) -> float:
    if raw in (None, ""):
        return DEFAULT_THRESHOLD
    try:
        value = float(str(raw))
    except ValueError:
        raise ApiError(422, "INVALID_THRESHOLD", "threshold must be a number 0..1") from None
    if not 0.0 <= value <= 1.0:
        raise ApiError(422, "INVALID_THRESHOLD", "threshold must be between 0 and 1")
    return value


async def read_body_capped(chunks: AsyncIterator[bytes], declared: int | None) -> bytes:
    """Collect the request body, refusing early on Content-Length and while streaming."""
    too_big = ApiError(
        413,
        "INFER_BODY_TOO_LARGE",
        f"too much data: at most {MAX_IMAGES} images of {MAX_IMAGE_BYTES // (1024 * 1024)} MB each",
    )
    if declared is not None and declared > MAX_BODY_BYTES:
        raise too_big
    buffer = bytearray()
    async for chunk in chunks:
        buffer.extend(chunk)
        if len(buffer) > MAX_BODY_BYTES:
            raise too_big
    return bytes(buffer)


def parse_multipart(
    content_type: str, body: bytes
) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    """``(text fields, [(filename, bytes)])`` from a multipart/form-data body."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ApiError(415, "NOT_MULTIPART", "send the images as multipart/form-data")
    message = BytesParser(policy=policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
    )
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    if not message.is_multipart():
        raise ApiError(400, "BAD_MULTIPART", "could not read the multipart body")
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            files.append((str(filename), bytes(payload)))
        elif isinstance(name, str):
            fields[name] = bytes(payload).decode("utf-8", "replace")
    return fields, files


def validate_image(name: str, data: bytes) -> tuple[str, str]:
    """Return ``(display name, extension)`` or raise a precise 4xx."""
    label = _NAME_SAFE.sub("_", Path(name).name)[:80] or "image"
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError(
            413,
            "IMAGE_TOO_LARGE",
            f"{label} is {len(data) / 1024 / 1024:.1f} MB; the limit is "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB per image",
        )
    try:
        with Image.open(io.BytesIO(data)) as opened:
            fmt = (opened.format or "").upper()
            opened.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ApiError(
            415, "UNSUPPORTED_IMAGE", f"{label} is not a readable PNG/JPG/WEBP"
        ) from None
    if fmt not in ALLOWED_FORMATS:
        raise ApiError(
            415, "UNSUPPORTED_IMAGE", f"{label} is {fmt or 'unknown'}; use PNG, JPG or WEBP"
        )
    return label, ALLOWED_FORMATS[fmt]


def _gpu_busy(home: Path) -> bool:
    """True when a training run holds a GPU slot (its flock is held)."""
    locks = home / "locks"
    for path in sorted(locks.glob("gpu-*.lock")) if locks.is_dir() else []:
        probe = FileLock(path, timeout=0.0)
        try:
            probe.acquire()
        except LockTimeout:
            return True
        probe.release()
    return False


def _require_detection_run(home: Path, run_id: str) -> dict[str, Any]:
    record = runs_service.get_run(home, run_id)
    if record.get("model_family") != "rtdetr":
        raise ApiError(
            422, "INFER_UNSUPPORTED", "Try it supports object-detection (rtdetr) runs only"
        )
    if not (home / "runs" / run_id / "checkpoints" / "best.ckpt").is_file():
        raise ApiError(
            409, "NO_BEST_CHECKPOINT", "this run has no best.ckpt yet; wait for it to train"
        )
    return record


def pick_samples(predictions: dict[str, Any]) -> list[dict[str, Any]]:
    """6 deterministic samples: test split if there is one, else val; evenly spaced by path."""
    images = [image for image in predictions.get("images") or [] if isinstance(image, dict)]
    pool = [image for image in images if image.get("split") == "test"] or [
        image for image in images if image.get("split") == "val"
    ]
    pool.sort(key=lambda image: str(image.get("path")))
    if len(pool) <= SAMPLE_COUNT:
        return pool
    return [pool[index * len(pool) // SAMPLE_COUNT] for index in range(SAMPLE_COUNT)]


async def _run_worker(
    home: Path, run_id: str, items: list[dict[str, str]], threshold: float, settings: Settings
) -> tuple[dict[str, Any], bool]:
    busy_gpu = _gpu_busy(home)
    scratch = home / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="infer-", dir=scratch))
    try:
        out = job_dir / "result.json"
        job = job_dir / "job.json"
        job.write_text(
            json.dumps(
                {
                    "home": str(home),
                    "run_id": run_id,
                    "items": items,
                    "threshold": threshold,
                    "device": settings.device,
                    "avoid_gpu": busy_gpu,
                    "out": str(out),
                }
            ),
            encoding="utf-8",
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "app.inference.detect_worker",
            "--job",
            str(job),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(home),
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), JOB_TIMEOUT_SECONDS)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ApiError(
                504, "INFER_TIMEOUT", "detection took too long and was stopped"
            ) from None
        result = read_json(out) if out.is_file() else None
        if not isinstance(result, dict) or not result.get("ok"):
            detail = (
                str(result.get("error")) if isinstance(result, dict) else stderr.decode()[-300:]
            )
            raise ApiError(500, "INFER_FAILED", f"detection failed: {detail}")
        return result, busy_gpu
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)  # user images never persist


def _device_note(device: dict[str, Any], busy_gpu: bool) -> str:
    if device.get("kind") == "cuda":
        return f"Ran on GPU ({device.get('name')})"
    if busy_gpu:
        return "GPU is busy training, so detection ran on CPU (slower, same result)"
    return "No GPU available, so detection ran on CPU (slower, same result)"


def _response(
    run_id: str, result: dict[str, Any], busy_gpu: bool, threshold: float
) -> dict[str, Any]:
    device = result.get("device") or {}
    return {
        "run_id": run_id,
        "threshold": threshold,
        "class_names": result["class_names"],
        "device": device,
        "device_note": _device_note(device, busy_gpu),
        "images": result["images"],
    }


async def infer_uploaded(
    settings: Settings, run_id: str, files: list[tuple[str, bytes]], threshold: float
) -> dict[str, Any]:
    home = settings.home
    _require_detection_run(home, run_id)
    if not files:
        raise ApiError(400, "NO_IMAGES", "attach at least one image")
    if len(files) > MAX_IMAGES:
        raise ApiError(400, "TOO_MANY_IMAGES", f"at most {MAX_IMAGES} images per request")
    checked = [(*validate_image(name, data), data) for name, data in files]
    if _busy.locked():
        raise ApiError(409, "INFER_BUSY", "another detection is running; try again in a moment")
    async with _busy:
        scratch = home / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        upload_dir = Path(tempfile.mkdtemp(prefix="infer-up-", dir=scratch))
        try:
            items: list[dict[str, str]] = []
            for index, (label, ext, data) in enumerate(checked):
                path = upload_dir / f"{index}{ext}"
                path.write_bytes(data)
                items.append({"key": f"{index}", "path": str(path), "name": label})
            result, busy_gpu = await _run_worker(home, run_id, items, threshold, settings)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)
    names = {item["key"]: item["name"] for item in items}
    body = _response(run_id, result, busy_gpu, threshold)
    for image in body["images"]:
        image["name"] = names[image.pop("key")]
    return body


async def infer_samples(settings: Settings, run_id: str, threshold: float) -> dict[str, Any]:
    home = settings.home
    record = _require_detection_run(home, run_id)
    predictions = read_json(home / "runs" / run_id / "artifacts" / "predictions.json")
    if not isinstance(predictions, dict):
        raise ApiError(404, "PREDICTIONS_NOT_FOUND", "this run has no predictions yet")
    root = runs_service.dataset_root(home, str(record.get("dataset_hash") or "")).resolve()
    samples = pick_samples(predictions)
    items: list[dict[str, str]] = []
    for sample in samples:
        resolved = (root / str(sample.get("path"))).resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            items.append({"key": str(sample["path"]), "path": str(resolved)})
    if not items:
        raise ApiError(404, "NO_SAMPLE_IMAGES", "no sample images are available for this run")
    if _busy.locked():
        raise ApiError(409, "INFER_BUSY", "another detection is running; try again in a moment")
    async with _busy:
        result, busy_gpu = await _run_worker(home, run_id, items, threshold, settings)
    by_path = {str(sample["path"]): sample for sample in samples}
    body = _response(run_id, result, busy_gpu, threshold)
    for image in body["images"]:
        sample = by_path[image.pop("key")]
        image.update({"path": sample["path"], "split": sample.get("split"), "gt": sample.get("gt")})
        image["name"] = Path(str(sample["path"])).name
    return body
