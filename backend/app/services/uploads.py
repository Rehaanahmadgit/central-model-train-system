"""Chunked, resumable upload service [doc 05 §1, doc 03 §3.1, D15/D34/D35].

Flow: POST /uploads → PUT chunks (idempotent, optional checksum) →
POST complete → safe extraction → dataset hash → dedup into datasets/{hash}.
"""

import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import ApiError
from app.storage.archive import (
    KIND_CSV,
    KIND_PARQUET,
    ArchiveTooLargeError,
    ArchiveUnsafeError,
    ExtractLimits,
    detect_kind,
    extract_archive,
)
from app.storage.atomic import atomic_write_bytes, atomic_write_json, read_json
from app.storage.audit import append_audit
from app.storage.dataset import (
    dataset_exists,
    dataset_hash_from_dir,
    read_dataset_meta,
    write_dataset_meta,
)
from app.storage.ids import new_upload_id
from app.storage.locks import FileLock


def _upload_dir(home: Path, upload_id: str) -> Path:
    return home / "tmp" / "uploads" / upload_id


def _load_upload(home: Path, upload_id: str) -> dict[str, Any]:
    path = _upload_dir(home, upload_id) / "upload.json"
    if not path.is_file():
        raise ApiError(404, "UPLOAD_NOT_FOUND", f"upload {upload_id} does not exist")
    try:
        data = read_json(path)
    except (OSError, ValueError):
        raise ApiError(500, "UPLOAD_CORRUPT", f"upload {upload_id} is unreadable") from None
    if not isinstance(data, dict):
        raise ApiError(500, "UPLOAD_CORRUPT", f"upload {upload_id} is unreadable")
    return data


def _received_chunks(home: Path, upload_id: str) -> dict[int, int]:
    directory = _upload_dir(home, upload_id)
    chunks: dict[int, int] = {}
    for path in directory.glob("*.chunk"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        chunks[index] = path.stat().st_size
    return chunks


def init_upload(
    home: Path,
    settings: Settings,
    *,
    filename: str,
    size_bytes: int,
    project_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    # Filename must be a plain basename (no separators, not a traversal).
    if filename in ("", ".", "..") or Path(filename).name != filename:
        raise ApiError(400, "INVALID_FILENAME", "filename must be a plain file name")
    if size_bytes > settings.max_upload_size_bytes:
        limit_gb = settings.max_upload_size_bytes / 1024**3
        raise ApiError(
            413,
            "UPLOAD_TOO_LARGE",
            (
                f"upload is {size_bytes / 1024**3:.2f} GB; this instance accepts at most "
                f"{limit_gb:.2f} GB (operator setting TRAINHUB_MAX_UPLOAD_SIZE_BYTES)"
            ),
            {"size_bytes": size_bytes, "max_bytes": settings.max_upload_size_bytes},
        )
    upload_id = new_upload_id()
    directory = _upload_dir(home, upload_id)
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "upload_id": upload_id,
        "filename": filename,
        "size_bytes": size_bytes,
        "chunk_size": settings.chunk_size_bytes,
        "created_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "project_id": project_id,
        "schema_version": 1,
    }
    atomic_write_json(directory / "upload.json", record)
    return {
        "upload_id": upload_id,
        "filename": filename,
        "size_bytes": size_bytes,
        "chunk_size": settings.chunk_size_bytes,
        "received_bytes": 0,
        "received_chunks": [],
    }


def upload_status(home: Path, upload_id: str) -> dict[str, Any]:
    info = _load_upload(home, upload_id)
    chunks = _received_chunks(home, upload_id)
    return {
        "upload_id": upload_id,
        "filename": info["filename"],
        "size_bytes": info["size_bytes"],
        "chunk_size": info["chunk_size"],
        "received_bytes": sum(chunks.values()),
        "received_chunks": sorted(chunks),
    }


def put_chunk(
    home: Path,
    settings: Settings,
    *,
    upload_id: str,
    chunk_index: int,
    data: bytes,
    checksum: str | None = None,
) -> dict[str, Any]:
    info = _load_upload(home, upload_id)
    chunk_size = int(info["chunk_size"])
    size_bytes = int(info["size_bytes"])
    expected_count = (size_bytes + chunk_size - 1) // chunk_size
    if chunk_index < 0 or chunk_index >= expected_count:
        raise ApiError(400, "INVALID_CHUNK_INDEX", f"chunk index out of range: {chunk_index}")
    chunks = _received_chunks(home, upload_id)
    received_without_this = sum(size for index, size in chunks.items() if index != chunk_index)
    if received_without_this + len(data) > size_bytes:
        raise ApiError(413, "UPLOAD_TOO_LARGE", "chunks exceed the declared upload size")
    if checksum is not None and hashlib.sha256(data).hexdigest() != checksum.lower():
        raise ApiError(400, "CHUNK_CHECKSUM_MISMATCH", "chunk sha256 does not match the header")
    # Idempotent: re-PUT of the same chunk simply overwrites it [doc 05 §1].
    atomic_write_bytes(_upload_dir(home, upload_id) / f"{chunk_index:08d}.chunk", data)
    return upload_status(home, upload_id)


def complete_upload(home: Path, settings: Settings, *, upload_id: str) -> dict[str, Any]:
    info = _load_upload(home, upload_id)
    directory = _upload_dir(home, upload_id)
    # Single completer per upload [doc 04 §3 rule 3].
    with FileLock(home / "locks" / f"upload-{upload_id}.lock"):
        chunks = _received_chunks(home, upload_id)
        chunk_size = int(info["chunk_size"])
        size_bytes = int(info["size_bytes"])
        expected_count = (size_bytes + chunk_size - 1) // chunk_size
        missing = [i for i in range(expected_count) if i not in chunks]
        if missing:
            raise ApiError(
                409,
                "INCOMPLETE_UPLOAD",
                "not all chunks have been uploaded",
                details={"missing_chunks": missing},
            )
        received = sum(chunks.values())
        if received != size_bytes:
            raise ApiError(
                400,
                "UPLOAD_SIZE_MISMATCH",
                f"received {received} bytes but {size_bytes} were declared",
            )

        assembled = directory / "upload.data"
        with assembled.open("wb") as out:
            for index in sorted(chunks):
                with (directory / f"{index:08d}.chunk").open("rb") as src:
                    shutil.copyfileobj(src, out, length=1024 * 1024)

        kind = detect_kind(assembled)
        extracted = directory / "extracted"
        extracted.mkdir(exist_ok=False)
        try:
            if kind in (KIND_CSV, KIND_PARQUET):
                # Single-file dataset; copies under the declared filename.
                shutil.copyfile(assembled, extracted / str(info["filename"]))
            else:
                limits = ExtractLimits(
                    max_files=settings.max_extract_files,
                    max_total_bytes=settings.max_extract_total_bytes,
                    zip_bomb_ratio=settings.zip_bomb_ratio,
                    zip_bomb_floor_bytes=settings.zip_bomb_floor_bytes,
                )
                extract_archive(assembled, extracted, limits)
        except ArchiveTooLargeError as exc:
            raise ApiError(413, "ARCHIVE_TOO_LARGE", str(exc)) from None
        except ArchiveUnsafeError as exc:
            raise ApiError(400, "ARCHIVE_UNSAFE", str(exc)) from None

        file_count = sum(1 for path in extracted.rglob("*") if path.is_file())
        if file_count == 0:
            raise ApiError(400, "EMPTY_DATASET", "the upload contains no files")

        dataset_hash = dataset_hash_from_dir(extracted)
        deduplicated = dataset_exists(home, dataset_hash)
        if deduplicated:
            # Identical content already exists → reuse it [doc 03 §3.1].
            meta = read_dataset_meta(home, dataset_hash) or {}
        else:
            target = home / "datasets" / dataset_hash
            target.mkdir(parents=True, exist_ok=False)
            extracted.rename(target / "raw")
            meta = write_dataset_meta(
                home,
                dataset_hash=dataset_hash,
                name=str(info["filename"]),
                size_bytes=size_bytes,
                file_count=file_count,
                source_filename=str(info["filename"]),
                project_id=info.get("project_id"),
            )
        append_audit(
            home,
            action="dataset.uploaded",
            entity_type="dataset",
            entity_id=dataset_hash,
            details={
                "name": str(info["filename"]),
                "size_bytes": size_bytes,
                "file_count": file_count,
                "deduplicated": deduplicated,
            },
        )
        shutil.rmtree(directory, ignore_errors=True)  # staging cleanup
    return {"dataset_hash": dataset_hash, "deduplicated": deduplicated, "dataset": meta}
