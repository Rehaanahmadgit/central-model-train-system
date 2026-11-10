"""Chunked upload API [doc 05 §1].

- POST /uploads                → init (D35: persists upload.json)
- PUT  /uploads/{id}/chunks/{n} → idempotent chunk with optional
  X-Chunk-Sha256 checksum header (D34)
- GET  /uploads/{id}          → received-chunk state (D15)
- POST /uploads/{id}/complete → safe extraction, hash, dedup [doc 03 §3.1]
"""

import re
from typing import Any, cast

from fastapi import APIRouter, Header, Request, status

from app.core.config import Settings
from app.core.errors import ApiError
from app.schemas.uploads import UploadComplete, UploadInit, UploadStatus
from app.services import uploads

router = APIRouter(prefix="/uploads", tags=["uploads"])

UPLOAD_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _check_upload_id(upload_id: str) -> str:
    if not UPLOAD_ID_PATTERN.match(upload_id):
        raise ApiError(400, "INVALID_ID", "upload id must be a UUID4")
    return upload_id


@router.post("", response_model=UploadStatus, status_code=201)
def init(body: UploadInit, request: Request) -> dict[str, Any]:
    return uploads.init_upload(
        _settings(request).home,
        _settings(request),
        filename=body.filename,
        size_bytes=body.size_bytes,
        project_id=body.project_id,
    )


@router.get("/{upload_id}", response_model=UploadStatus)
def get(upload_id: str, request: Request) -> dict[str, Any]:
    return uploads.upload_status(_settings(request).home, _check_upload_id(upload_id))


@router.put("/{upload_id}/chunks/{chunk_index}", response_model=UploadStatus)
async def put_chunk(
    upload_id: str,
    chunk_index: int,
    request: Request,
    x_chunk_sha256: str | None = Header(default=None, alias="X-Chunk-Sha256"),
) -> dict[str, Any]:
    # Never buffer more than one chunk: the declared Content-Length and the real
    # stream are both capped, so a lying client cannot exhaust memory/disk [Slice E].
    limit = _settings(request).chunk_size_bytes
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise ApiError(
            413,
            "CHUNK_TOO_LARGE",
            f"chunk is {int(declared)} bytes; the maximum chunk size is {limit} bytes",
        )
    buffer = bytearray()
    async for block in request.stream():
        buffer.extend(block)
        if len(buffer) > limit:
            raise ApiError(
                413,
                "CHUNK_TOO_LARGE",
                f"chunk exceeds the maximum chunk size of {limit} bytes",
            )
    data = bytes(buffer)
    return uploads.put_chunk(
        _settings(request).home,
        _settings(request),
        upload_id=_check_upload_id(upload_id),
        chunk_index=chunk_index,
        data=data,
        checksum=x_chunk_sha256,
    )


@router.post("/{upload_id}/complete", response_model=UploadComplete, status_code=status.HTTP_200_OK)
def complete(upload_id: str, request: Request) -> dict[str, Any]:
    return uploads.complete_upload(
        _settings(request).home, _settings(request), upload_id=_check_upload_id(upload_id)
    )
