"""Chunked upload schemas [doc 05 §1, D15]."""

from pydantic import BaseModel, Field


class UploadInit(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    # D33: optional project linkage (additive field on meta.json).
    project_id: str | None = Field(default=None, min_length=1, max_length=100)


class UploadStatus(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int
    chunk_size: int
    received_bytes: int
    received_chunks: list[int]


class UploadComplete(BaseModel):
    dataset_hash: str
    deduplicated: bool
