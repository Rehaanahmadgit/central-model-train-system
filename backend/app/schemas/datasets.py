"""Dataset schemas; meta.json fields per doc 04 §1 (+ optional project_id, D33)."""

from pydantic import BaseModel, Field


class Dataset(BaseModel):
    hash: str
    name: str
    size_bytes: int
    file_count: int
    format_detected: str | None  # null until verification writes it (D16)
    created_at: str
    source_filename: str
    project_id: str | None = None
    # Verification status summary [doc 05 §1]; filled in slice 5/6.
    verification_summary: None = None


class SampleCreate(BaseModel):
    kind: str = Field(default="tabular", pattern="^(tabular|classification|yolo)$")
