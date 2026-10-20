"""Runs schemas [doc 05 §5]."""

from typing import Any

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    verification_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    config: dict[str, Any] | None = None
    checkpoint_catalog_id: str | None = None
    from_scratch: bool = False


class RunOut(BaseModel):
    id: str
    name: str
    state: str
    model_family: str
    dataset_hash: str
    task_type: str | None = None
    queued_at: str | None = None
    schema_version: int


class RunRetry(BaseModel):
    resume_from_last: bool = False
    name: str | None = None
    config: dict[str, Any] | None = None
