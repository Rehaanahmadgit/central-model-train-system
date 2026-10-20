"""Verification schemas [doc 05 §3]."""

from pydantic import BaseModel, Field


class VerificationCreate(BaseModel):
    dataset_hash: str = Field(min_length=64, max_length=64)
    model_family: str = Field(min_length=1, max_length=50)
    task_type: str | None = Field(default=None, max_length=50)


class VerificationOut(BaseModel):
    id: str
    status: str
