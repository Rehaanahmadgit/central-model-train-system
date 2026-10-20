"""Verification schemas [doc 05 §3]."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class VerificationCreate(BaseModel):
    dataset_hash: str = Field(min_length=64, max_length=64)
    model_family: str = Field(min_length=1, max_length=50)
    task_type: str | None = Field(default=None, max_length=50)
    # auto: dev preset on CPU, full on GPU [doc 14 §3.3/§3.4].
    preset: Literal["auto", "dev", "full"] = "auto"


class VerificationStarted(BaseModel):
    """202 response for POST /verifications [doc 05 §3]."""

    id: str
    status: str
    cached: bool = False
    verifier_version: str | None = None


class VerificationOut(BaseModel):
    id: str
    status: str
    cached: bool = False
    verifier_version: str | None = None
    dataset_hash: str | None = None
    model_family: str | None = None
    task_type: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    can_train: bool | None = None
    checks: list[dict[str, Any]] | None = None
    derived: dict[str, Any] | None = None
    schema_version: int | None = None
