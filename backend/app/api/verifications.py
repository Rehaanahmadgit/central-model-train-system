"""Verifications API [doc 05 §3]."""

import re
from typing import Any, cast

from fastapi import APIRouter, Request

from app.core.config import Settings
from app.core.errors import ApiError
from app.realtime.broadcaster import EventBroadcaster
from app.schemas.verifications import (
    VerificationCreate,
    VerificationOut,
    VerificationStarted,
)
from app.services import verifications

router = APIRouter(prefix="/verifications", tags=["verifications"])

VER_ID_PATTERN = re.compile(r"^ver_[0-9]{8}_[0-9]{6}_[0-9a-f]{4}$")
DATASET_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MODEL_FAMILY_PATTERN = re.compile(r"^[a-z0-9_]{1,50}$")


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _check_ver_id(verification_id: str) -> str:
    if not VER_ID_PATTERN.match(verification_id):
        raise ApiError(400, "INVALID_ID", "verification id format is invalid")
    return verification_id


@router.post("", response_model=VerificationStarted, status_code=202)
async def create(body: VerificationCreate, request: Request) -> dict[str, Any]:
    if not DATASET_HASH_PATTERN.match(body.dataset_hash):
        raise ApiError(400, "INVALID_ID", "dataset id must be a 64-char sha256 hex string")
    if not MODEL_FAMILY_PATTERN.match(body.model_family):
        raise ApiError(400, "INVALID_MODEL_FAMILY", "model family must match ^[a-z0-9_]{1,50}$")
    settings = _settings(request)
    return verifications.create_verification(
        settings.home,
        settings,
        request.app.state.executor,
        cast(EventBroadcaster, request.app.state.broadcaster),
        dataset_hash=body.dataset_hash,
        model_family=body.model_family,
        task_type=body.task_type,
    )


@router.get("/{verification_id}", response_model=VerificationOut)
def get(verification_id: str, request: Request) -> dict[str, Any]:
    return verifications.get_verification(
        _settings(request).home, _check_ver_id(verification_id)
    )


@router.get("/{verification_id}/checks")
def get_checks(verification_id: str, request: Request) -> dict[str, Any]:
    return {
        "verification_id": _check_ver_id(verification_id),
        "checks": verifications.get_checks(_settings(request).home, verification_id),
    }
