"""Datasets API [doc 05 §1]."""

import re
from typing import Any, cast

from fastapi import APIRouter, Query, Request, Response, status

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.pagination import paginate
from app.schemas.datasets import Dataset, SampleCreate
from app.services import datasets, samples

router = APIRouter(prefix="/datasets", tags=["datasets"])

DATASET_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _check_hash(dataset_hash: str) -> str:
    if not DATASET_HASH_PATTERN.match(dataset_hash):
        raise ApiError(400, "INVALID_ID", "dataset id must be a 64-char sha256 hex string")
    return dataset_hash


@router.get("")
def list_all(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    project: str | None = Query(default=None, min_length=1, max_length=100),
) -> dict[str, Any]:
    return paginate(
        datasets.list_datasets(_settings(request).home, project_id=project),
        page,
        page_size,
    )


@router.post("/sample", status_code=201)
def create_sample(body: SampleCreate, request: Request) -> dict[str, Any]:
    """Create a tiny synthetic dataset through the normal flow [DIST-11]."""
    return samples.create_sample_dataset(
        _settings(request).home, _settings(request), kind=body.kind
    )


@router.get("/{dataset_hash}", response_model=Dataset)
def get(dataset_hash: str, request: Request) -> dict[str, Any]:
    return datasets.get_dataset(_settings(request).home, _check_hash(dataset_hash))


@router.delete("/{dataset_hash}", status_code=204)
def delete(dataset_hash: str, request: Request) -> Response:
    datasets.delete_dataset(_settings(request).home, _check_hash(dataset_hash))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
