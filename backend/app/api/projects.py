"""Projects API [doc 05 §1]."""

import re
from typing import Any, cast

from fastapi import APIRouter, Query, Request, Response, status

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.pagination import paginate
from app.schemas.projects import Project, ProjectCreate, ProjectUpdate
from app.services import projects

router = APIRouter(prefix="/projects", tags=["projects"])

# Path parameters must match the documented id pattern and nothing else [doc 10 §5].
ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,100}$")


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _check_id(project_id: str) -> str:
    if not ID_PATTERN.match(project_id):
        raise ApiError(400, "INVALID_ID", "project id must match ^[A-Za-z0-9_-]{1,100}$")
    return project_id


@router.post("", response_model=Project, status_code=201)
def create(body: ProjectCreate, request: Request) -> dict[str, Any]:
    return projects.create_project(
        _settings(request).home, name=body.name, description=body.description
    )


@router.get("")
def list_all(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    return paginate(projects.list_projects(_settings(request).home), page, page_size)


@router.get("/{project_id}", response_model=Project)
def get(project_id: str, request: Request) -> dict[str, Any]:
    return projects.get_project(_settings(request).home, _check_id(project_id))


@router.patch("/{project_id}", response_model=Project)
def update(project_id: str, body: ProjectUpdate, request: Request) -> dict[str, Any]:
    return projects.update_project(
        _settings(request).home,
        _check_id(project_id),
        name=body.name,
        description=body.description,
    )


@router.delete("/{project_id}", status_code=204)
def delete(project_id: str, request: Request) -> Response:
    projects.delete_project(_settings(request).home, _check_id(project_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
