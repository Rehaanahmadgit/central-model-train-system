"""Project service: projects are named folders [doc 00 §3, doc 04 §1]."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.errors import ApiError
from app.storage.atomic import atomic_write_json, read_json
from app.storage.audit import append_audit
from app.storage.ids import new_project_id


def _project_dir(home: Path, project_id: str) -> Path:
    return home / "projects" / project_id


def _load(home: Path, project_id: str) -> dict[str, Any]:
    path = _project_dir(home, project_id) / "project.json"
    if not path.is_file():
        raise ApiError(404, "PROJECT_NOT_FOUND", f"project {project_id} does not exist")
    try:
        data = read_json(path)
    except (OSError, ValueError):
        raise ApiError(500, "PROJECT_CORRUPT", f"project {project_id} is unreadable") from None
    if not isinstance(data, dict):
        raise ApiError(500, "PROJECT_CORRUPT", f"project {project_id} is unreadable")
    return data


def create_project(
    home: Path, *, name: str, description: str, now: datetime | None = None
) -> dict[str, Any]:
    project_id = new_project_id(now)
    created_at = (now or datetime.now()).isoformat(timespec="seconds")
    record = {
        "id": project_id,
        "name": name,
        "description": description,
        "created_at": created_at,
        "schema_version": 1,
    }
    directory = _project_dir(home, project_id)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(directory / "project.json", record)
    # State-changing endpoint → audit line [doc 05 §9]. Action name is an
    # extension of the enumerated list (D38).
    append_audit(
        home,
        action="project.created",
        entity_type="project",
        entity_id=project_id,
        details={"name": name},
    )
    return record


def list_projects(home: Path) -> list[dict[str, Any]]:
    projects_dir = home / "projects"
    if not projects_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in projects_dir.glob("*/project.json"):
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue  # corrupt entries surface in system health later [doc 03 §8]
        if isinstance(data, dict):
            records.append(data)
    records.sort(key=lambda record: record.get("created_at", ""), reverse=True)
    return records


def get_project(home: Path, project_id: str) -> dict[str, Any]:
    return _load(home, project_id)


def update_project(
    home: Path,
    project_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    record = _load(home, project_id)
    if name is not None:
        record["name"] = name
    if description is not None:
        record["description"] = description
    atomic_write_json(_project_dir(home, project_id) / "project.json", record)
    append_audit(
        home,
        action="project.updated",
        entity_type="project",
        entity_id=project_id,
        details={},
    )
    return record


def delete_project(home: Path, project_id: str) -> None:
    _load(home, project_id)  # 404 if missing
    shutil.rmtree(_project_dir(home, project_id))
    append_audit(
        home,
        action="project.deleted",
        entity_type="project",
        entity_id=project_id,
        details={},
    )
