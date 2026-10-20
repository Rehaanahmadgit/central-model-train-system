"""Projects API tests [doc 05 §1]."""

import re

from fastapi.testclient import TestClient

from app.storage.jsonl import read_jsonl

PROJECT_ID_RE = re.compile(r"^proj_\d{8}_\d{6}_[0-9a-f]{4}$")


def _audit_actions(home, *actions) -> list[dict]:
    records: list[dict] = []
    for path in (home / "audit").glob("audit-*.jsonl"):
        records.extend(read_jsonl(path))
    return [r for r in records if r["action"] in actions]


def test_create_project(client: TestClient, home) -> None:
    response = client.post(
        "/api/v1/projects", json={"name": "defect-detection", "description": "factory parts"}
    )
    assert response.status_code == 201
    body = response.json()
    assert PROJECT_ID_RE.match(body["id"])
    assert body["name"] == "defect-detection"
    assert body["description"] == "factory parts"
    assert body["created_at"]
    # Persisted as projects/{id}/project.json [doc 04 §1].
    assert (home / "projects" / body["id"] / "project.json").is_file()
    assert _audit_actions(home, "project.created")


def test_list_projects_paginated(client: TestClient) -> None:
    for index in range(3):
        response = client.post("/api/v1/projects", json={"name": f"project-{index}"})
        assert response.status_code == 201
    page_one = client.get("/api/v1/projects?page=1&page_size=2")
    assert page_one.status_code == 200
    body = page_one.json()
    assert body["total"] == 3
    assert body["page"] == 1
    assert len(body["items"]) == 2
    page_two = client.get("/api/v1/projects?page=2&page_size=2").json()
    assert len(page_two["items"]) == 1


def test_get_project_and_404(client: TestClient) -> None:
    created = client.post("/api/v1/projects", json={"name": "p"}).json()
    assert client.get(f"/api/v1/projects/{created['id']}").json()["name"] == "p"
    missing = client.get("/api/v1/projects/proj_missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "PROJECT_NOT_FOUND"


def test_patch_project(client: TestClient, home) -> None:
    created = client.post("/api/v1/projects", json={"name": "p", "description": "d"}).json()
    response = client.patch(
        f"/api/v1/projects/{created['id']}", json={"name": "renamed"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "renamed"
    assert response.json()["description"] == "d"
    assert _audit_actions(home, "project.updated")


def test_patch_with_empty_body_rejected(client: TestClient) -> None:
    created = client.post("/api/v1/projects", json={"name": "p"}).json()
    response = client.patch(f"/api/v1/projects/{created['id']}", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_delete_project(client: TestClient, home) -> None:
    created = client.post("/api/v1/projects", json={"name": "p"}).json()
    response = client.delete(f"/api/v1/projects/{created['id']}")
    assert response.status_code == 204
    assert not (home / "projects" / created["id"]).exists()
    assert client.get(f"/api/v1/projects/{created['id']}").status_code == 404
    assert _audit_actions(home, "project.deleted")


def test_invalid_id_rejected(client: TestClient) -> None:
    response = client.get("/api/v1/projects/bad!id")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ID"
