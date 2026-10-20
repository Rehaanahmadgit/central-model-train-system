"""Shared API helpers for tests."""

import hashlib

from fastapi.testclient import TestClient


def upload_dataset(
    client: TestClient,
    filename: str,
    content: bytes,
    *,
    project_id: str | None = None,
) -> dict:
    """Drive the full chunked upload API and return the complete response [doc 05 §1]."""
    body: dict = {"filename": filename, "size_bytes": len(content)}
    if project_id is not None:
        body["project_id"] = project_id
    init = client.post("/api/v1/uploads", json=body)
    assert init.status_code == 201, init.text
    info = init.json()
    chunk_size = info["chunk_size"]
    for index, start in enumerate(range(0, len(content), chunk_size)):
        part = content[start : start + chunk_size]
        response = client.put(
            f"/api/v1/uploads/{info['upload_id']}/chunks/{index}",
            content=part,
            headers={"X-Chunk-Sha256": hashlib.sha256(part).hexdigest()},
        )
        assert response.status_code == 200, response.text
    complete = client.post(f"/api/v1/uploads/{info['upload_id']}/complete")
    assert complete.status_code == 200, complete.text
    return complete.json()
