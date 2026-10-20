"""Chunked upload API tests [doc 05 §1, doc 11 §3b]."""

import hashlib
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _init(client: TestClient, filename: str = "data.csv", size: int = 16) -> dict:
    response = client.post("/api/v1/uploads", json={"filename": filename, "size_bytes": size})
    assert response.status_code == 201, response.text
    return response.json()


def test_init_upload_returns_status(client: TestClient) -> None:
    info = _init(client, "data.csv", 16)
    assert uuid.UUID(info["upload_id"]).version == 4
    assert info["filename"] == "data.csv"
    assert info["size_bytes"] == 16
    assert info["chunk_size"] > 0
    assert info["received_bytes"] == 0
    assert info["received_chunks"] == []


def test_init_rejects_too_large_upload(client: TestClient) -> None:
    response = client.post(
        "/api/v1/uploads", json={"filename": "huge.zip", "size_bytes": 10**15}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


def test_init_rejects_traversal_filename(client: TestClient) -> None:
    for bad in ("../evil.csv", "/etc/passwd", "a/b.csv", ".."):
        response = client.post("/api/v1/uploads", json={"filename": bad, "size_bytes": 10})
        assert response.status_code == 400, bad
        assert response.json()["error"]["code"] == "INVALID_FILENAME"


def test_put_chunk_and_status(client: TestClient) -> None:
    info = _init(client, "data.csv", 16)
    data = b"01234567"
    response = client.put(f"/api/v1/uploads/{info['upload_id']}/chunks/0", content=data)
    assert response.status_code == 200
    status = response.json()
    assert status["received_bytes"] == 8
    assert status["received_chunks"] == [0]

    listing = client.get(f"/api/v1/uploads/{info['upload_id']}").json()
    assert listing["received_bytes"] == 8


def test_put_chunk_is_idempotent(client: TestClient) -> None:
    info = _init(client, "data.csv", 16)
    chunk = b"01234567"
    for _ in range(2):
        response = client.put(f"/api/v1/uploads/{info['upload_id']}/chunks/0", content=chunk)
        assert response.status_code == 200
    assert client.get(f"/api/v1/uploads/{info['upload_id']}").json()["received_bytes"] == 8


def test_put_chunk_checksum_mismatch(client: TestClient) -> None:
    info = _init(client, "data.csv", 16)
    response = client.put(
        f"/api/v1/uploads/{info['upload_id']}/chunks/0",
        content=b"01234567",
        headers={"X-Chunk-Sha256": "0" * 64},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "CHUNK_CHECKSUM_MISMATCH"


def test_put_chunk_index_out_of_range(client: TestClient) -> None:
    info = _init(client, "data.csv", 16)
    response = client.put(f"/api/v1/uploads/{info['upload_id']}/chunks/5", content=b"x")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_CHUNK_INDEX"


def test_put_chunks_exceeding_declared_size(client: TestClient) -> None:
    info = _init(client, "data.csv", 8)
    # Chunk 0 with 9 bytes exceeds the declared 8 → 413.
    response = client.put(f"/api/v1/uploads/{info['upload_id']}/chunks/0", content=b"123456789")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


def test_complete_with_missing_chunks(tmp_path: Path) -> None:
    # 8-byte chunks so a 16-byte upload needs two chunks.
    home = tmp_path / "trainhub_home"
    with TestClient(
        create_app(Settings(home=home, chunk_size_bytes=8, log_level="DEBUG"))
    ) as small_client:
        init = small_client.post("/api/v1/uploads", json={"filename": "data.csv", "size_bytes": 16})
        assert init.status_code == 201
        info = init.json()
        assert info["chunk_size"] == 8
        response = small_client.put(
            f"/api/v1/uploads/{info['upload_id']}/chunks/0", content=b"01234567"
        )
        assert response.status_code == 200
        response = small_client.post(f"/api/v1/uploads/{info['upload_id']}/complete")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "INCOMPLETE_UPLOAD"
    assert error["details"]["missing_chunks"] == [1]


def test_complete_assembles_and_hashes(client: TestClient, home) -> None:
    content = b"a,b\n1,2\n3,4\n"
    info = _init(client, "data.csv", len(content))
    chunk_size = info["chunk_size"]
    for index, start in enumerate(range(0, len(content), chunk_size)):
        part = content[start : start + chunk_size]
        response = client.put(
            f"/api/v1/uploads/{info['upload_id']}/chunks/{index}",
            content=part,
            headers={"X-Chunk-Sha256": hashlib.sha256(part).hexdigest()},
        )
        assert response.status_code == 200
    complete = client.post(f"/api/v1/uploads/{info['upload_id']}/complete")
    assert complete.status_code == 200, complete.text
    body = complete.json()
    assert len(body["dataset_hash"]) == 64
    assert body["deduplicated"] is False
    # Staging cleaned up after completion.
    assert not (home / "tmp" / "uploads" / info["upload_id"]).exists()
    # Raw content landed under datasets/{hash}/raw.
    raw = home / "datasets" / body["dataset_hash"] / "raw" / "data.csv"
    assert raw.read_bytes() == content


def test_unknown_upload_404(client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/v1/uploads/{missing}").status_code == 404
    response = client.post(f"/api/v1/uploads/{missing}/complete")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "UPLOAD_NOT_FOUND"
