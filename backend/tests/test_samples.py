"""Sample dataset endpoint (DIST-11) [doc 16 §2]."""

from fastapi.testclient import TestClient


def test_sample_dataset_creates_tabular_and_dedups(client: TestClient) -> None:
    first = client.post("/api/v1/datasets/sample", json={"kind": "tabular"})
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["dataset_hash"]
    assert body["deduplicated"] is False
    assert body["dataset"]["file_count"] >= 1

    second = client.post("/api/v1/datasets/sample", json={"kind": "tabular"})
    assert second.status_code == 201
    assert second.json()["deduplicated"] is True
    assert second.json()["dataset_hash"] == body["dataset_hash"]


def test_sample_dataset_rejects_unknown_kind(client: TestClient) -> None:
    response = client.post("/api/v1/datasets/sample", json={"kind": "bogus"})
    assert response.status_code == 422  # pydantic pattern rejects it
