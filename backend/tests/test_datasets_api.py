"""Datasets API tests: upload → hash → dedup → list → delete [doc 05 §1]."""

import io
import zipfile

from fastapi.testclient import TestClient
from helpers import upload_dataset

from app.storage.atomic import atomic_write_json
from app.storage.jsonl import read_jsonl


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _audit_actions(home, *actions) -> list[dict]:
    records: list[dict] = []
    for path in (home / "audit").glob("audit-*.jsonl"):
        records.extend(read_jsonl(path))
    return [r for r in records if r["action"] in actions]


def test_upload_zip_lists_and_fetches(client: TestClient, home) -> None:
    content = _zip_bytes({"data/train.csv": b"a,b\n1,2\n", "data/test.csv": b"a,b\n"})
    completed = upload_dataset(client, "dataset.zip", content)
    dataset_hash = completed["dataset_hash"]
    assert completed["deduplicated"] is False

    listing = client.get("/api/v1/datasets")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    assert body["items"][0]["hash"] == dataset_hash

    fetched = client.get(f"/api/v1/datasets/{dataset_hash}")
    assert fetched.status_code == 200
    meta = fetched.json()
    assert meta["file_count"] == 2
    assert meta["size_bytes"] == len(content)
    assert meta["format_detected"] is None  # verification fills it (D16)
    assert meta["verification_summary"] is None  # slice 5/6
    assert (home / "datasets" / dataset_hash / "raw" / "data" / "train.csv").is_file()
    assert _audit_actions(home, "dataset.uploaded")


def test_identical_upload_deduplicates(client: TestClient, home) -> None:
    content = b"a,b\n1,2\n"
    first = upload_dataset(client, "data.csv", content)
    second = upload_dataset(client, "data.csv", content)
    assert first["dataset_hash"] == second["dataset_hash"]
    assert second["deduplicated"] is True
    folders = list((home / "datasets").iterdir())
    assert len(folders) == 1


def test_different_content_different_hashes(client: TestClient, home) -> None:
    first = upload_dataset(client, "one.csv", b"a,b\n1,2\n")
    second = upload_dataset(client, "two.csv", b"a,b\n3,4\n")
    assert first["dataset_hash"] != second["dataset_hash"]
    assert len(list((home / "datasets").iterdir())) == 2


def test_project_filter(client: TestClient) -> None:
    project = client.post("/api/v1/projects", json={"name": "p"}).json()
    upload_dataset(client, "data.csv", b"a,b\n", project_id=project["id"])
    upload_dataset(client, "other.csv", b"x,y\n")
    filtered = client.get(f"/api/v1/datasets?project={project['id']}").json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["project_id"] == project["id"]


def test_get_unknown_dataset_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/datasets/{'d' * 64}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_invalid_hash_rejected(client: TestClient) -> None:
    response = client.get("/api/v1/datasets/not-a-hash")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ID"


def test_delete_dataset(client: TestClient, home) -> None:
    completed = upload_dataset(client, "data.csv", b"a,b\n")
    dataset_hash = completed["dataset_hash"]
    response = client.delete(f"/api/v1/datasets/{dataset_hash}")
    assert response.status_code == 204
    assert not (home / "datasets" / dataset_hash).exists()
    assert client.get(f"/api/v1/datasets/{dataset_hash}").status_code == 404
    assert _audit_actions(home, "dataset.deleted")


def test_delete_dataset_in_use_409(client: TestClient, home) -> None:
    completed = upload_dataset(client, "data.csv", b"a,b\n")
    dataset_hash = completed["dataset_hash"]
    run_dir = home / "runs" / "run_active"
    run_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "run.json",
        {
            "id": "run_active",
            "state": "running",
            "dataset_hash": dataset_hash,
            "schema_version": 1,
        },
    )
    response = client.delete(f"/api/v1/datasets/{dataset_hash}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DATASET_IN_USE"
    assert (home / "datasets" / dataset_hash).exists()
