"""Verification API + WebSocket integration tests [doc 05 §3, doc 11 §2]."""

import io
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fixtures.make_datasets import build_yolo, to_zip
from helpers import upload_dataset
from starlette.websockets import WebSocketDisconnect
from verification_fixtures import _png_bytes  # type: ignore[attr-defined]

from app.storage.jsonl import read_jsonl


def _zip_dataset(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _good_yolo_zip() -> bytes:
    return to_zip(build_yolo())


def _wait_for_verification(client: TestClient, verification_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/verifications/{verification_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError("verification did not finish in time")


def test_verification_full_flow_passes(client: TestClient, home: Path) -> None:
    completed = upload_dataset(client, "yolo.zip", _good_yolo_zip())
    response = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    )
    assert response.status_code == 202
    verification_id = response.json()["id"]
    assert response.json()["status"] == "running"

    body = _wait_for_verification(client, verification_id)
    # A good dataset on a CPU-only laptop warns about hardware/size, not data.
    assert body["status"] == "passed_with_warnings"
    assert body["can_train"] is True
    assert body["cached"] is False
    assert body["verifier_version"] == "1.3.0"
    assert body["finished_at"] is not None
    names = [check["name"] for check in body["checks"]]
    assert len(names) == 24
    assert names[0] == "archive_integrity"
    assert names[-1] == "resource_estimate"
    assert {"label_file_pairing", "leakage_between_splits", "min_samples_per_class"} <= set(names)

    # Derived facts + normalized manifest are cached once [doc 06 §6].
    derived = body["derived"]
    assert derived["task_type"] == "object_detection"
    assert derived["format"] == "yolo"
    assert derived["num_images"] == 24
    assert derived["task_detection_reason"]
    # RT-DETR input is a fixed multiple of 32, never shrunk to image size [S14b].
    assert derived["recommended_hparams"]["imgsz"] == 512
    expected_manifest = f"datasets/{completed['dataset_hash']}/manifest.parquet"
    assert derived["split_manifest_path"] == expected_manifest
    assert (home / "datasets" / completed["dataset_hash"] / "manifest.parquet").is_file()
    assert (home / "verifications" / verification_id / "report.json").is_file()

    checks_file = home / "verifications" / verification_id / "checks.jsonl"
    records = list(read_jsonl(checks_file))
    assert len(records) == 24
    assert [record["seq"] for record in records] == list(range(1, 25))


def test_verification_fails_on_corrupt_dataset(client: TestClient) -> None:
    entries = {
        "images/img0.png": b"not an image",
        "images/img1.png": _png_bytes((255, 0, 0)),
        "images/img2.png": _png_bytes((0, 255, 0)),
        "labels/img0.txt": b"0 0.25 0.25 0.5 0.5\n",
        "labels/img1.txt": b"0 0.25 0.25 0.5 0.5\n",
        "labels/img2.txt": b"1 0.25 0.25 0.5 0.5\n",
        "data.yaml": b"names:\n  - cat\n  - dog\n",
    }
    completed = upload_dataset(client, "bad.zip", _zip_dataset(entries))
    verification_id = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    ).json()["id"]
    body = _wait_for_verification(client, verification_id)
    assert body["status"] == "failed"
    assert body["can_train"] is False
    corrupt = next(check for check in body["checks"] if check["name"] == "corrupt_files")
    assert corrupt["status"] == "fail"
    assert corrupt["samples"]


def test_verification_warns_on_imbalance(client: TestClient) -> None:
    entries: dict[str, bytes] = {"data.yaml": b"names:\n  - cat\n  - dog\n"}
    for index in range(41):
        entries[f"images/img{index}.png"] = _png_bytes((255, 0, 0))
        entries[f"labels/img{index}.txt"] = b"0 0.25 0.25 0.5 0.5\n"
    entries["images/last.png"] = _png_bytes((0, 255, 0))
    entries["labels/last.txt"] = b"1 0.25 0.25 0.5 0.5\n"
    entries["images/last2.png"] = _png_bytes((0, 0, 255))
    entries["labels/last2.txt"] = b"1 0.25 0.25 0.5 0.5\n"
    completed = upload_dataset(client, "imbalanced.zip", _zip_dataset(entries))
    verification_id = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    ).json()["id"]
    body = _wait_for_verification(client, verification_id)
    assert body["status"] == "passed_with_warnings"
    distribution = next(check for check in body["checks"] if check["name"] == "class_distribution")
    assert distribution["status"] == "warn"


def test_verification_unknown_dataset_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": "a" * 64, "model_family": "rtdetr"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_verification_unknown_model_family_400(client: TestClient) -> None:
    completed = upload_dataset(client, "data.csv", b"a,b\n1,2\n")
    response = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "mystery_model"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNKNOWN_MODEL_FAMILY"


def test_verification_writes_audit_lines(client: TestClient, home: Path) -> None:
    """verification.started/finished land in the person-free audit trail [doc 10 §6]."""
    completed = upload_dataset(client, "yolo.zip", _good_yolo_zip())
    verification_id = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    ).json()["id"]
    body = _wait_for_verification(client, verification_id)
    assert body["status"] == "passed_with_warnings"
    records = [
        record
        for path in sorted((home / "audit").glob("audit-*.jsonl"))
        for record in read_jsonl(path)
    ]
    actions = [
        record["action"] for record in records if record.get("entity_id") == verification_id
    ]
    assert actions == ["verification.started", "verification.finished"]


def test_verification_checks_endpoint(client: TestClient) -> None:
    completed = upload_dataset(client, "yolo.zip", _good_yolo_zip())
    verification_id = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    ).json()["id"]
    _wait_for_verification(client, verification_id)
    body = client.get(f"/api/v1/verifications/{verification_id}/checks").json()
    assert len(body["checks"]) == 24


def test_verification_ws_streams_checks(client: TestClient) -> None:
    completed = upload_dataset(client, "yolo.zip", _good_yolo_zip())
    verification_id = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    ).json()["id"]
    with client.websocket_connect(f"/ws/verifications/{verification_id}") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["type"] == "snapshot"
        names: set[str] = set()
        types: set[str] = set()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            message = websocket.receive_json()
            types.add(message["type"])
            if message["type"] == "check":
                names.add(message["data"]["name"])
            if message["type"] == "state" and message["data"]["to"] in (
                "passed",
                "passed_with_warnings",
                "failed",
            ):
                break
        assert "check_started" in types
        assert {"archive_integrity", "format_supported", "corrupt_files", "label_validity"} <= names
        assert len(names) == 24


def test_verification_ws_on_finished_stream_closes(client: TestClient) -> None:
    completed = upload_dataset(client, "yolo.zip", _good_yolo_zip())
    verification_id = client.post(
        "/api/v1/verifications",
        json={"dataset_hash": completed["dataset_hash"], "model_family": "rtdetr"},
    ).json()["id"]
    _wait_for_verification(client, verification_id)
    with client.websocket_connect(f"/ws/verifications/{verification_id}") as websocket:
        snapshot = websocket.receive_json()
        assert snapshot["type"] == "snapshot"
        assert snapshot["data"]["verification"]["status"] == "passed_with_warnings"
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()
