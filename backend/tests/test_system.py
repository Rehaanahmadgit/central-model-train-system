"""System endpoint tests [doc 05 §7, doc 14 §3.2]."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.storage.atomic import atomic_write_json


def test_system_endpoint_shape(home: Path) -> None:
    with TestClient(create_app(Settings(home=home, log_level="DEBUG"))) as client:
        response = client.get("/api/v1/system")
    assert response.status_code == 200
    body = response.json()
    assert body["running_jobs"] == []
    assert body["queued_jobs"] == []
    assert body["recovery_warnings"] == []
    assert {"total_bytes", "used_bytes", "free_bytes"} <= set(body["disk"])
    assert body["gpu"] == {"available": False, "message": "No GPU detected"}


def test_system_lists_running_job(home: Path) -> None:
    with TestClient(create_app(Settings(home=home, log_level="DEBUG"))) as client:
        # Created after startup so crash recovery does not mark it interrupted.
        run_dir = home / "runs" / "run_active"
        run_dir.mkdir(parents=True)
        atomic_write_json(
            run_dir / "run.json",
            {"id": "run_active", "name": "hello-world", "state": "running", "schema_version": 1},
        )
        body = client.get("/api/v1/system").json()
    assert [job["id"] for job in body["running_jobs"]] == ["run_active"]
