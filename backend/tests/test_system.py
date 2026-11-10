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
    assert body["gpu"]["available"] is False
    assert body["gpu"]["message"] == "No GPU detected"


def test_system_reports_a_real_gpu(home: Path, monkeypatch) -> None:
    gpu = {"kind": "cuda", "name": "NVIDIA Test", "memory_gb": 4.0, "reason": "CUDA runtime 12.6"}
    monkeypatch.setattr("app.api.system.probe_device", lambda _requested: gpu)
    with TestClient(create_app(Settings(home=home, log_level="DEBUG"))) as client:
        body = client.get("/api/v1/system").json()
    assert body["gpu"] == {
        "available": True,
        "name": "NVIDIA Test",
        "memory_gb": 4.0,
        "message": "GPU: NVIDIA Test",
    }


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


def test_api_process_never_imports_torch_for_system(home: Path) -> None:
    """Regression: importing torch in the API process made forked verification hang."""
    import subprocess
    import sys

    code = (
        "import sys; from fastapi.testclient import TestClient; "
        "from app.core.config import Settings; from app.main import create_app; "
        f"c = TestClient(create_app(Settings(home=r'{home}'))); "
        "c.__enter__(); c.get('/api/v1/system'); "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert out.stdout.strip().splitlines()[-1] == "False", out.stdout + out.stderr
