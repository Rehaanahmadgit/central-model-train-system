"""Fake demo run + system endpoint tests [doc 12 Phase 0, doc 05 §7]."""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.storage.atomic import atomic_write_json, read_json
from app.storage.jsonl import read_jsonl


def test_fake_run_streams_and_completes(tmp_path: Path) -> None:
    home = tmp_path / "trainhub_home"
    settings = Settings(
        home=home,
        log_level="DEBUG",
        demo_fake_run=True,
        demo_fake_run_steps=6,
        demo_fake_run_interval=0.01,
    )
    with TestClient(create_app(settings)) as client:
        # Wait for the demo run to complete (poll with a deadline).
        deadline = time.monotonic() + 5.0
        run_dirs: list[Path] = []
        while time.monotonic() < deadline:
            run_dirs = list((home / "runs").glob("*/run.json"))
            if run_dirs:
                state = read_json(run_dirs[0]).get("state")
                if state == "completed":
                    break
            time.sleep(0.05)
        assert run_dirs, "demo run was not created"
        run_id = run_dirs[0].parent.name

        records = list(read_jsonl(home / "runs" / run_id / "events.jsonl"))
        metric_records = [record for record in records if record["type"] == "metric"]
        assert len(metric_records) >= 6 * 3  # three metrics per step
        names = {record["data"]["name"] for record in metric_records}
        assert names == {"train/loss", "val/loss", "lr"}
        seqs = [record["seq"] for record in records]
        assert seqs == sorted(seqs)
        assert [record["type"] for record in records[-2:]] == ["state", "log"]
        assert records[-2]["data"]["to"] == "completed"

        # System endpoint reflects the finished run [doc 05 §7].
        system = client.get("/api/v1/system").json()
        assert system["running_jobs"] == []
        assert system["gpu"]["message"] == "No GPU detected"
        assert system["disk"]["total_bytes"] > 0


def test_system_endpoint_shape(home: Path) -> None:
    settings = Settings(home=home, log_level="DEBUG")
    with TestClient(create_app(settings)) as client:
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
