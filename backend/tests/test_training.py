"""S8 tests: real trainers, Job Manager, subprocess supervision [doc 08, doc 11 §3].

The trainers are real (sklearn RandomForest + a tiny PyTorch CNN), no fake
metric publisher. ``kill -9``, crash recovery and the slow control tests are
marked ``@pytest.mark.slow`` per doc 16 FAST-02.
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from fixtures.make_datasets import build_classification, build_tabular, to_zip
from helpers import upload_dataset

from app.core.config import Settings
from app.core.states import can_transition, transition
from app.main import create_app
from app.storage.atomic import read_json
from app.storage.jsonl import read_jsonl

PID_RE = re.compile(r"pid=(\d+)")


def _wait_verification(client: TestClient, verification_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/verifications/{verification_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError("verification did not finish")


def _wait_run(client: TestClient, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/runs/{run_id}").json()
        if body["state"] in ("completed", "failed", "stopped"):
            return body
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def _start_run(
    client: TestClient,
    *,
    entries: dict[str, bytes],
    model_family: str,
    config: dict | None = None,
    checkpoint_catalog_id: str | None = None,
) -> str:
    completed = upload_dataset(client, f"{model_family}.zip", to_zip(entries))
    verification = client.post(
        "/api/v1/verifications",
        json={
            "dataset_hash": completed["dataset_hash"],
            "model_family": model_family,
            "preset": "dev",
        },
    ).json()
    _wait_verification(client, verification["id"])
    body = client.post(
        "/api/v1/runs",
        json={
            "verification_id": verification["id"],
            "name": "test-run",
            "config": config or {},
            "checkpoint_catalog_id": checkpoint_catalog_id,
        },
    )
    assert body.status_code == 202, body.text
    return body.json()["id"]


# --- state machine (unit) ---------------------------------------------------


def test_state_machine_valid_and_invalid_transitions() -> None:
    assert can_transition("queued", "preparing")
    assert can_transition("preparing", "running")
    assert can_transition("running", "paused")
    assert can_transition("paused", "running")
    assert can_transition("running", "stopped")
    assert can_transition("running", "completed")
    assert can_transition("running", "failed")
    assert not can_transition("completed", "running")
    assert not can_transition("failed", "stopped")
    with pytest.raises(ValueError):
        transition("running", "queued")


# --- real trainers (fast) ---------------------------------------------------


def test_random_forest_trains_end_to_end(client: TestClient, home: Path) -> None:
    run_id = _start_run(client, entries=build_tabular(rows=60), model_family="random_forest")
    body = _wait_run(client, run_id)
    assert body["state"] == "completed", body
    assert body["best"]["metric"] == "val/accuracy"
    assert body["best"]["value"] > 0
    assert body["env"]["torch"] is not None  # run info panel data [doc 05 §5]
    assert body["env"]["python"]
    assert body["env"]["hostname"]
    assert (home / "runs" / run_id / "artifacts" / "model.joblib").is_file()
    records = list(read_jsonl(home / "runs" / run_id / "events.jsonl"))
    metric_names = {r["data"]["name"] for r in records if r["type"] == "metric"}
    assert {"train/accuracy", "val/accuracy"} <= metric_names
    seqs = [r["seq"] for r in records]
    assert seqs == sorted(seqs)
    # No checkpoint requested → nothing was downloaded [doc 14 §3.4].
    assert body["checkpoint"]["source"] == "none"
    assert not (home / "checkpoints" / "index.json").is_file()
    # The run appears in the list endpoint [doc 05 §5].
    listed = client.get("/api/v1/runs").json()
    assert any(item["id"] == run_id for item in listed["items"])


def test_cnn_trains_end_to_end(client: TestClient, home: Path) -> None:
    run_id = _start_run(
        client,
        entries=build_classification(per_class=4, size=(32, 32)),
        model_family="cnn",
        config={"epochs": 1, "batch_size": 2, "imgsz": 32},
    )
    body = _wait_run(client, run_id)
    assert body["state"] == "completed", body
    assert body["best"]["metric"] == "val/accuracy"
    assert (home / "runs" / run_id / "checkpoints" / "last.ckpt").is_file()
    records = list(read_jsonl(home / "runs" / run_id / "events.jsonl"))
    metric_names = {r["data"]["name"] for r in records if r["type"] == "metric"}
    assert {"train/loss", "val/loss", "val/accuracy"} <= metric_names


def test_models_endpoint_lists_trainers(client: TestClient) -> None:
    body = client.get("/api/v1/models").json()
    families = {item["family"] for item in body["items"]}
    assert {"random_forest", "cnn"} <= families


# --- control: stop is not a failure (fast, needs a slow enough loop) --------


def test_stop_marks_stopped_not_failed(client: TestClient, home: Path) -> None:
    run_id = _start_run(
        client,
        entries=build_classification(per_class=6, size=(32, 32)),
        model_family="cnn",
        config={"epochs": 10_000, "batch_size": 2, "batch_delay": 0.05},
    )
    # Wait until it is actually running before asking it to stop.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/runs/{run_id}").json()["state"] == "running":
            break
        time.sleep(0.1)
    response = client.post(f"/api/v1/runs/{run_id}/stop")
    assert response.status_code == 200
    body = _wait_run(client, run_id, timeout=60)
    assert body["state"] == "stopped", body
    assert body["failure"]["reason"] is None


# --- checkpoint resolution at start -----------------------------------------


def test_checkpoint_resolved_at_start(client: TestClient, home: Path) -> None:
    payload = bytes(range(256)) * 100
    digest = hashlib.sha256(payload).hexdigest()
    source = home / "pretrained.bin"
    source.write_bytes(payload)
    # Override the shipped catalog with a verified, local entry [doc 14 §5].
    (home / "checkpoints" / "catalog.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "name": "tiny-backbone",
                    "version": "1.0",
                    "model_family": "cnn",
                    "task_types": ["image_classification"],
                    "pretrained_on": "unit-test",
                    "source_type": "file",
                    "source_uri": f"file://{source}",
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "license": "MIT",
                    "filename": "tiny-backbone.bin",
                }
            ]
        ),
        encoding="utf-8",
    )
    run_id = _start_run(
        client,
        entries=build_classification(per_class=3, size=(32, 32)),
        model_family="cnn",
        config={"epochs": 1, "batch_size": 2},
        checkpoint_catalog_id="tiny-backbone@1.0",
    )
    body = _wait_run(client, run_id)
    assert body["state"] == "completed"
    checkpoint = body["checkpoint"]
    assert checkpoint["catalog_id"] == "tiny-backbone@1.0"
    assert checkpoint["sha256"] == digest
    assert checkpoint["source"] == "download"
    assert Path(checkpoint["path"]).is_file()


# --- crash recovery (slow) --------------------------------------------------


@pytest.mark.slow
def test_kill_worker_marks_failed_worker_lost(tmp_path: Path) -> None:
    home = tmp_path / "home"
    settings = Settings(
        home=home,
        log_level="DEBUG",
        heartbeat_seconds=0.2,
        worker_lost_seconds=2.0,
    )
    with TestClient(create_app(settings)) as client:
        run_id = _start_run(
            client,
            entries=build_classification(per_class=8, size=(32, 32)),
            model_family="cnn",
            config={"epochs": 10_000, "batch_size": 2, "batch_delay": 0.2},
        )
        # Find the worker pid from the parent's event log.
        pid: int | None = None
        deadline = time.monotonic() + 30
        while pid is None and time.monotonic() < deadline:
            records = list(read_jsonl(home / "runs" / run_id / "events.jsonl"))
            for record in records:
                if record["type"] == "log" and "subprocess started" in record["data"]["message"]:
                    match = PID_RE.search(record["data"]["message"])
                    if match:
                        pid = int(match.group(1))
            if pid is None:
                time.sleep(0.1)
        assert pid is not None, "worker pid was never logged"

        os.killpg(pid, signal.SIGKILL)  # kill the whole process group, like a crash
        body = _wait_run(client, run_id, timeout=60)
        assert body["state"] == "failed", body
        assert body["failure"]["reason"] == "WORKER_LOST"
        # The API stays up after the worker died [doc 01 §10.6].
        assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.slow
def test_restart_marks_running_run_interrupted(tmp_path: Path) -> None:
    """A run left in `running` is marked `interrupted` on the next startup [doc 04 §3 rule 6]."""
    home = tmp_path / "home"
    (home / "runs" / "run_dead").mkdir(parents=True)
    from app.storage.atomic import atomic_write_json

    atomic_write_json(
        home / "runs" / "run_dead" / "run.json",
        {"id": "run_dead", "name": "dead", "state": "running", "schema_version": 1},
    )
    with TestClient(create_app(Settings(home=home, log_level="DEBUG"))) as client:
        assert client.get("/healthz").status_code == 200
    assert read_json(home / "runs" / "run_dead" / "run.json")["state"] == "interrupted"
