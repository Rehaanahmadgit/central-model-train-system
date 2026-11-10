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
from fixtures.make_datasets import build_classification, build_tabular, materialize, to_zip
from helpers import make_pretrained_checkpoint, upload_dataset

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


@pytest.mark.slow
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


@pytest.mark.slow
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
    # Evaluation report is written and served [doc 05 §5].
    assert (home / "runs" / run_id / "artifacts" / "evaluation.json").is_file()
    evaluation = client.get(f"/api/v1/runs/{run_id}/evaluation")
    assert evaluation.status_code == 200
    assert evaluation.json()["task"] == "image_classification"
    assert evaluation.json()["confusion_matrix"]
    assert evaluation.json()["per_class"]


def test_models_endpoint_lists_trainers(client: TestClient) -> None:
    body = client.get("/api/v1/models").json()
    families = {item["family"] for item in body["items"]}
    assert {"random_forest", "cnn"} <= families


# --- control: stop is not a failure (fast, needs a slow enough loop) --------


@pytest.mark.slow
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


@pytest.mark.slow
def test_checkpoint_resolved_at_start(client: TestClient, home: Path) -> None:
    source = home / "pretrained.ckpt"
    make_pretrained_checkpoint(source, num_classes=2, fill=0.5)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
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
                    "size_bytes": source.stat().st_size,
                    "license": "MIT",
                    "filename": "tiny-backbone.ckpt",
                }
            ]
        ),
        encoding="utf-8",
    )
    from app.checkpoints.catalog import load_catalog
    from app.checkpoints.licences import LicenceStore

    for entry in load_catalog(home):  # the user accepted the licence in the UI
        LicenceStore(home).accept(entry)
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
    # The resolved checkpoint was actually consumed by the trainer.
    events = list(read_jsonl(home / "runs" / run_id / "events.jsonl"))
    assert any(
        "loaded pretrained checkpoint" in e["data"]["message"] for e in events if e["type"] == "log"
    )


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


@pytest.mark.slow
def test_pause_then_resume_then_stop(client: TestClient, home: Path) -> None:
    """Pause/resume flows through the state machine to the real subprocess [doc 08 §3]."""
    run_id = _start_run(
        client,
        entries=build_classification(per_class=8, size=(32, 32)),
        model_family="cnn",
        config={"epochs": 10_000, "batch_size": 2, "batch_delay": 0.05},
    )
    # Wait until running, then pause and confirm the state flips to paused.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/runs/{run_id}").json()["state"] == "running":
            break
        time.sleep(0.1)
    assert client.post(f"/api/v1/runs/{run_id}/pause").status_code == 200
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/runs/{run_id}").json()["state"] == "paused":
            break
        time.sleep(0.1)
    assert client.get(f"/api/v1/runs/{run_id}").json()["state"] == "paused"

    # Resume → back to running.
    assert client.post(f"/api/v1/runs/{run_id}/resume").status_code == 200
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/runs/{run_id}").json()["state"] == "running":
            break
        time.sleep(0.1)
    assert client.get(f"/api/v1/runs/{run_id}").json()["state"] == "running"

    assert client.post(f"/api/v1/runs/{run_id}/stop").status_code == 200
    body = _wait_run(client, run_id, timeout=60)
    assert body["state"] == "stopped"
    assert body["failure"]["reason"] is None


@pytest.mark.slow
def test_exact_resume_from_last_checkpoint(client: TestClient, home: Path) -> None:
    """A retry with resume_from_last continues from the parent's last.ckpt [doc 08 §4]."""
    first_id = _start_run(
        client,
        entries=build_classification(per_class=4, size=(32, 32)),
        model_family="cnn",
        config={"epochs": 1, "batch_size": 2},
    )
    first = _wait_run(client, first_id)
    assert first["state"] == "completed"
    assert (home / "runs" / first_id / "checkpoints" / "last.ckpt").is_file()

    response = client.post(
        f"/api/v1/runs/{first_id}/retry",
        json={"resume_from_last": True, "config": {"epochs": 2}},
    )
    assert response.status_code == 202, response.text
    second_id = response.json()["id"]

    second = _wait_run(client, second_id)
    assert second["state"] == "completed", second
    assert second["parent_run_id"] == first_id
    # The worker logged the exact resume and continued past the parent's epoch.
    records = list(read_jsonl(home / "runs" / second_id / "events.jsonl"))
    assert any(
        "resumed from last.ckpt" in r["data"]["message"] for r in records if r["type"] == "log"
    )
    assert any(r["type"] == "metric" and r["data"].get("epoch") == 2 for r in records)


@pytest.mark.slow
def test_rf_resume_unavailable_without_checkpoint(client: TestClient, home: Path) -> None:
    """RandomForest saves no checkpoints; resume must be rejected, not silently skipped."""
    first_id = _start_run(client, entries=build_tabular(rows=40), model_family="random_forest")
    first = _wait_run(client, first_id)
    assert first["state"] == "completed"
    response = client.post(f"/api/v1/runs/{first_id}/retry", json={"resume_from_last": True})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "RESUME_UNAVAILABLE"


@pytest.mark.slow
def test_cnn_best_epoch_is_real(client: TestClient, home: Path) -> None:
    """A fresh run reports the actual best epoch, never 0 [bug fix (a)]."""
    run_id = _start_run(
        client,
        entries=build_classification(per_class=4, size=(32, 32)),
        model_family="cnn",
        config={"epochs": 2, "batch_size": 2},
    )
    body = _wait_run(client, run_id)
    assert body["state"] == "completed", body
    assert body["best"]["epoch"] >= 1
    assert body["best"]["metric"] == "val/accuracy"


@pytest.mark.slow
def test_resume_without_epoch_increase_is_clean(client: TestClient, home: Path) -> None:
    """Retry resume_from_last with equal epochs must not crash [bug fix (b)]."""
    first_id = _start_run(
        client,
        entries=build_classification(per_class=3, size=(32, 32)),
        model_family="cnn",
        config={"epochs": 2, "batch_size": 2},
    )
    first = _wait_run(client, first_id)
    assert first["state"] == "completed"

    response = client.post(f"/api/v1/runs/{first_id}/retry", json={"resume_from_last": True})
    assert response.status_code == 202, response.text
    second = _wait_run(client, response.json()["id"])
    # start_epoch (3) > epochs (2): the loop is empty and the run still
    # completes cleanly, reporting the parent's best epoch — no UnboundLocalError.
    assert second["state"] == "completed", second
    assert second["best"]["epoch"] >= 1


def test_cnn_patience_zero_disables_early_stopping(tmp_path: Path) -> None:
    """patience=0 must disable early stopping, not silently become 3 [bug fix (c)]."""
    from types import SimpleNamespace

    from app.core.device import resolve_device
    from app.training.plugins.cnn import CNNTrainer
    from app.verification.manifest import build_manifest, write_manifest

    raw = materialize(tmp_path / "raw", build_classification(per_class=2, size=(32, 32)))
    summary = build_manifest(raw, "image_folder", sample_cap=100)
    write_manifest(tmp_path / "manifest.parquet", summary.rows)

    def make(config: dict) -> CNNTrainer:
        ctx = SimpleNamespace(
            dataset_root=raw,
            manifest_path=tmp_path / "manifest.parquet",
            config=config,
            seed=42,
            device=resolve_device("cpu"),
            output_dir=tmp_path / "artifacts",
            emit_log=lambda *args, **kwargs: None,
        )
        trainer = CNNTrainer()
        trainer.prepare(ctx)
        return trainer

    assert make({"epochs": 1, "batch_size": 2, "imgsz": 32, "patience": 0})._patience == 0
    assert make({"epochs": 1, "batch_size": 2, "imgsz": 32})._patience == 3


# --- config types (D124) ----------------------------------------------------


def test_normalize_config_coerces_form_strings() -> None:
    from app.services.runs import normalize_config

    schema = {
        "type": "object",
        "properties": {
            "epochs": {"type": "integer", "default": 1},
            "batch_size": {"type": "integer", "default": 1},
            "lr": {"type": "number", "default": 0.0001},
            "amp": {"type": "boolean", "default": False},
        },
    }

    config = normalize_config(
        {"epochs": "3", "batch_size": "4", "lr": "0.002", "amp": "true"}, schema
    )

    assert config == {"epochs": 3, "batch_size": 4, "lr": 0.002, "amp": True}
    assert isinstance(config["batch_size"], int)
    assert isinstance(config["lr"], float)


def test_normalize_config_rejects_unusable_values() -> None:
    from app.core.errors import ApiError
    from app.services.runs import normalize_config

    schema = {
        "properties": {
            "batch_size": {"type": "integer"},
            "lr": {"type": "number"},
            "amp": {"type": "boolean"},
        }
    }

    for bad in (
        {"batch_size": "four"},
        {"batch_size": 2.5},
        {"lr": "fast"},
        {"amp": "maybe"},
        {"batch_size": True},
    ):
        with pytest.raises(ApiError) as excinfo:
            normalize_config(bad, schema)
        assert excinfo.value.status_code == 400
        assert excinfo.value.code == "INVALID_CONFIG"


def test_normalize_config_keeps_unknown_keys() -> None:
    from app.services.runs import normalize_config

    config = normalize_config({"seed": "7", "custom": "x"}, {"properties": {}})

    assert config == {"seed": "7", "custom": "x"}


@pytest.mark.slow
def test_run_config_is_saved_with_numeric_types(client: TestClient, home: Path) -> None:
    """The wizard sends strings from HTML inputs; run.json must hold numbers."""
    run_id = _start_run(
        client,
        entries=build_tabular(rows=40),
        model_family="random_forest",
        config={"n_estimators": "7"},
    )
    _wait_run(client, run_id)

    record = read_json(home / "runs" / run_id / "run.json")
    assert record["config"]["n_estimators"] == 7
    assert isinstance(record["config"]["n_estimators"], int)


@pytest.mark.slow
def test_run_rejects_non_numeric_config(client: TestClient) -> None:
    completed = upload_dataset(client, "tabular.zip", to_zip(build_tabular(rows=20)))
    verification = client.post(
        "/api/v1/verifications",
        json={
            "dataset_hash": completed["dataset_hash"],
            "model_family": "random_forest",
            "preset": "dev",
        },
    ).json()
    _wait_verification(client, verification["id"])

    response = client.post(
        "/api/v1/runs",
        json={
            "verification_id": verification["id"],
            "name": "bad-config",
            "config": {"n_estimators": "many"},
        },
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "INVALID_CONFIG"


# --- Slice F: Run again keeps numeric types ------------------------------------


def test_run_again_coerces_string_config_from_an_old_run(client: TestClient, home: Path) -> None:
    """A parent run saved by the old wizard (string numbers) must retry with real numbers."""
    from app.storage.atomic import atomic_write_json

    run_id = _start_run(
        client,
        entries=build_classification(per_class=3, classes=("cat", "dog"), size=(32, 32)),
        model_family="cnn",
        config={"epochs": 1},
    )
    client.post(f"/api/v1/runs/{run_id}/stop")
    deadline = time.monotonic() + 60
    while client.get(f"/api/v1/runs/{run_id}").json()["state"] not in (
        "completed",
        "failed",
        "stopped",
    ):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    path = home / "runs" / run_id / "run.json"
    record = read_json(path)
    record["config"]["epochs"] = "2"
    record["config"]["lr"] = "0.01"
    atomic_write_json(path, record)

    retried = client.post(f"/api/v1/runs/{run_id}/retry", json={})
    assert retried.status_code == 202, retried.text
    new_config = read_json(home / "runs" / retried.json()["id"] / "run.json")["config"]
    assert new_config["epochs"] == 2 and isinstance(new_config["epochs"], int)
    assert new_config["lr"] == 0.01 and isinstance(new_config["lr"], float)
    client.post(f"/api/v1/runs/{retried.json()['id']}/stop")
