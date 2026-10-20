"""Demo fake run: hello-world job streaming fake metrics [doc 12 Phase 0 M0].

Deterministic (sine-based) so tests and demos are reproducible [doc 08 §5].
The real Job Manager + subprocess supervision arrives in slice 8.
"""

import asyncio
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app.realtime.broadcaster import EventBroadcaster
from app.realtime.events import RunEventLog
from app.storage.atomic import atomic_write_json, read_json
from app.storage.ids import new_run_id
from app.storage.locks import FileLock

# Synthetic placeholder — the demo run has no real dataset (doc 12 Phase 0).
DEMO_DATASET_HASH = "0" * 64


def create_fake_run(home: Path) -> dict[str, Any]:
    run_id = new_run_id()
    now = datetime.now().isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "id": run_id,
        "name": "hello-world",
        "dataset_hash": DEMO_DATASET_HASH,
        "model_family": "demo",
        "task_type": "demo",
        "state": "running",
        "queued_at": now,
        "started_at": now,
        "schema_version": 1,
        "seed": 42,
    }
    directory = home / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(directory / "run.json", record)
    return record


def _update_state(home: Path, run_id: str, **fields: Any) -> None:
    path = home / "runs" / run_id / "run.json"
    data = read_json(path)
    if not isinstance(data, dict):
        return
    data.update(fields)
    with FileLock(home / "locks" / f"run-{run_id}.lock"):
        atomic_write_json(path, data)


async def run_fake_training(
    home: Path,
    broadcaster: EventBroadcaster,
    *,
    steps: int,
    interval: float,
) -> str:
    """Run a tiny fake training loop; returns the run id."""
    record = create_fake_run(home)
    run_id = str(record["id"])
    log = RunEventLog(home, run_id)
    try:
        log.append("state", {"from": "queued", "to": "running"})
        log.append("log", {"level": "INFO", "message": "hello-world fake run started"})
        for step in range(steps):
            epoch = step // 20
            train_loss = 2.5 * math.exp(-3 * step / max(steps, 1)) + 0.05
            val_loss = train_loss + 0.08 + 0.02 * math.sin(step / 2)
            lr = 0.001 * (0.95 ** step)
            for name, value in (
                ("train/loss", train_loss),
                ("val/loss", val_loss),
                ("lr", lr),
            ):
                event = log.append(
                    "metric",
                    {"name": name, "value": round(value, 4), "step": step, "epoch": epoch},
                )
                broadcaster.publish(run_id, event)
            if step > 0 and step % 20 == 0:
                event = log.append(
                    "log", {"level": "INFO", "message": f"epoch {epoch}/5 finished"}
                )
                broadcaster.publish(run_id, event)
            if step == max(1, steps // 2):
                event = log.append(
                    "event",
                    {
                        "name": "checkpoint_saved",
                        "epoch": epoch,
                        "path": "runs/{id}/checkpoints/last.ckpt",
                    },
                )
                broadcaster.publish(run_id, event)
            if step % 4 == 0:
                event = log.append(
                    "system",
                    {
                        "gpu_util": None,
                        "gpu_mem_mb": None,
                        "cpu": 30 + 20 * math.sin(step / 5),
                        "ram_mb": 8000 + 200 * math.sin(step / 3),
                    },
                )
                broadcaster.publish(run_id, event)
            await asyncio.sleep(interval)
        log.append("state", {"from": "running", "to": "completed"})
        log.append("log", {"level": "INFO", "message": "hello-world fake run completed"})
        log.sync()
    finally:
        log.close()
    _update_state(
        home,
        run_id,
        state="completed",
        finished_at=datetime.now().isoformat(timespec="seconds"),
    )
    return run_id
