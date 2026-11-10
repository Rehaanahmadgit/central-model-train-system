"""System health API [doc 05 §7].

``gpu`` comes from the same device resolver the trainers use, so the Configure
step can say honestly whether training will run on a GPU or on the CPU.
"""

import shutil
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Request

from app.core.config import Settings
from app.core.device import probe_device
from app.storage.atomic import read_json

router = APIRouter(prefix="/system", tags=["system"])

RUNNING_STATES = frozenset({"preparing", "running", "paused", "evaluating"})


def _scan_jobs(home: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    running: list[dict[str, Any]] = []
    queued: list[dict[str, Any]] = []
    runs_dir = home / "runs"
    if runs_dir.is_dir():
        for run_json in sorted(runs_dir.glob("*/run.json")):
            try:
                data = read_json(run_json)
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            state = data.get("state")
            entry = {
                "id": data.get("id") or run_json.parent.name,
                "name": data.get("name"),
                "state": state,
            }
            if state in RUNNING_STATES:
                running.append(entry)
            elif state == "queued":
                queued.append(entry)
    return running, queued


def _gpu_info(settings: Settings) -> dict[str, Any]:
    # Subprocess probe: this process must not import torch (see probe_device).
    device = probe_device(settings.device)
    if device.get("kind") == "cuda":
        return {
            "available": True,
            "name": device.get("name"),
            "memory_gb": device.get("memory_gb"),
            "message": f"GPU: {device.get('name')}",
        }
    return {"available": False, "message": "No GPU detected", "reason": device.get("reason")}


@router.get("")
def system(request: Request) -> dict[str, Any]:
    settings = cast(Settings, request.app.state.settings)
    running, queued = _scan_jobs(settings.home)
    usage = shutil.disk_usage(settings.home)
    recovery_warnings = list(getattr(request.app.state, "recovery_warnings", []))
    return {
        "running_jobs": running,
        "queued_jobs": queued,
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        "gpu": _gpu_info(settings),
        "recovery_warnings": recovery_warnings,
        "security": dict(getattr(request.app.state, "security", {})),
    }
