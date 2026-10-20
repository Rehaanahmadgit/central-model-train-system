"""System health API [doc 05 §7].

GPU values show "No GPU detected" when pynvml is unavailable [doc 14 §3.2];
real GPU metrics arrive with the training slices.
"""

import shutil
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Request

from app.core.config import Settings
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
        # No GPU on the laptop; pynvml-based values land with slice 8 [doc 14 §3.2].
        "gpu": {"available": False, "message": "No GPU detected"},
        "recovery_warnings": recovery_warnings,
    }
