"""Liveness and readiness probes [doc 05 §7, doc 10 §7].

``/healthz``: the process is alive.
``/readyz``: workspace writable AND free disk above threshold.
"""

import shutil
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.core.paths import ensure_workspace

router = APIRouter(tags=["system"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Process liveness [doc 10 §7]."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Readiness: workspace writable + disk space [doc 10 §7]."""
    settings = cast(Settings, request.app.state.settings)
    checks: dict[str, Any] = {}
    ok = True

    try:
        ensure_workspace(settings.home)
        probe = settings.home / ".write_probe"
        try:
            probe.write_text("ok")
        finally:
            probe.unlink(missing_ok=True)
        checks["workspace_writable"] = "ok"
    except OSError as exc:
        checks["workspace_writable"] = f"not writable: {exc}"
        ok = False

    try:
        usage = shutil.disk_usage(settings.home)
    except OSError as exc:
        checks["disk_free"] = f"unavailable: {exc}"
        ok = False
    else:
        if usage.free < settings.min_free_disk_bytes:
            checks["disk_free"] = "below threshold"
            ok = False
        else:
            checks["disk_free"] = "ok"

    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "not_ready", "checks": checks},
    )
