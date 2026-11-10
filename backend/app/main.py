"""TrainHub API entry point [doc 03 §2.1].

- Async FastAPI app; no heavy ML in the event loop (doc 13 §3).
- Error envelope ``{"error": {"code", "message", "details"}}`` everywhere [doc 05 §0].
- Optional global access token is added in a later slice [doc 10 §2, FR-20].
"""

import logging
import multiprocessing
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.health import router as health_router
from app.api.router import api_router
from app.checkpoints.catalog import ensure_catalog
from app.core.auth import ensure_access_token, security_state, token_file, verify_token
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.logging import reset_request_id, set_request_id, setup_logging
from app.core.paths import ensure_workspace
from app.realtime.broadcaster import EventBroadcaster
from app.realtime.verification_ws import router as verification_ws_router
from app.realtime.ws import router as ws_router
from app.storage.recovery import recover_workspace
from app.training.job_manager import JobManager

REQUEST_ID_HEADER = "X-Request-Id"


def _queued_run_ids(home: Path) -> list[str]:
    """Run ids left in ``queued`` state, oldest first [doc 04 §3 rule 6]."""
    from app.storage.atomic import read_json

    runs_dir = home / "runs"
    if not runs_dir.is_dir():
        return []
    queued: list[tuple[str, str]] = []
    for path in runs_dir.glob("*/run.json"):
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("state") == "queued":
            queued.append((str(data.get("queued_at") or ""), path.parent.name))
    return [run_id for _, run_id in sorted(queued)]


logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory; tests inject their own Settings [doc 11 §2]."""
    resolved = settings if settings is not None else get_settings()
    setup_logging(resolved)
    broadcaster = EventBroadcaster()
    recovery_warnings: list[str] = []

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Verification runs in a process pool, never the event loop [doc 13 §3].
        # fork context (D46): spawn/forkserver re-import the test entry point
        # under pytest and deadlock. The fork-in-threads deprecation warning is
        # accepted: workers only do file I/O + PIL.
        # Created here (not at import) so it is always shut down cleanly.
        application.state.executor = ProcessPoolExecutor(
            max_workers=resolved.verify_workers,
            mp_context=multiprocessing.get_context("fork"),
        )
        # Best-effort layout creation; /readyz reports the truth if it fails [doc 04 §1].
        workspace_ok = False
        try:
            ensure_workspace(resolved.home)
        except OSError:
            logger.exception("workspace unavailable at startup; /readyz will report not ready")
        else:
            workspace_ok = True
            # Ship the default checkpoint catalog into the volume once [doc 14 §5].
            try:
                ensure_catalog(resolved.home)
            except OSError:
                logger.exception("could not install the default checkpoint catalog")
            # uvicorn's own startup line is suppressed (it shares a logger with
            # connection lines that contain client IPs), so log one ours instead.
            logger.info(
                "trainhub started: home=%s device=%s frontend=%s",
                resolved.home,
                resolved.device,
                "built" if resolved.frontend_dist else "dev",
            )
            # Auth is ON by default: load/generate the token BEFORE serving [Slice E].
            try:
                generated = ensure_access_token(resolved)
            except OSError:
                logger.exception("could not read/write the access token file; API stays locked")
                generated = False
            if generated and resolved.access_token is not None:
                logger.warning(
                    "FIRST BOOT: generated the access token (shown once): %s  -- it is also "
                    "stored in %s (mode 0600). Lost it? run: trainhub --reset-token",
                    resolved.access_token.get_secret_value(),
                    token_file(resolved.home),
                )
            security = security_state(resolved)
            application.state.security = security
            if security.get("warning"):
                logger.warning("SECURITY: %s", security["warning"])
            # Crash recovery on startup [doc 03 §2.2, doc 04 §3 rule 6].
            try:
                report = recover_workspace(resolved.home)
            except OSError:
                logger.exception("crash recovery failed at startup")
            else:
                if report.changed_anything:
                    logger.warning(
                        "crash recovery: interrupted=%d requeued=%d repaired_jsonl=%d "
                        "removed_stale_tmp=%d quarantined=%d",
                        len(report.interrupted_run_ids),
                        len(report.requeued_run_ids),
                        len(report.repaired_jsonl),
                        len(report.removed_stale_tmp),
                        len(report.quarantined),
                    )
                recovery_warnings[:] = [
                    f"quarantined corrupt file: {path}" for path in report.quarantined
                ] + [f"repaired JSONL tail: {path}" for path in report.repaired_jsonl]
        # In-process Job Manager: supervised training subprocesses [doc 08 §3].
        job_manager = JobManager(resolved.home, resolved, broadcaster)
        application.state.job_manager = job_manager
        if workspace_ok:
            # Re-queue runs left in `queued` before a restart [doc 04 §3 rule 6].
            for run_id in _queued_run_ids(resolved.home):
                try:
                    await job_manager.start(run_id)
                except Exception:
                    logger.warning("could not requeue run %s at startup", run_id, exc_info=True)
        try:
            yield
        finally:
            await job_manager.shutdown()
            executor = getattr(application.state, "executor", None)
            if executor is not None:
                # wait=True joins the worker processes; wait=False was leaving
                # fork-context workers unjoined, which could hang the whole
                # interpreter at shutdown (S9 root-cause fix).
                executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(
        title="TrainHub",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
    )
    application.state.settings = resolved
    application.state.broadcaster = broadcaster
    application.state.recovery_warnings = recovery_warnings

    @application.middleware("http")
    async def attach_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Reuses an incoming X-Request-Id if present (not personal data) [doc 10 §8].
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming if incoming else uuid.uuid4().hex
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            reset_request_id(token)

    @application.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Security headers [doc 10 §5]. CSP is added when the SPA is served in-container.
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        return response

    @application.middleware("http")
    async def access_token(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Optional shared access token [doc 10 §2, DIST-07]. Only /api paths are
        # gated; the SPA, /healthz and /readyz stay open so the browser can load
        # and prompt for the token. OPTIONS (CORS preflight) is never gated.
        path = request.url.path
        if request.method == "OPTIONS" or not path.startswith("/api/"):
            return await call_next(request)
        provided = request.headers.get("X-Access-Token")
        if verify_token(resolved, provided):
            return await call_next(request)
        if provided:
            code, message = (
                "ACCESS_TOKEN_INVALID",
                "That access token was rejected (401). Use the token printed in the docker "
                "logs on first boot, or read it from /data/access-token.",
            )
        else:
            code, message = (
                "ACCESS_TOKEN_REQUIRED",
                "This instance requires the access token (found in /data/access-token).",
            )
        return JSONResponse(
            status_code=401, content={"error": {"code": code, "message": message, "details": {}}}
        )

    if resolved.cors_origins:
        # CORS allowlist [doc 10 §5].
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @application.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        # Documented error envelope [doc 05 §0].
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "HTTP_ERROR", "message": detail, "details": {}}},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Error format [doc 05 §0]. Code VALIDATION_ERROR is an ASSUMPTION (not in docs).
        # Keep only serializable, privacy-safe fields: no input/ctx echo.
        errors = [
            {"type": entry.get("type"), "loc": entry.get("loc"), "msg": entry.get("msg")}
            for entry in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {"errors": errors},
                }
            },
        )

    @application.exception_handler(Exception)
    async def handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {"code": "INTERNAL", "message": "Internal server error", "details": {}}
            },
        )

    application.include_router(health_router)
    application.include_router(api_router)
    application.include_router(ws_router)  # /ws/runs/{id} [doc 05 §8]
    application.include_router(verification_ws_router)  # /ws/verifications/{id} [doc 05 §3]

    # Serve the built SPA as a fallback for non-API GET paths [doc 14 §4].
    # A plain Mount at "/" shadows the routers, so use a last-added catch-all
    # route with path confinement inside the dist dir [doc 10 §5].
    frontend_dist = resolved.frontend_dist
    if frontend_dist is not None and frontend_dist.is_dir():
        dist_root = frontend_dist.resolve()

        @application.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str) -> Response:
            if full_path.startswith(("api/", "ws")):
                # Unknown API/WS paths keep the documented error envelope [doc 05 §0].
                raise StarletteHTTPException(status_code=404, detail="Not Found")
            requested = (dist_root / full_path).resolve()
            if requested.is_file() and requested.is_relative_to(dist_root):
                return FileResponse(requested)
            index = dist_root / "index.html"
            if index.is_file():
                # SPA client-side routing: unknown non-API paths get the shell.
                return FileResponse(index)
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Frontend bundle not found",
                        "details": {},
                    }
                },
            )

    return application


app = create_app()
