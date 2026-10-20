"""Structured JSON logging [doc 10 §8].

Privacy rule: client IPs and user agents are never logged [doc 10 §1];
uvicorn's access logger is disabled by default for exactly that reason.
"""

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from .config import Settings

# Per-request id threaded through log records [doc 10 §8].
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(value: str) -> Token[str | None]:
    return request_id_var.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """Attach the per-request id to every record emitted while handling it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__["request_id"] = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. ``run_id`` becomes populated in the training slices."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "run_id": getattr(record, "run_id", None),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(settings: Settings) -> None:
    """Install JSON logging on the root and uvicorn loggers [doc 10 §8]."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)

    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # uvicorn emits connection lines (including client IPs) at INFO on
    # uvicorn.error; raise its level so they never reach the logs [doc 10 §1].
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    if settings.log_access:
        access_logger.propagate = True
    else:
        # Access lines contain client IPs; off unless explicitly enabled [doc 10 §1].
        access_logger.disabled = True
