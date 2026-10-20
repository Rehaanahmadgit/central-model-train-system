"""Structured logging tests [doc 10 §1/§8]."""

import logging

from app.core.config import Settings
from app.core.logging import setup_logging


def test_setup_logging_disables_uvicorn_access() -> None:
    settings = Settings(log_level="DEBUG")
    setup_logging(settings)
    assert logging.getLogger("uvicorn.access").disabled is True


def test_setup_logging_hides_uvicorn_connection_lines() -> None:
    # uvicorn.error INFO lines contain client IPs [doc 10 §1]; level must be
    # raised so they never reach handlers.
    settings = Settings(log_level="DEBUG")
    setup_logging(settings)
    assert logging.getLogger("uvicorn.error").level == logging.WARNING


def test_root_logger_is_json_with_request_filter() -> None:
    settings = Settings(log_level="DEBUG")
    setup_logging(settings)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert any(hasattr(f, "__class__") for f in root.handlers[0].filters)
