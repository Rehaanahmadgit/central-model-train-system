"""Typed API errors mapped to the documented error envelope [doc 05 §0].

Codes are not enumerated in doc 05 (only examples), so the names follow the
doc's style (VERIFICATION_FAILED, VERIFICATION_STALE) — ASSUMPTION (D23/D36).
"""

from typing import Any


class ApiError(Exception):
    """Raised by services; converted to ``{"error": {code, message, details}}`` in main."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
