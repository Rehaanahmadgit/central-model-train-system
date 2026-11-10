"""Access token enforcement, ON by default [doc 10 §2, FR-20, DIST-07, Slice E].

The token identifies nobody: it is a single shared secret from
``TRAINHUB_ACCESS_TOKEN``, compared in constant time. When it is set, every
``/api/`` request must send ``X-Access-Token`` and every WebSocket must send
``?token=``. The SPA, ``/healthz``, ``/readyz`` and the docs stay reachable so
the browser can load the page and prompt for the token.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from app.core.config import Settings

# Hosts considered "local": no token means no loud warning there [DIST-07].
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


TOKEN_FILE_NAME = "access-token"


def token_file(home: Path) -> Path:
    return home / TOKEN_FILE_NAME


def token_is_set(settings: Settings) -> bool:
    return settings.access_token is not None


def token_required(settings: Settings) -> bool:
    """Auth is ON unless explicitly disabled [Slice E: default ON]."""
    return not settings.auth_disabled


def verify_token(settings: Settings, provided: str | None) -> bool:
    """Constant-time compare [doc 10 §2]. Fails closed: no token loaded = deny."""
    expected = settings.access_token
    if expected is None:
        # No token loaded: only the explicit opt-out lets requests through.
        return settings.auth_disabled
    if not provided:
        return False
    return secrets.compare_digest(expected.get_secret_value(), provided)


def _write_new_token(path: Path) -> str:
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    os.chmod(path, 0o600)  # umask-proof, and fixes a file created with looser bits
    return token


def ensure_access_token(settings: Settings) -> bool:
    """Make sure ``settings.access_token`` is loaded; True when it was just generated.

    Order: explicit TRAINHUB_ACCESS_TOKEN > existing ``<home>/access-token`` > new
    random token written there with mode 0600.
    """
    if settings.auth_disabled or settings.access_token is not None:
        return False
    path = token_file(settings.home)
    if path.is_file():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                os.chmod(path, 0o600)
            settings.access_token = SecretStr(existing)
            return False
    settings.access_token = SecretStr(_write_new_token(path))
    return True


def reset_token(home: Path) -> str:
    """New token for ``trainhub --reset-token``; the old one stops working on restart."""
    return _write_new_token(token_file(home))


def security_state(settings: Settings) -> dict[str, Any]:
    """What the /system endpoint and the UI banner need [DIST-07]."""
    if settings.auth_disabled and settings.access_token is None:
        return {
            "token_set": False,
            "token_required": False,
            "warning": (
                "Authentication is DISABLED (TRAINHUB_AUTH_DISABLED). Anyone who can reach "
                "this instance can train, download and delete data."
            ),
        }
    return {"token_set": True, "token_required": True, "warning": None}
