"""Optional access token enforcement [doc 10 §2, FR-20, DIST-07].

The token identifies nobody: it is a single shared secret from
``TRAINHUB_ACCESS_TOKEN``, compared in constant time. When it is set, every
``/api/`` request must send ``X-Access-Token`` and every WebSocket must send
``?token=``. The SPA, ``/healthz``, ``/readyz`` and the docs stay reachable so
the browser can load the page and prompt for the token.
"""

from __future__ import annotations

import secrets
from typing import Any

from app.core.config import Settings

# Hosts considered "local": no token means no loud warning there [DIST-07].
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def token_is_set(settings: Settings) -> bool:
    return settings.access_token is not None


def token_required(settings: Settings) -> bool:
    """Whether the app will reject requests without a token."""
    return token_is_set(settings)


def verify_token(settings: Settings, provided: str | None) -> bool:
    """Constant-time compare [doc 10 §2]."""
    expected = settings.access_token
    if expected is None:
        return True
    if not provided:
        return False
    return secrets.compare_digest(expected.get_secret_value(), provided)


def security_state(settings: Settings) -> dict[str, Any]:
    """What the /system endpoint and the UI banner need [DIST-07]."""
    non_local = settings.bind_host not in LOCAL_HOSTS
    if token_is_set(settings):
        return {"token_set": True, "token_required": True, "warning": None}
    if non_local:
        return {
            "token_set": False,
            "token_required": False,
            "warning": (
                "This instance is reachable on a network but no access token is set. "
                "Anyone who can reach it can train and delete data. Set TRAINHUB_ACCESS_TOKEN "
                "and restart [doc 16 DIST-07]."
            ),
        }
    return {"token_set": False, "token_required": False, "warning": None}
