"""Person-free audit trail [doc 04 §2, doc 10 §6].

One line per action in ``audit/audit-YYYY-MM.jsonl``, append-only, fsynced at
write time (state changes). Forbidden personal-data fields are never written
[doc 04 §2, doc 13 §1.2].
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from .jsonl import JsonlAppender

FORBIDDEN_AUDIT_FIELDS = frozenset(
    {
        "user_name",
        "email",
        "ip",
        "ip_address",
        "user_agent",
        "device_id",
        "client_id",
        "user_id",
        "org_id",
        "started_by",
    }
)


def append_audit(
    home: Path,
    *,
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Path:
    """Append one audit line and return the audit file path [doc 04 §2]."""
    ts = now or datetime.now()
    resolved_details = details or {}
    forbidden = sorted(set(resolved_details.keys()) & FORBIDDEN_AUDIT_FIELDS)
    if forbidden:
        raise ValueError(f"audit details contain forbidden fields: {forbidden}")
    record = {
        "ts": ts.isoformat(timespec="seconds"),
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "details": resolved_details,
    }
    path = home / "audit" / f"audit-{ts:%Y-%m}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlAppender(path) as appender:
        appender.append(record)
        appender.sync()  # fsync at state changes [doc 04 §3 rule 2]
    return path
