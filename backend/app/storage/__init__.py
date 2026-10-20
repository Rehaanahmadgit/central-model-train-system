"""Filesystem-only storage layer [doc 04].

No database. Every write goes through here (doc 13 §3):
- ``atomic``  — atomic JSON/text/bytes writes (tmp + fsync + os.replace)
- ``jsonl``   — append-only JSONL with flush timer, fsync, tail repair
- ``locks``   — cross-process fcntl.flock locks
- ``ids``     — ID generation per doc 04 §4
- ``index``   — rebuildable in-memory index with corrupt-file quarantine
- ``recovery``— startup crash recovery (doc 04 §3 rule 6)
- ``audit``   — person-free audit trail (doc 04 §2)
"""

from .atomic import atomic_write_bytes, atomic_write_json, atomic_write_text, read_json
from .audit import append_audit
from .ids import new_project_id, new_run_id, new_upload_id, new_verification_id
from .index import FileIndex, quarantine
from .jsonl import JsonlAppender, JsonlParseError, read_jsonl, repair_tail
from .locks import FileLock, LockTimeout
from .recovery import RecoveryReport, recover_workspace

__all__ = [
    "FileIndex",
    "FileLock",
    "JsonlAppender",
    "JsonlParseError",
    "LockTimeout",
    "RecoveryReport",
    "append_audit",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "new_project_id",
    "new_run_id",
    "new_upload_id",
    "new_verification_id",
    "quarantine",
    "read_json",
    "read_jsonl",
    "repair_tail",
    "recover_workspace",
]
