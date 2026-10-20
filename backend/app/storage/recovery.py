"""Startup crash recovery [doc 04 §3 rule 6, doc 03 §2.2].

- runs in `preparing|running|evaluating` → state `interrupted`
  (the resume-from-`last.ckpt` offer arrives with the training slices)
- `queued` runs are collected in `queued_at` order for the Job Manager (slice 8)
  to re-queue
- half-written JSONL tails are repaired
- `.tmp`/`.part` leftovers older than 1 hour are deleted
- corrupt `run.json` files are quarantined to `*.corrupt` [doc 03 §8]
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from app.core.states import INTERRUPTED, INTERRUPTIBLE_RUN_STATES, QUEUED

from .atomic import atomic_write_json, read_json
from .index import quarantine
from .jsonl import repair_tail
from .locks import FileLock

logger = logging.getLogger(__name__)

STALE_TMP_AGE = timedelta(hours=1)
RUN_SCHEMA_VERSION = 1  # run.json schema_version [doc 04 §2]
JSONL_TAIL_PATTERNS = ("runs/*/events.jsonl", "verifications/*/checks.jsonl", "audit/*.jsonl")
STALE_TMP_PATTERNS = ("tmp/**/*.part", "checkpoints/.tmp/**/*.part")


@dataclass
class RecoveryReport:
    interrupted_run_ids: list[str] = field(default_factory=list)
    requeued_run_ids: list[str] = field(default_factory=list)
    repaired_jsonl: list[Path] = field(default_factory=list)
    removed_stale_tmp: list[Path] = field(default_factory=list)
    quarantined: list[Path] = field(default_factory=list)

    @property
    def changed_anything(self) -> bool:
        return bool(
            self.interrupted_run_ids
            or self.requeued_run_ids
            or self.repaired_jsonl
            or self.removed_stale_tmp
            or self.quarantined
        )


def recover_workspace(
    home: Path,
    *,
    now: datetime | None = None,
    stale_after: timedelta = STALE_TMP_AGE,
) -> RecoveryReport:
    current = now or datetime.now()
    report = RecoveryReport()
    _recover_runs(home, report)
    _repair_jsonl_tails(home, report)
    _clean_stale_tmp(home, report, current, stale_after)
    return report


def _recover_runs(home: Path, report: RecoveryReport) -> None:
    queued: list[tuple[str, str]] = []  # (queued_at, run_id); ISO strings sort correctly
    runs_dir = home / "runs"
    if not runs_dir.is_dir():
        return
    for run_json in sorted(runs_dir.glob("*/run.json")):
        run_id = run_json.parent.name
        try:
            data = read_json(run_json)
        except (OSError, ValueError):
            report.quarantined.append(quarantine(run_json))
            continue
        if not isinstance(data, dict) or data.get("schema_version") != RUN_SCHEMA_VERSION:
            # Only version 1 is known; migrations are version 1 only [doc 04 §3 rule 7].
            report.quarantined.append(quarantine(run_json))
            continue
        state = data.get("state")
        if state in INTERRUPTIBLE_RUN_STATES:
            data["state"] = INTERRUPTED
            lock_path = home / "locks" / f"run-{run_id}.lock"
            with FileLock(lock_path):
                atomic_write_json(run_json, data)
            report.interrupted_run_ids.append(run_id)
            logger.warning("crash recovery marked run %s as interrupted", run_id)
        elif state == QUEUED:
            queued.append((str(data.get("queued_at") or ""), run_id))
    queued.sort(key=lambda pair: pair[0])
    report.requeued_run_ids = [run_id for _, run_id in queued]


def _repair_jsonl_tails(home: Path, report: RecoveryReport) -> None:
    for pattern in JSONL_TAIL_PATTERNS:
        for path in home.glob(pattern):
            try:
                removed = repair_tail(path)
            except OSError:
                logger.exception("failed to repair JSONL tail: %s", path)
                continue
            if removed:
                report.repaired_jsonl.append(path)
                logger.warning("repaired half-written JSONL tail: %s (%d bytes)", path, removed)


def _clean_stale_tmp(
    home: Path, report: RecoveryReport, now: datetime, stale_after: timedelta
) -> None:
    for pattern in STALE_TMP_PATTERNS:
        for path in home.glob(pattern):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                continue
            if now - mtime > stale_after:
                try:
                    path.unlink()
                except OSError:
                    logger.exception("failed to remove stale tmp file: %s", path)
                else:
                    report.removed_stale_tmp.append(path)
                    logger.warning("removed stale tmp file: %s", path)
