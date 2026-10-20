"""Structural checks: raw-tree integrity [doc 06 §4, doc 10 §3].

The archive was already safely extracted at upload (doc 03 §3.1); this check
re-validates the extracted tree defensively (no symlinks/special files,
readable, within limits) so a tampered or broken tree never reaches training.
"""

import os
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path

from app.verification.checks.base import Check
from app.verification.context import CheckResult, VerificationContext


@dataclass(frozen=True)
class TreeLimits:
    max_files: int = 100_000
    max_total_bytes: int = 20 * 1024**3


class ArchiveIntegrityCheck(Check):
    name = "archive_integrity"
    severity = "error"

    def __init__(self, limits: TreeLimits | None = None) -> None:
        self._limits = limits or TreeLimits()

    def run(self, ctx: VerificationContext) -> CheckResult:
        root = ctx.dataset_root
        if not root.is_dir() or not any(root.iterdir()):
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="dataset directory is missing or empty",
                fix_hint="Re-upload the dataset.",
            )
        problems: list[str] = []
        file_count = 0
        total_bytes = 0
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            directory = Path(dirpath)
            for entry_name in list(dirnames):
                entry = directory / entry_name
                if entry.is_symlink():
                    problems.append(f"symlink: {entry.relative_to(root)}")
                    dirnames.remove(entry_name)  # never descend into symlinked dirs
            for entry_name in filenames:
                entry = directory / entry_name
                if entry.is_symlink():
                    problems.append(f"symlink: {entry.relative_to(root)}")
                    continue
                try:
                    mode = entry.stat().st_mode
                except OSError:
                    problems.append(f"unreadable: {entry.relative_to(root)}")
                    continue
                if not stat_module.S_ISREG(mode):
                    problems.append(f"special file: {entry.relative_to(root)}")
                else:
                    file_count += 1
                    total_bytes += entry.stat().st_size
        if problems:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"unsafe or unreadable files in the dataset tree: {len(problems)}",
                details={"file_count": file_count, "total_bytes": total_bytes},
                samples=[str(p) for p in problems[:10]],
                fix_hint="Re-upload the dataset; symlinks and special files are not allowed.",
            )
        if file_count > self._limits.max_files:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"too many files: {file_count}",
                details={"file_count": file_count, "limit": self._limits.max_files},
                fix_hint="Reduce the dataset size.",
            )
        if total_bytes > self._limits.max_total_bytes:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"dataset too large: {total_bytes} bytes",
                details={"total_bytes": total_bytes, "limit": self._limits.max_total_bytes},
                fix_hint="Reduce the dataset size.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{file_count} files, {total_bytes} bytes, tree is clean",
            details={"file_count": file_count, "total_bytes": total_bytes},
        )
