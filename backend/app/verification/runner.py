"""Pipeline runner [doc 06 §2].

Runs checks in dependency order with per-check timeouts [doc 06 §7];
a failing error-severity check short-circuits its dependents (`skipped`).
Progress is reported through the provided callback.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from app.verification import VERIFIER_VERSION
from app.verification.checks.archive_integrity import ArchiveIntegrityCheck, TreeLimits
from app.verification.checks.base import Check, skipped_result
from app.verification.checks.class_distribution import ClassDistributionCheck
from app.verification.checks.content import CorruptFilesCheck, LabelValidityCheck
from app.verification.checks.format_supported import FormatSupportedCheck
from app.verification.context import CheckResult, VerificationContext

Progress = Callable[[dict[str, Any]], None]


def build_checks(limits: TreeLimits | None = None) -> list[Check]:
    # Pipeline order [doc 06 §2]: structural → format → content → statistical.
    return [
        ArchiveIntegrityCheck(limits),
        FormatSupportedCheck(),
        CorruptFilesCheck(),
        LabelValidityCheck(),
        ClassDistributionCheck(),
    ]


def _execute_check(
    check: Check, ctx: VerificationContext, timeout_seconds: float
) -> CheckResult:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(check.run, ctx)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout:
            return CheckResult(
                name=check.name,
                severity=check.severity,
                status="fail",
                message=f"check timed out after {timeout_seconds:.0f}s",
                fix_hint="Reduce the dataset size or raise the timeout.",
            )


def run_pipeline(
    dataset_root: Path,
    dataset_hash: str,
    model_family: str,
    task_type: str | None,
    *,
    sample_cap: int,
    check_timeout_seconds: float,
    progress: Progress,
    limits: TreeLimits | None = None,
) -> dict[str, Any]:
    """Run all checks; returns {status, can_train, checks}."""
    ctx = VerificationContext(
        dataset_root=dataset_root,
        dataset_hash=dataset_hash,
        model_family=model_family,
        task_type=task_type,
        verifier_version=VERIFIER_VERSION,
        sample_cap=sample_cap,
    )
    failed_error_checks: set[str] = set()
    results: list[CheckResult] = []
    for check in build_checks(limits):
        progress({"type": "check_started", "data": {"name": check.name}})
        blocked_by = [name for name in check.depends_on if name in failed_error_checks]
        if blocked_by:
            result = skipped_result(check.name, blocked_by[0])
        else:
            result = _execute_check(check, ctx, check_timeout_seconds)
        if result.status == "fail" and result.severity == "error":
            failed_error_checks.add(check.name)
        results.append(result)
        progress({"type": "check_finished", "data": result.as_dict()})
    status = "failed"
    if not any(
        result.status == "fail" and result.severity == "error" for result in results
    ):
        status = (
            "passed_with_warnings"
            if any(result.status == "warn" for result in results)
            else "passed"
        )
    return {
        "status": status,
        "can_train": status != "failed",
        "checks": [result.as_dict() for result in results],
    }
