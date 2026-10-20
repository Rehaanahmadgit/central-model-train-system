"""Pipeline runner [doc 06 §2].

Order: format detection → manifest build → task detection → checks in
dependency order → derived facts + recommendations. Everything expensive is
computed once here and cached for training [doc 06 §6].

A failing error-severity check short-circuits its dependents (`skipped`).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from app.core.device import DeviceInfo, resolve_device
from app.verification import VERIFIER_VERSION
from app.verification.checks.archive_integrity import ArchiveIntegrityCheck, TreeLimits
from app.verification.checks.base import Check, skipped_result
from app.verification.checks.class_distribution import ClassDistributionCheck
from app.verification.checks.compat import (
    InputRequirementsCheck,
    ResourceEstimateCheck,
    TaskModelMatchCheck,
)
from app.verification.checks.content import CorruptFilesCheck, LabelValidityCheck
from app.verification.checks.format_supported import FormatSupportedCheck, detect_format
from app.verification.checks.quality import (
    ClassNamesConsistentCheck,
    DuplicateFilesCheck,
    EmptyLabelsCheck,
    LabelFilePairingCheck,
    LeakageBetweenSplitsCheck,
)
from app.verification.checks.stats import (
    ColorModeCheck,
    DatasetSizeCheck,
    ImageResolutionStatsCheck,
    MinSamplesPerClassCheck,
    NumClassesCheck,
    ObjectSizeStatsCheck,
    TabularQualityCheck,
)
from app.verification.checks.structural import (
    FileExtensionsCheck,
    RequiredFilesPresentCheck,
    SplitPresenceCheck,
)
from app.verification.context import CheckResult, VerificationContext
from app.verification.manifest import ManifestSummary, build_manifest
from app.verification.recommend import (
    rank_checkpoints,
    recommend_hparams,
    size_bucket,
)
from app.verification.tasks import detect_task

Progress = Callable[[dict[str, Any]], None]

REPORT_SCHEMA_VERSION = 1


def build_checks(limits: TreeLimits | None = None) -> list[Check]:
    """Pipeline order [doc 06 §2]: structural → content → statistical → compatibility."""
    return [
        ArchiveIntegrityCheck(limits),
        FormatSupportedCheck(),
        RequiredFilesPresentCheck(),
        FileExtensionsCheck(),
        SplitPresenceCheck(),
        CorruptFilesCheck(),
        LabelFilePairingCheck(),
        LabelValidityCheck(),
        EmptyLabelsCheck(),
        DuplicateFilesCheck(),
        LeakageBetweenSplitsCheck(),
        ClassNamesConsistentCheck(),
        NumClassesCheck(),
        ClassDistributionCheck(),
        MinSamplesPerClassCheck(),
        DatasetSizeCheck(),
        ImageResolutionStatsCheck(),
        ObjectSizeStatsCheck(),
        ColorModeCheck(),
        TabularQualityCheck(),
        TaskModelMatchCheck(),
        InputRequirementsCheck(),
        ResourceEstimateCheck(),
    ]


def _execute_check(check: Check, ctx: VerificationContext, timeout_seconds: float) -> CheckResult:
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


def _prepare(ctx: VerificationContext, home: Path | None) -> None:
    """Format → manifest → task, computed once and shared with every check."""
    detected_format, reason = detect_format(ctx.dataset_root)
    ctx.facts["format"] = detected_format
    ctx.facts["format_reason"] = reason
    summary = build_manifest(ctx.dataset_root, detected_format, sample_cap=ctx.sample_cap)
    ctx.facts["manifest"] = summary
    task_type, task_reason = detect_task(ctx.dataset_root, detected_format, summary)
    ctx.facts["task_type"] = task_type
    ctx.facts["task_detection_reason"] = task_reason
    if summary.class_counts:
        ctx.facts.setdefault("class_counts", dict(summary.class_counts))
        ctx.facts.setdefault("class_names", list(summary.class_names))
    if home is not None:
        from app.checkpoints.catalog import CatalogError, load_catalog

        try:
            entries = load_catalog(home)
        except CatalogError:
            entries = []
        ctx.facts["checkpoint_catalog_configured"] = bool(entries)
        ctx.facts["checkpoint_catalog_ready"] = sum(1 for entry in entries if entry.verified)
        ctx.facts["catalog_entries"] = entries
    else:
        ctx.facts["checkpoint_catalog_configured"] = False
        ctx.facts["checkpoint_catalog_ready"] = 0


def _read_checkpoint_index(ctx: VerificationContext) -> dict[str, Any]:
    """Local availability comes from index.json [doc 04 §1]; read-only here."""
    catalog_path = ctx.facts.get("catalog_path")
    if not isinstance(catalog_path, str):
        return {}
    index_path = Path(catalog_path).parent / "index.json"
    if not index_path.is_file():
        return {}
    try:
        from app.storage.atomic import read_json

        data = read_json(index_path)
    except (OSError, ValueError):
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    return entries if isinstance(entries, dict) else {}


def _derived(
    ctx: VerificationContext, summary: ManifestSummary, device: DeviceInfo
) -> dict[str, Any]:
    """Derived facts cached for training [doc 06 §6]."""
    total = sum(summary.class_counts.values())
    hparams = recommend_hparams(
        summary,
        task_type=ctx.facts.get("task_type"),
        model_family=ctx.model_family,
        device=device,
        preset=str(ctx.facts.get("preset") or "") or None,
    )
    catalog = list(ctx.facts.get("catalog_entries") or [])
    recommended = rank_checkpoints(
        catalog,
        task_type=ctx.facts.get("task_type"),
        model_family=ctx.model_family,
        bucket=size_bucket(total),
        device=device,
    )
    local_index = _read_checkpoint_index(ctx)
    for item in recommended:
        record = local_index.get(str(item["catalog_id"])) or {}
        item["local"] = record.get("status") == "ready"
        item["local_size_bytes"] = record.get("size") if item["local"] else None
    return {
        "task_type": ctx.facts.get("task_type"),
        "task_detection_reason": ctx.facts.get("task_detection_reason"),
        "format": ctx.facts.get("format"),
        "num_classes": len(summary.class_names),
        "classes": list(summary.class_names),
        "class_counts": dict(summary.class_counts),
        "class_to_id": dict(summary.class_to_id),
        "num_images": len(summary.rows),
        "num_samples": dict(summary.num_samples),
        "image_stats": dict(summary.image_stats),
        "object_stats": dict(summary.object_stats),
        "tabular_schema": dict(summary.tabular_schema),
        "split_manifest_path": None,
        "recommended_hparams": hparams,
        "recommended_checkpoints": recommended,
        "checkpoint_catalog_configured": bool(ctx.facts.get("checkpoint_catalog_configured")),
        "checkpoint_catalog_ready": int(ctx.facts.get("checkpoint_catalog_ready") or 0),
        "preset": ctx.facts.get("preset"),
        "preset_requested": ctx.facts.get("preset_requested"),
        "dataset_hash": ctx.dataset_hash,
        "verifier_version": VERIFIER_VERSION,
    }


def run_pipeline(
    dataset_root: Path,
    dataset_hash: str,
    model_family: str,
    task_type: str | None,
    *,
    preset: str = "auto",
    sample_cap: int,
    check_timeout_seconds: float,
    progress: Progress,
    limits: TreeLimits | None = None,
    home: Path | None = None,
    device: DeviceInfo | None = None,
) -> dict[str, Any]:
    """Run all checks; returns status, checks, derived facts and manifest rows."""
    ctx = VerificationContext(
        dataset_root=dataset_root,
        dataset_hash=dataset_hash,
        model_family=model_family,
        task_type=task_type,
        verifier_version=VERIFIER_VERSION,
        sample_cap=sample_cap,
    )
    resolved_device = device or resolve_device("auto")
    # "auto" means: use the CPU dev preset unless a GPU is actually available [doc 14 §3.3].
    resolved_preset = preset if preset in ("dev", "full") else (
        "full" if resolved_device.is_gpu else "dev"
    )
    ctx.facts["preset"] = resolved_preset
    ctx.facts["preset_requested"] = preset
    if home is not None:
        ctx.facts["catalog_path"] = str(home / "checkpoints" / "catalog.yaml")

    progress({"type": "phase", "data": {"phase": "manifest_build", "status": "running"}})
    _prepare(ctx, home)
    summary = ctx.facts["manifest"]
    assert isinstance(summary, ManifestSummary)
    progress(
        {
            "type": "phase",
            "data": {
                "phase": "manifest_build",
                "status": "done",
                "files": len(summary.rows),
                "task_type": ctx.facts.get("task_type"),
            },
        }
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
    if not any(result.status == "fail" and result.severity == "error" for result in results):
        status = (
            "passed_with_warnings"
            if any(result.status == "warn" for result in results)
            else "passed"
        )
    return {
        "status": status,
        "can_train": status != "failed",
        "checks": [result.as_dict() for result in results],
        "derived": _derived(ctx, summary, resolved_device),
        "manifest_rows": summary.rows,
    }
