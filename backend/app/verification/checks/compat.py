"""Model compatibility checks [doc 06 §4, doc 14 §3.3].

Explicit task/model match, model input requirements and an honest resource
estimate that says "this model needs a GPU; this machine has none" instead of
failing somewhere deep in training.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.device import resolve_device
from app.verification.checks.base import Check
from app.verification.checks.format_supported import MODEL_TASK_MAP
from app.verification.context import CheckResult, VerificationContext
from app.verification.manifest import ManifestSummary

GPU_REQUIRED_FAMILIES = {"rtdetr", "yolo"}
MIN_IMAGES_PER_CLASS = 1


def _summary(ctx: VerificationContext) -> ManifestSummary | None:
    summary = ctx.facts.get("manifest")
    return summary if isinstance(summary, ManifestSummary) else None


class TaskModelMatchCheck(Check):
    """The selected model family supports the detected task [doc 06 §4]."""

    name = "task_model_match"
    severity = "error"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        task = ctx.facts.get("task_type")
        supported = MODEL_TASK_MAP.get(ctx.model_family, set())
        if task is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="task type could not be detected, so the model cannot be matched",
                fix_hint="Pass an explicit task_type, or fix the dataset layout.",
            )
        if task not in supported:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=(
                    f"model family {ctx.model_family!r} supports {sorted(supported)} "
                    f"but the dataset is {task!r}"
                ),
                details={"detected_task": task, "supported": sorted(supported)},
                fix_hint="Pick a model family that supports this task.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{ctx.model_family!r} supports {task!r}",
            details={"detected_task": task},
        )


class InputRequirementsCheck(Check):
    """Model input constraints such as channel count and class limits [doc 06 §4]."""

    name = "input_requirements"
    severity = "error"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        if summary is None:
            return CheckResult(
                name=self.name, severity=self.severity, status="pass", message="no manifest"
            )
        problems: list[str] = []
        modes = dict(summary.image_stats.get("color_modes") or {})
        if ctx.model_family in {"resnet", "efficientnet", "rtdetr", "yolo"}:
            unsupported = [mode for mode in modes if mode not in ("RGB", "L", "RGBA")]
            if unsupported:
                problems.append(f"unsupported colour mode(s): {unsupported}")
        if not summary.rows:
            problems.append("dataset has no samples")
        elif ctx.model_family in {"resnet", "efficientnet"} and len(summary.class_names) < 2:
            problems.append("classification needs at least two classes")
        if problems:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="; ".join(problems),
                details={"problems": problems, "color_modes": modes},
                fix_hint="Convert images to RGB (3 channels) and provide at least two classes.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message="model input requirements are satisfied",
            details={"color_modes": modes, "min_images_per_class": MIN_IMAGES_PER_CLASS},
        )


class ResourceEstimateCheck(Check):
    """Estimate VRAM/time and flag hardware mismatch [doc 06 §4, doc 14 §3.3]."""

    name = "resource_estimate"
    severity = "warn"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        try:
            device = resolve_device(get_settings().device)
        except Exception:  # pragma: no cover - settings always valid in practice
            device = resolve_device("cpu")
        summary = _summary(ctx)
        samples = len(summary.rows) if summary is not None else 0
        bucket = "tiny" if samples < 1_000 else "small" if samples < 10_000 else "medium"
        needs_gpu = ctx.model_family in GPU_REQUIRED_FAMILIES
        details = {
            "device": device.as_dict(),
            "samples": samples,
            "dataset_bucket": bucket,
            "needs_gpu_family": needs_gpu,
        }
        if needs_gpu and not device.is_gpu:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=(
                    "This model needs a GPU; this machine has none "
                    f"({device.reason}). A CPU dev preset is required."
                ),
                details=details,
                fix_hint=(
                    "Choose a CPU-capable preset for a demo run, or deploy the GPU image on a "
                    "CUDA server [doc 14 §3.3]."
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=(
                f"{samples} samples on {device.kind}"
                f"{' (' + device.name + ')' if device.is_gpu else ''}; "
                f"{bucket} bucket"
            ),
            details=details,
        )
