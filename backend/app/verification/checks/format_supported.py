"""Format detection + model compatibility [doc 06 §3/§4/§5].

Detects the dataset format from the extracted tree and maps it to a task
type; the selected model must support that task (`task_model_match`).
"""

from pathlib import Path

from app.verification.checks.base import Check
from app.verification.context import CheckResult, VerificationContext

# Model family → supported task types [doc 00 §3, doc 07 §2 task_types].
MODEL_TASK_MAP: dict[str, set[str]] = {
    "rtdetr": {"object_detection"},
    "yolo": {"object_detection"},
    "resnet": {"image_classification"},
    "efficientnet": {"image_classification"},
    "random_forest": {"tabular_classification", "tabular_regression"},
    "xgboost": {"tabular_classification", "tabular_regression"},
}

FORMAT_TO_TASK: dict[str, str] = {
    "coco": "object_detection",
    "yolo": "object_detection",
    "image_folder": "image_classification",
    "csv_path_label": "image_classification",
    "tabular": "tabular_classification",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SPLIT_DIR_NAMES = {
    "train",
    "training",
    "val",
    "valid",
    "validation",
    "test",
    "testing",
    "eval",
}


def detect_format(root: Path) -> tuple[str | None, str]:
    """Detect the normalized format and a human-readable reason [doc 06 §5]."""
    files = [path for path in root.rglob("*") if path.is_file()]
    by_suffix: dict[str, list[Path]] = {}
    for path in files:
        by_suffix.setdefault(path.suffix.lower(), []).append(path)
    images = [
        path for suffix, paths in by_suffix.items() if suffix in IMAGE_EXTENSIONS for path in paths
    ]
    has_images = bool(images)
    coco_files = [path for path in files if path.suffix.lower() == ".json"]
    if coco_files and has_images:
        return "coco", "found image files plus a COCO-style annotations json"
    yaml_files = [path for path in files if path.name.lower() in ("data.yaml", "data.yml")]
    label_txts = [
        path
        for path in files
        if path.suffix.lower() == ".txt" and path.name.lower() != "classes.txt"
    ]
    if yaml_files and (label_txts or has_images):
        return "yolo", "found data.yaml with YOLO-style label txt files"
    csv_files = [path for path in files if path.suffix.lower() == ".csv"]
    if csv_files:
        header = csv_files[0].read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        first_line = (header[0] if header else "").lower()
        if has_images and ("path" in first_line or "image" in first_line) and "label" in first_line:
            return "csv_path_label", "csv has image-path and label columns and images are present"
        if has_images:
            return "image_folder", "folder-per-class image layout (csv is metadata)"
        return "tabular", "tabular csv dataset"
    if has_images:
        return "image_folder", "folder-per-class image layout"
    return None, "no recognized dataset layout found"


class FormatSupportedCheck(Check):
    """Format recognized and compatible with the selected model (P0)."""

    name = "format_supported"
    severity = "error"

    def run(self, ctx: VerificationContext) -> CheckResult:
        # The runner computes this once for the manifest [doc 06 §2 step 2];
        # direct check tests fall back to detecting here.
        if "format" in ctx.facts:
            detected_format = ctx.facts["format"]
            reason = str(ctx.facts.get("format_reason", ""))
        else:
            detected_format, reason = detect_format(ctx.dataset_root)
        if detected_format is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"unrecognized dataset format: {reason}",
                fix_hint=(
                    "Provide one of: folder-per-class images, CSV (path,label), "
                    "YOLO txt + data.yaml, or COCO json."
                ),
            )
        task_type = FORMAT_TO_TASK[detected_format]
        supported = MODEL_TASK_MAP.get(ctx.model_family, set())
        ctx.facts["format"] = detected_format
        ctx.facts["task_type"] = task_type
        if not supported:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"unknown model family {ctx.model_family!r}",
                details={"detected_format": detected_format, "detected_task": task_type},
                fix_hint="Pick a supported model family.",
            )
        if task_type not in supported:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=(
                    f"model family {ctx.model_family!r} does not support "
                    f"the detected task {task_type!r}"
                ),
                details={
                    "detected_format": detected_format,
                    "detected_task": task_type,
                    "supported_tasks": sorted(supported),
                },
                fix_hint="Choose a model family that supports this task.",
            )
        message = f"format {detected_format!r} detected ({reason}); task {task_type!r}"
        if ctx.task_type is not None and ctx.task_type != task_type:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=(
                    f"{message}; caller-provided task_type {ctx.task_type!r} "
                    "differs from the detected one"
                ),
                details={
                    "detected_format": detected_format,
                    "detected_task": task_type,
                    "provided_task": ctx.task_type,
                    "supported_tasks": sorted(supported),
                },
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=message,
            details={
                "detected_format": detected_format,
                "detected_task": task_type,
                "supported_tasks": sorted(supported),
            },
        )
