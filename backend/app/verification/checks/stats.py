"""Statistical checks [doc 06 §4].

num_classes, per-class minimum, dataset size, image resolution, object size,
colour mode and tabular quality. Everything reads the cached manifest facts
so nothing is recomputed downstream [doc 06 §6].
"""

from __future__ import annotations

from app.verification.checks.base import Check
from app.verification.context import CheckResult, VerificationContext
from app.verification.manifest import ManifestSummary

CLASSIFICATION_TASKS = {"image_classification", "tabular_classification"}


def _summary(ctx: VerificationContext) -> ManifestSummary | None:
    summary = ctx.facts.get("manifest")
    return summary if isinstance(summary, ManifestSummary) else None


class NumClassesCheck(Check):
    """Class count and names; error when a classification has < 2 [doc 06 §4]."""

    name = "num_classes"
    severity = "info"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        if summary is None:
            return CheckResult(
                name=self.name, severity=self.severity, status="pass", message="no manifest"
            )
        names = list(summary.class_names)
        task = ctx.facts.get("task_type") or ctx.task_type
        details = {"num_classes": len(names), "class_names": names[:50]}
        if not names:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="no per-class labels in this dataset",
                details=details,
            )
        if len(names) < 2 and task in CLASSIFICATION_TASKS:
            return CheckResult(
                name=self.name,
                severity="error",
                status="fail",
                message=f"only {len(names)} class ({names[0]!r}) for a classification task",
                details=details,
                fix_hint="Add at least two classes; single-class classification cannot train.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{len(names)} classes: {', '.join(names[:8])}"
            + (" …" if len(names) > 8 else ""),
            details=details,
        )


class MinSamplesPerClassCheck(Check):
    """Rare classes make metrics meaningless [doc 06 §4].

    Thresholds: < 2 samples → error, < 10 → warn (ASSUMPTION on the exact
    numbers; doc 06 §4 gives them as examples).
    """

    name = "min_samples_per_class"
    severity = "warn"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        counts = dict(summary.class_counts) if summary is not None else {}
        if not counts:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="no per-class counts",
            )
        minimum = min(counts.values())
        rare = sorted(name for name, value in counts.items() if value < 2)
        if rare:
            return CheckResult(
                name=self.name,
                severity="error",
                status="fail",
                message=f"class(es) with fewer than 2 samples: {rare[:5]}",
                details={"class_counts": counts},
                fix_hint="Add samples for these classes or merge them.",
            )
        low = {name: value for name, value in counts.items() if value < 10}
        if low:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"{len(low)} class(es) have fewer than 10 samples (min {minimum})",
                details={"class_counts": counts},
                fix_hint="Collect more samples per class, or rely on augmentation/weighting.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"every class has at least 10 samples (min {minimum})",
            details={"min": minimum},
        )


class DatasetSizeCheck(Check):
    """Tiny datasets need freezing / stronger augmentation [doc 06 §4]."""

    name = "dataset_size"
    severity = "info"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        total = len(summary.rows) if summary is not None else 0
        if total == 0:
            return CheckResult(
                name=self.name, severity=self.severity, status="pass", message="no rows"
            )
        if total < 300:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"tiny dataset ({total} samples): freeze the backbone and augment",
                details={"samples": total},
                fix_hint="Expect low accuracy; freeze pretrained layers and augment aggressively.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{total} samples",
            details={"samples": total},
        )


class ImageResolutionStatsCheck(Check):
    """min/max/median resolution and very small images [doc 06 §4]."""

    name = "image_resolution_stats"
    severity = "info"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        stats = dict(summary.image_stats) if summary is not None else {}
        if not stats:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="no image dimensions (tabular dataset or undecodable images)",
            )
        tiny = int(stats.get("tiny_images") or 0)
        details = {key: value for key, value in stats.items()}
        if tiny:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"{tiny} image(s) smaller than 32 px on a side",
                details=details,
                fix_hint="Upscaling very small images rarely helps; check them first.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=(
                f"width {stats.get('min_wh', [0])[0]}–{stats.get('max_wh', [0])[0]} px, "
                f"median {stats.get('median_w')} px"
            ),
            details=details,
        )


class ObjectSizeStatsCheck(Check):
    """Many tiny boxes → recommend a larger input size [doc 06 §4]."""

    name = "object_size_stats"
    severity = "info"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        stats = dict(summary.object_stats) if summary is not None else {}
        if not stats:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="no bounding boxes (not a detection dataset)",
            )
        tiny = int(stats.get("tiny_boxes") or 0)
        boxes = int(stats.get("boxes") or 0)
        details = dict(stats)
        if tiny and boxes and tiny / boxes > 0.3:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"{tiny}/{boxes} boxes cover < 0.1% of the image area",
                details=details,
                fix_hint="Use a larger input size (e.g. imgsz 960) to keep small objects visible.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{boxes} boxes, median area {stats.get('median_area')}",
            details=details,
        )


class ColorModeCheck(Check):
    """Grayscale vs RGB vs RGBA consistency [doc 06 §4]."""

    name = "color_mode"
    severity = "info"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        stats = dict(summary.image_stats) if summary is not None else {}
        modes = stats.get("color_modes") or {}
        if not modes:
            return CheckResult(
                name=self.name, severity=self.severity, status="pass", message="no image modes"
            )
        if len(modes) == 1:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message=f"all sampled images are {next(iter(modes))}",
                details={"modes": modes},
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="warn",
            message=f"mixed colour modes: {modes}",
            details={"modes": modes},
            fix_hint="Convert the outlier images so the model sees a consistent channel count.",
        )


class TabularQualityCheck(Check):
    """Missing values, constant columns, high cardinality [doc 06 §4]."""

    name = "tabular_quality"
    severity = "warn"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        schema = dict(summary.tabular_schema) if summary is not None else {}
        if not schema:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="not a tabular dataset",
            )
        columns = [column for column in schema.get("columns", []) if isinstance(column, dict)]
        target = schema.get("target")
        problems: list[str] = []
        for column in columns:
            name = str(column.get("name"))
            if name == target:
                continue
            if float(column.get("missing_pct") or 0) > 10:
                problems.append(f"{name}: {column.get('missing_pct')}% missing")
            if column.get("constant"):
                problems.append(f"{name}: constant column")
            numeric = bool(column.get("numeric"))
            if not numeric and float(column.get("distinct") or 0) >= max(
                10, 0.9 * int(schema.get("rows") or 1)
            ):
                problems.append(f"{name}: high cardinality ({column.get('distinct')} values)")
        details = {
            "rows": schema.get("rows"),
            "target": target,
            "columns": len(columns),
            "problems": problems[:10],
        }
        if problems:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"{len(problems)} tabular quality issue(s)",
                details=details,
                samples=problems[:10],
                fix_hint=(
                    "Drop constant columns, impute or drop missing values, "
                    "encode high-cardinality ids."
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{schema.get('rows')} rows, {len(columns)} columns, target {target!r}",
            details=details,
        )
