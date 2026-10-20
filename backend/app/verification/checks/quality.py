"""Content quality checks [doc 06 §4].

label pairing, images without annotations, exact duplicates, leakage across
splits and label-map consistency. All read the prebuilt manifest so the tree
is only walked once.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.verification.checks.base import Check
from app.verification.context import CheckResult, VerificationContext
from app.verification.manifest import ManifestSummary


def _pairing_key(relative: str) -> str:
    """Normalize ``labels/x.txt`` and ``images/x.png`` to the same key."""
    parts = list(Path(relative).parts)
    parts[-1] = Path(parts[-1]).stem
    return "/".join("images" if part.lower() == "labels" else part for part in parts)


def _summary(ctx: VerificationContext) -> ManifestSummary | None:
    summary = ctx.facts.get("manifest")
    return summary if isinstance(summary, ManifestSummary) else None


class LabelFilePairingCheck(Check):
    """Every image has a label (if required) and vice versa [doc 06 §4]."""

    name = "label_file_pairing"
    severity = "error"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        detected = ctx.facts.get("format")
        if summary is None:
            return self._pass("no manifest available")
        if detected != "yolo":
            return self._pass(f"pairing not required for format {detected!r}")

        missing_labels = [
            row["path"] for row in summary.rows if row.get("annotation_path") is None
        ]
        root = ctx.dataset_root
        images = {_pairing_key(str(row["path"])) for row in summary.rows if summary.rows}
        orphan_labels: list[str] = []
        for label_path in root.rglob("*.txt"):
            if label_path.name.lower() == "classes.txt":
                continue
            relative = label_path.relative_to(root).as_posix()
            if _pairing_key(relative) not in images:
                orphan_labels.append(relative)

        if missing_labels or orphan_labels:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=(
                    f"{len(missing_labels)} image(s) without a label file and "
                    f"{len(orphan_labels)} label file(s) without an image"
                ),
                details={
                    "missing_labels": len(missing_labels),
                    "orphan_labels": len(orphan_labels),
                },
                samples=(missing_labels + orphan_labels)[:10],
                fix_hint="Add the missing .txt label files, or remove the orphaned ones.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"all {len(summary.rows)} images are paired with a label file",
        )

    def _pass(self, message: str) -> CheckResult:
        return CheckResult(
            name=self.name, severity=self.severity, status="pass", message=message
        )


class EmptyLabelsCheck(Check):
    """Images with no annotations (could be background) [doc 06 §4]."""

    name = "empty_labels"
    severity = "warn"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        if summary is None or not summary.rows:
            return CheckResult(
                name=self.name, severity=self.severity, status="pass", message="no rows"
            )
        task = ctx.facts.get("task_type")
        if task not in ("object_detection", "segmentation"):
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="image-level labels: every sample carries a class",
            )
        empty = [row["path"] for row in summary.rows if not int(row.get("num_boxes") or 0)]
        share = 100 * len(empty) / len(summary.rows)
        if empty:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"{len(empty)} of {len(summary.rows)} images have no boxes ({share:.1f}%)",
                details={"empty": len(empty), "share_pct": round(share, 2)},
                samples=empty[:10],
                fix_hint="Confirm these are intentional background images.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message="every image has at least one annotation",
        )


class DuplicateFilesCheck(Check):
    """Exact duplicates by content hash [doc 06 §4]."""

    name = "duplicate_files"
    severity = "warn"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        if summary is None:
            return CheckResult(
                name=self.name, severity=self.severity, status="pass", message="no manifest"
            )
        seen: dict[str, list[str]] = {}
        for row in summary.rows:
            digest = row.get("file_sha256")
            if digest:
                seen.setdefault(str(digest), []).append(str(row["path"]))
        duplicates = {digest: paths for digest, paths in seen.items() if len(paths) > 1}
        if duplicates:
            sample = [path for paths in duplicates.values() for path in paths[1:]][:10]
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"{len(duplicates)} duplicate content hash(es) found",
                details={
                    "duplicate_groups": len(duplicates),
                    "hashed": summary.files_hashed,
                },
                samples=sample,
                fix_hint="Remove duplicated files so they are not counted twice.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"no duplicates among {summary.files_hashed} hashed file(s)",
        )


class LeakageBetweenSplitsCheck(Check):
    """Same content in train and val/test leaks answers [doc 06 §4]."""

    name = "leakage_between_splits"
    severity = "error"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = _summary(ctx)
        if summary is None:
            return CheckResult(
                name=self.name, severity=self.severity, status="pass", message="no manifest"
            )
        by_hash: dict[str, set[str]] = {}
        for row in summary.rows:
            digest = row.get("file_sha256")
            if digest:
                by_hash.setdefault(str(digest), set()).add(str(row["split"]))
        leaking = [
            digest
            for digest, splits in by_hash.items()
            if "train" in splits and ({"val", "test"} & splits)
        ]
        if leaking:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"{len(leaking)} file(s) appear in both train and val/test",
                details={"leaking_hashes": len(leaking)},
                fix_hint="Remove the duplicated files from one of the splits.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message="no identical files shared between train and val/test",
        )


class ClassNamesConsistentCheck(Check):
    """Class ids stay inside the declared name map [doc 06 §4]."""

    name = "class_names_consistent"
    severity = "error"
    depends_on = ["format_supported", "required_files_present"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        detected = ctx.facts.get("format")
        summary = _summary(ctx)
        if detected != "yolo" or summary is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="no separate class-name map to compare for this format",
            )
        root = ctx.dataset_root
        yaml_file = next(
            (path for path in root.rglob("*") if path.name.lower() in ("data.yaml", "data.yml")),
            None,
        )
        if yaml_file is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="data.yaml missing",
                fix_hint="Add data.yaml with a names list.",
            )
        try:
            parsed = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="data.yaml is not valid YAML",
                fix_hint="Fix the YAML syntax.",
            )
        declared = parsed.get("names") if isinstance(parsed, dict) else None
        if not isinstance(declared, list):
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="data.yaml 'names' is not a list",
                fix_hint="Use a YAML list: names: [cat, dog].",
            )
        used = sorted(int(label) for label in summary.class_counts)
        out_of_range = [value for value in used if value < 0 or value >= len(declared)]
        if out_of_range:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=(
                    f"class id(s) {out_of_range[:5]} outside the {len(declared)} declared names"
                ),
                details={"declared": len(declared), "used": used},
                fix_hint="Either add the missing names to data.yaml or fix the label ids.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{len(used)} class id(s) match the {len(declared)} declared names",
            details={"declared": len(declared), "used": used},
        )
