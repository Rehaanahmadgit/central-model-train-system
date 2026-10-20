"""Structural checks: required files, extensions, split presence [doc 06 §4].

These run before content checks because a missing annotation file makes the
content checks meaningless.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import yaml

from app.verification.checks.base import Check
from app.verification.checks.format_supported import IMAGE_EXTENSIONS
from app.verification.context import CheckResult, VerificationContext

KNOWN_EXTENSIONS = IMAGE_EXTENSIONS | {
    ".txt",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".parquet",
    ".md",
}

def _files(ctx: VerificationContext) -> list[Path]:
    return sorted(path for path in ctx.dataset_root.rglob("*") if path.is_file())


class RequiredFilesPresentCheck(Check):
    """Labels/yaml/annotation file exist for the detected format [doc 06 §4]."""

    name = "required_files_present"
    severity = "error"
    depends_on = ["format_supported"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        detected = ctx.facts.get("format")
        files = _files(ctx)

        if detected == "yolo":
            yaml_file = next(
                (path for path in files if path.name.lower() in ("data.yaml", "data.yml")), None
            )
            if yaml_file is None:
                return self._fail("data.yaml is missing", "Add a data.yaml listing class names.")
            try:
                parsed = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError):
                return self._fail(
                    "data.yaml is unreadable", "Fix the YAML syntax of data.yaml."
                )
            names_field = parsed.get("names") if isinstance(parsed, dict) else None
            if not names_field:
                return self._fail(
                    "data.yaml has no 'names' list",
                    "Add a 'names:' list with one entry per class.",
                )
            labels = [path for path in files if path.suffix.lower() == ".txt"]
            if not labels:
                return self._fail(
                    "no YOLO label .txt files",
                    "Add one .txt per image next to the images (labels/ folder).",
                )
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message=(
                    f"data.yaml with {len(names_field)} class names "
                    f"and {len(labels)} label files"
                ),
                details={"classes": len(names_field), "label_files": len(labels)},
            )

        if detected == "coco":
            if not any(path.suffix.lower() == ".json" for path in files):
                return self._fail("COCO annotations json is missing", "Add the annotations json.")
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="COCO annotations json present",
            )

        if detected in ("image_folder", "csv_path_label"):
            if not any(path.suffix.lower() in IMAGE_EXTENSIONS for path in files):
                return self._fail("no image files found", "Add images (jpg/png).")
            if detected == "csv_path_label":
                csv_file = next(
                    (path for path in files if path.suffix.lower() == ".csv"), None
                )
                if csv_file is None:
                    return self._fail("CSV label file is missing", "Add a CSV with path,label.")
                header = (
                    csv_file.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
                )
                first = (header[0] if header else "").lower()
                if "label" not in first:
                    return self._fail(
                        "CSV has no label column",
                        "Use columns `path,label` (or `image,label`).",
                    )
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="images and label source present",
            )

        if detected == "tabular":
            csv_file = next((path for path in files if path.suffix.lower() == ".csv"), None)
            if csv_file is None:
                return self._fail("no CSV file found", "Upload a CSV dataset.")
            lines = csv_file.read_text(encoding="utf-8", errors="replace").splitlines()
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            fields = [name for name in reader.fieldnames or []]
            if not any(name.lower() in ("target", "label", "class", "y") for name in fields):
                return self._fail(
                    "tabular CSV has no target column",
                    "Add a `target` (or `label`) column to the CSV.",
                    details={"columns": fields},
                )
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message=f"tabular CSV with {len(fields)} columns",
                details={"columns": fields},
            )

        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message="no required-file rule for this format",
        )

    def _fail(
        self, message: str, fix_hint: str, details: dict[str, object] | None = None
    ) -> CheckResult:
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="fail",
            message=message,
            details=dict(details or {}),
            fix_hint=fix_hint,
        )


class FileExtensionsCheck(Check):
    """Unsupported files are ignored but listed [doc 06 §4]."""

    name = "file_extensions"
    severity = "warn"
    depends_on = ["archive_integrity"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        unexpected = [
            path.relative_to(ctx.dataset_root).as_posix()
            for path in _files(ctx)
            if path.suffix.lower() not in KNOWN_EXTENSIONS
        ]
        if unexpected:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"{len(unexpected)} file(s) have extensions the pipeline does not use",
                details={"count": len(unexpected)},
                samples=unexpected[:10],
                fix_hint="Remove unrelated files, or confirm they can be ignored.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message="all files have recognized extensions",
        )


class SplitPresenceCheck(Check):
    """train/val/test exist; otherwise suggest a stratified split [doc 06 §4]."""

    name = "split_presence"
    severity = "warn"
    depends_on = ["format_supported"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        summary = ctx.facts.get("manifest")
        num_samples: dict[str, int] = (
            dict(summary.num_samples) if summary is not None else {}
        )
        present = [name for name in ("train", "val", "test") if num_samples.get(name)]
        if {"train", "val"} <= set(present):
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message=f"explicit splits found: {', '.join(present)}",
                details={"num_samples": num_samples},
            )
        if present:
            missing = sorted({"train", "val"} - set(present))
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="warn",
                message=f"partial splits: found {', '.join(present)}; missing {', '.join(missing)}",
                details={"num_samples": num_samples},
                fix_hint=(
                    "Add the missing split, or accept an automatic "
                    "stratified split at training time."
                ),
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="warn",
            message="no train/val/test directories detected; a stratified split will be created",
            details={"num_samples": num_samples},
            fix_hint="Use train/ and val/ subfolders, or accept the automatic stratified split.",
        )
