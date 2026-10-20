"""Content checks: corrupt files and label validity [doc 06 §4]."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

from app.verification.checks.base import Check
from app.verification.checks.format_supported import IMAGE_EXTENSIONS
from app.verification.context import CheckResult, VerificationContext


def _decode_image(path: Path) -> str | None:
    """Return an error description or None if the image decodes fine."""
    if path.stat().st_size == 0:
        return "zero-byte file"
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return "failed to decode"
    return None


class CorruptFilesCheck(Check):
    """Images that fail decode or are zero-byte; lists the first N [doc 06 §4]."""

    name = "corrupt_files"
    severity = "error"
    depends_on = ["format_supported"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        images = [
            path
            for path in ctx.dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="pass",
                message="no image files to decode (tabular dataset)",
                details={"checked": 0, "corrupt": 0},
            )
        sampled = images[: ctx.sample_cap]
        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(_decode_image, sampled))
        corrupt = [
            path.relative_to(ctx.dataset_root).as_posix()
            for path, error in zip(sampled, outcomes, strict=False)
            if error is not None
        ]
        details = {
            "checked": len(sampled),
            "total_images": len(images),
            "sampled": len(images) > len(sampled),
            "corrupt": len(corrupt),
        }
        if corrupt:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"{len(corrupt)} of {len(sampled)} image files are corrupt",
                details=details,
                samples=corrupt[:10],
                fix_hint="Remove or replace the corrupt files, then re-verify.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"all {len(sampled)} image files decode successfully",
            details=details,
        )


class LabelValidityCheck(Check):
    """Class ids in range, no NaN, bboxes inside [0,1] or image bounds [doc 06 §4]."""

    name = "label_validity"
    severity = "error"
    depends_on = ["format_supported"]

    def run(self, ctx: VerificationContext) -> CheckResult:
        detected = ctx.facts.get("format")
        if detected == "yolo":
            return self._check_yolo(ctx)
        if detected == "image_folder":
            return self._check_folder(ctx)
        if detected == "csv_path_label":
            return self._check_csv(ctx)
        if detected == "coco":
            return self._check_coco(ctx)
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message="no structured labels to validate for this format",
        )

    def _check_yolo(self, ctx: VerificationContext) -> CheckResult:
        problems: list[str] = []
        class_counts: dict[int, int] = {}
        checked = 0
        for path in ctx.dataset_root.rglob("*.txt"):
            if path.name.lower() == "classes.txt":
                continue
            for line_number, raw in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                line = raw.strip()
                if not line:
                    continue
                checked += 1
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"{path.name}:{line_number}: expected 5 values")
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    problems.append(f"{path.name}:{line_number}: non-numeric values")
                    continue
                if class_id < 0:
                    problems.append(f"{path.name}:{line_number}: negative class id")
                    continue
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                x, y, width, height = values
                inside = (
                    0.0 <= x <= 1.0
                    and 0.0 <= y <= 1.0
                    and 0.0 <= width <= 1.0
                    and 0.0 <= height <= 1.0
                )
                if not inside:
                    problems.append(f"{path.name}:{line_number}: box outside [0,1]")
                elif width <= 0 or height <= 0:
                    problems.append(f"{path.name}:{line_number}: zero-area box")
        # data.yaml class-count consistency [doc 06 §4 class_names_consistent].
        data_yaml = next(
            (p for p in ctx.dataset_root.rglob("*") if p.name.lower() in ("data.yaml", "data.yml")),
            None,
        )
        details: dict[str, int | None] = {
            "checked": checked,
            "invalid": len(problems),
            "classes": len(class_counts),
        }
        if data_yaml is not None:
            try:
                parsed = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                names = parsed.get("names") if isinstance(parsed, dict) else None
                if isinstance(names, list) and class_counts:
                    max_id = max(class_counts)
                    if max_id >= len(names):
                        problems.append(
                            f"{data_yaml.name}: class ids up to {max_id} but "
                            f"names lists {len(names)}"
                        )
                details["declared_classes"] = len(names) if isinstance(names, list) else None
            except yaml.YAMLError:
                problems.append(f"{data_yaml.name}: malformed YAML")
        ctx.facts["class_counts"] = class_counts
        ctx.facts["class_names"] = [str(i) for i in sorted(class_counts)]
        if problems:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"{len(problems)} invalid label lines",
                details=details,
                samples=problems[:10],
                fix_hint="Fix class ids and bounding boxes (YOLO: class x y w h in [0,1]).",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{checked} label lines are valid across {len(class_counts)} classes",
            details=details,
        )

    def _check_folder(self, ctx: VerificationContext) -> CheckResult:
        class_names = sorted(
            entry.name
            for entry in ctx.dataset_root.iterdir()
            if entry.is_dir() and any(entry.iterdir())
        )
        if len(class_names) < 2:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=(
                    f"found {len(class_names)} class folder(s); "
                    "classification needs at least 2"
                ),
                details={"classes": class_names},
                fix_hint="Provide at least two non-empty class folders.",
            )
        class_counts = {
            name: sum(1 for entry in (ctx.dataset_root / name).iterdir() if entry.is_file())
            for name in class_names
        }
        ctx.facts["class_counts"] = class_counts
        ctx.facts["class_names"] = class_names
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{len(class_names)} class folders with labels",
            details={"classes": class_names, "counts": class_counts},
        )

    def _check_csv(self, ctx: VerificationContext) -> CheckResult:
        csv_path = next(ctx.dataset_root.rglob("*.csv"), None)
        if csv_path is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="csv with image paths not found",
                fix_hint="Provide a CSV with a path column and a label column.",
            )
        lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
        header = [part.strip().lower() for part in lines[0].split(",")]
        try:
            path_col = next(i for i, name in enumerate(header) if name in ("path", "image", "file"))
            label_col = next(i for i, name in enumerate(header) if name == "label")
        except StopIteration:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="csv must have a path/image column and a label column",
                details={"header": header},
                fix_hint="Use columns `path,label` (or `image,label`).",
            )
        problems: list[str] = []
        class_counts: dict[str, int] = {}
        checked = 0
        for row_number, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) <= max(path_col, label_col):
                problems.append(f"{csv_path.name}:{row_number}: too few columns")
                continue
            checked += 1
            label = parts[label_col]
            if not label:
                problems.append(f"{csv_path.name}:{row_number}: empty label")
                continue
            class_counts[label] = class_counts.get(label, 0) + 1
            relative = parts[path_col]
            if not (ctx.dataset_root / relative).is_file():
                problems.append(f"{csv_path.name}:{row_number}: missing file {relative}")
        ctx.facts["class_counts"] = class_counts
        ctx.facts["class_names"] = sorted(class_counts)
        details = {"checked": checked, "invalid": len(problems), "classes": len(class_counts)}
        if problems:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"{len(problems)} invalid rows",
                details=details,
                samples=problems[:10],
                fix_hint="Fix labels and paths in the CSV.",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{checked} rows are valid across {len(class_counts)} classes",
            details=details,
        )

    def _check_coco(self, ctx: VerificationContext) -> CheckResult:
        coco_path = next(ctx.dataset_root.rglob("*.json"), None)
        if coco_path is None:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="COCO annotations json not found",
                fix_hint="Provide a COCO-style annotations file.",
            )
        try:
            data = json.loads(coco_path.read_text(encoding="utf-8"))
        except ValueError:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message="COCO json is not valid JSON",
                fix_hint="Fix the annotations file.",
            )
        images = {entry["id"]: entry for entry in data.get("images", [])}
        categories = {entry["id"]: entry for entry in data.get("categories", [])}
        problems: list[str] = []
        class_counts: dict[int, int] = {}
        checked = 0
        for annotation in data.get("annotations", []):
            checked += 1
            image = images.get(annotation.get("image_id"))
            box = annotation.get("bbox")
            category_id = annotation.get("category_id")
            if category_id not in categories:
                problems.append(
                    f"annotation {annotation.get('id')}: unknown category {category_id}"
                )
                continue
            class_counts[category_id] = class_counts.get(category_id, 0) + 1
            if image is None:
                problems.append(f"annotation {annotation.get('id')}: unknown image id")
                continue
            if not isinstance(box, list) or len(box) != 4:
                problems.append(f"annotation {annotation.get('id')}: malformed bbox")
                continue
            x, y, width, height = (float(value) for value in box)
            if width <= 0 or height <= 0:
                problems.append(f"annotation {annotation.get('id')}: zero-area bbox")
            else:
                right = x + width
                bottom = y + height
                if right > float(image.get("width", 0)) or bottom > float(image.get("height", 0)):
                    problems.append(
                        f"annotation {annotation.get('id')}: bbox outside image bounds"
                    )
        ctx.facts["class_counts"] = class_counts
        ctx.facts["class_names"] = [
            str(categories[cid].get("name", cid)) for cid in sorted(class_counts)
        ]
        details = {
            "checked": checked,
            "invalid": len(problems),
            "classes": len(categories),
        }
        if problems:
            return CheckResult(
                name=self.name,
                severity=self.severity,
                status="fail",
                message=f"{len(problems)} invalid annotations",
                details=details,
                samples=problems[:10],
                fix_hint="Fix the COCO annotations (category ids, bboxes within image bounds).",
            )
        return CheckResult(
            name=self.name,
            severity=self.severity,
            status="pass",
            message=f"{checked} annotations are valid across {len(categories)} categories",
            details=details,
        )
