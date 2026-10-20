"""Verification check unit tests [doc 06 §4, doc 11 §3]."""

from pathlib import Path

from fixtures.make_datasets import build_classification, build_yolo, materialize
from verification_fixtures import make_folder_dataset, make_yolo_dataset

from app.verification.checks.archive_integrity import ArchiveIntegrityCheck
from app.verification.checks.class_distribution import ClassDistributionCheck
from app.verification.checks.content import CorruptFilesCheck, LabelValidityCheck
from app.verification.checks.format_supported import FormatSupportedCheck
from app.verification.context import VerificationContext
from app.verification.runner import run_pipeline


def _ctx(
    root: Path, model_family: str = "rtdetr", task_type: str | None = None
) -> VerificationContext:
    return VerificationContext(
        dataset_root=root,
        dataset_hash="a" * 64,
        model_family=model_family,
        task_type=task_type,
        verifier_version="1.0.0",
        sample_cap=100,
    )


def test_archive_integrity_passes_clean_tree(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path)
    result = ArchiveIntegrityCheck().run(_ctx(raw))
    assert result.status == "pass"


def test_archive_integrity_fails_on_symlink(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path)
    (raw / "evil").symlink_to("/etc/passwd")
    result = ArchiveIntegrityCheck().run(_ctx(raw))
    assert result.status == "fail"
    assert any("symlink" in sample for sample in result.samples)


def test_archive_integrity_fails_on_missing_dir(tmp_path: Path) -> None:
    result = ArchiveIntegrityCheck().run(_ctx(tmp_path / "missing"))
    assert result.status == "fail"


def test_format_supported_yolo_for_rtdetr(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path)
    result = FormatSupportedCheck().run(_ctx(raw, "rtdetr"))
    assert result.status == "pass"
    assert result.details["detected_format"] == "yolo"


def test_format_supported_task_mismatch(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path)
    result = FormatSupportedCheck().run(_ctx(raw, "resnet"))
    assert result.status == "fail"
    assert result.details["detected_task"] == "object_detection"


def test_format_supported_unrecognized(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "random.txt").write_text("hello")
    result = FormatSupportedCheck().run(_ctx(raw, "rtdetr"))
    assert result.status == "fail"


def test_corrupt_files_detects_bad_image(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path, corrupt_image=True)
    ctx = _ctx(raw)
    FormatSupportedCheck().run(ctx)
    result = CorruptFilesCheck().run(ctx)
    assert result.status == "fail"
    assert result.samples and result.samples[0].startswith("images/img0.png")


def test_corrupt_files_passes_clean_images(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path)
    ctx = _ctx(raw)
    FormatSupportedCheck().run(ctx)
    result = CorruptFilesCheck().run(ctx)
    assert result.status == "pass"


def test_label_validity_rejects_bad_bbox(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path, bad_label="0 1.5 0.5 0.5 0.5")
    ctx = _ctx(raw)
    FormatSupportedCheck().run(ctx)
    result = LabelValidityCheck().run(ctx)
    assert result.status == "fail"
    assert any("outside [0,1]" in sample for sample in result.samples)


def test_label_validity_rejects_out_of_range_class(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path, bad_label="9 0.5 0.5 0.5 0.5")
    ctx = _ctx(raw)
    FormatSupportedCheck().run(ctx)
    result = LabelValidityCheck().run(ctx)
    assert result.status == "fail"


def test_label_validity_passes_good_yolo(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path)
    ctx = _ctx(raw)
    FormatSupportedCheck().run(ctx)
    result = LabelValidityCheck().run(ctx)
    assert result.status == "pass"
    assert ctx.facts["class_counts"] == {0: 3, 1: 1}


def test_class_distribution_warns_on_imbalance(tmp_path: Path) -> None:
    raw = make_yolo_dataset(tmp_path, imbalance=True)
    ctx = _ctx(raw)
    FormatSupportedCheck().run(ctx)
    LabelValidityCheck().run(ctx)
    result = ClassDistributionCheck().run(ctx)
    assert result.status == "warn"
    assert "imbalance" in result.message


def test_class_distribution_passes_balanced(tmp_path: Path) -> None:
    raw = make_folder_dataset(tmp_path)
    ctx = _ctx(raw, "resnet")
    FormatSupportedCheck().run(ctx)
    LabelValidityCheck().run(ctx)
    result = ClassDistributionCheck().run(ctx)
    assert result.status == "pass"


def test_pipeline_dependency_skip_on_failed_format(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "random.txt").write_text("hello")
    events: list[dict] = []
    outcome = run_pipeline(
        raw,
        "a" * 64,
        "rtdetr",
        None,
        sample_cap=100,
        check_timeout_seconds=30,
        progress=events.append,
    )
    assert outcome["status"] == "failed"
    assert outcome["can_train"] is False
    statuses = {
        event["data"]["name"]: event["data"]["status"]
        for event in events
        if event["type"] == "check_finished"
    }
    assert statuses["label_validity"] == "skipped"
    assert statuses["class_distribution"] == "skipped"


EXPECTED_CHECKS = [
    "archive_integrity",
    "format_supported",
    "required_files_present",
    "file_extensions",
    "split_presence",
    "corrupt_files",
    "label_file_pairing",
    "label_validity",
    "empty_labels",
    "duplicate_files",
    "leakage_between_splits",
    "class_names_consistent",
    "num_classes",
    "class_distribution",
    "min_samples_per_class",
    "dataset_size",
    "image_resolution_stats",
    "object_size_stats",
    "color_mode",
    "tabular_quality",
    "task_model_match",
    "input_requirements",
    "resource_estimate",
]


def test_pipeline_runs_every_documented_check(tmp_path: Path) -> None:
    """Full doc 06 §4 pipeline; a good YOLO set trains but warns about hardware."""
    raw = materialize(tmp_path / "raw", build_yolo())
    outcome = run_pipeline(
        raw,
        "a" * 64,
        "rtdetr",
        None,
        sample_cap=100,
        check_timeout_seconds=30,
        progress=lambda event: None,
    )
    assert [check["name"] for check in outcome["checks"]] == EXPECTED_CHECKS
    # On CPU the dev preset is chosen automatically, so no false GPU warning [doc 14 §3.3].
    assert outcome["can_train"] is True
    statuses = {check["name"]: check["status"] for check in outcome["checks"]}
    assert statuses["corrupt_files"] == "pass"
    assert statuses["resource_estimate"] == "pass"
    assert outcome["derived"]["preset"] == "dev"


def test_pipeline_clean_pass_with_splits(tmp_path: Path) -> None:
    """Good classification data with train/val splits passes with zero warnings."""
    raw = materialize(
        tmp_path / "raw", build_classification(per_class=160, split=True, size=(64, 64))
    )
    outcome = run_pipeline(
        raw,
        "a" * 64,
        "resnet",
        None,
        sample_cap=500,
        check_timeout_seconds=60,
        progress=lambda event: None,
    )
    assert outcome["status"] == "passed"
    assert outcome["can_train"] is True
    assert outcome["derived"]["task_type"] == "image_classification"
    assert outcome["derived"]["num_classes"] == 2
    assert set(outcome["derived"]["num_samples"]) >= {"train", "val"}


def test_pipeline_derived_facts_and_manifest(tmp_path: Path) -> None:
    raw = materialize(tmp_path / "raw", build_yolo())
    outcome = run_pipeline(
        raw,
        "a" * 64,
        "rtdetr",
        None,
        sample_cap=100,
        check_timeout_seconds=30,
        progress=lambda event: None,
    )
    derived = outcome["derived"]
    assert derived["format"] == "yolo"
    assert derived["task_type"] == "object_detection"
    assert "bounding boxes" in str(derived["task_detection_reason"])
    assert derived["num_images"] == len(outcome["manifest_rows"])
    assert derived["recommended_hparams"]["batch_size"] >= 1
    assert derived["recommended_checkpoints"] == []
    assert derived["checkpoint_catalog_configured"] is False
