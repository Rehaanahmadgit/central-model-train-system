"""trainhub doctor tests [DIST-05, doc 14 §3.6]."""

from pathlib import Path

from app.core.doctor import (
    check_part_files,
    check_write,
    format_report,
    run_doctor,
)


def test_doctor_report_is_plain_words(tmp_path: Path) -> None:
    """The report must read like English, not JSON [DIST-05]."""
    report = run_doctor(tmp_path)
    names = {check.name for check in report.checks}
    assert {
        "python",
        "torch",
        "gpu",
        "disk",
        "write_permission",
        "partial_downloads",
        "checkpoint_hosts",
    } <= names
    text = format_report(report)
    assert "TrainHub doctor" in text
    assert "[OK] python:" in text
    for check in report.checks:
        if check.status == "fail":
            assert check.fix, f"{check.name} FAIL has no fix"


def test_doctor_reports_writable_workspace(tmp_path: Path) -> None:
    status, _message, fix = check_write(tmp_path)
    assert status == "ok"
    assert fix is None


def test_doctor_detects_leftover_part_files(tmp_path: Path) -> None:
    status, _message, _fix = check_part_files(tmp_path)
    assert status == "ok"

    part_dir = tmp_path / "checkpoints" / ".tmp"
    part_dir.mkdir(parents=True)
    (part_dir / "model.bin.part").write_bytes(b"x")
    status, message, fix = check_part_files(tmp_path)
    assert status == "warn"
    assert "1 leftover" in message
    assert "delete" in (fix or "")


def test_doctor_write_failure_hints_uid_10001(tmp_path: Path, monkeypatch) -> None:
    """A bind-mount ownership problem must point at uid 10001 and the fix [DIST-05]."""

    def denied(self, *args, **kwargs):  # noqa: ARG001
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", denied)
    status, _message, fix = check_write(tmp_path)
    assert status == "fail"
    assert "uid 10001" in (fix or "")
