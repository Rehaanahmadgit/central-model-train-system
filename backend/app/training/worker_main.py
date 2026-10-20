"""Training subprocess entry point [doc 08 §3 step 4-5].

Spawned by the Job Manager as ``python -m app.training.worker_main --home … --run-id …``.
Loads the plugin, runs prepare → train → evaluate and reports the result on
stdout. Any failure is classified and emitted as a ``state failed`` event so
the parent records the reason [doc 08 §3.8].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.device import resolve_device
from app.storage.atomic import read_json
from app.training.registry import get_plugin
from app.training.worker import WorkerRunContext, emit, start_control_and_heartbeat

FAILURE_HINTS = (
    ("nan", "NAN_LOSS"),
    ("memory", "OOM"),
    ("cuda out of memory", "OOM"),
    ("valueerror", "DATA_ERROR"),
    ("keyerror", "DATA_ERROR"),
    ("filenotfound", "DATA_ERROR"),
)


def classify_failure(exc: BaseException) -> tuple[str, str]:
    message = f"{type(exc).__name__}: {exc}".lower()
    for needle, reason in FAILURE_HINTS:
        if needle in message:
            return reason, f"{type(exc).__name__}: {exc}"
    return "UNKNOWN", f"{type(exc).__name__}: {exc}"


def _run(record: dict[str, Any], home: Path, run_id: str) -> int:
    dataset_hash = str(record.get("dataset_hash") or "")
    model_family = str(record.get("model_family") or "")
    task_type = record.get("task_type")
    config = dict(record.get("config") or {})
    seed = int(record.get("seed") or 42)

    verification_id = str(record.get("verification_id") or "")
    report_path = home / "verifications" / verification_id / "report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    facts = dict(report.get("derived") or {}) if isinstance(report, dict) else {}
    manifest_path = home / "datasets" / dataset_hash / "manifest.parquet"
    dataset_root = home / "datasets" / dataset_hash / "raw"
    checkpoint_record = record.get("checkpoint")
    checkpoint_path = (
        Path(str(checkpoint_record["path"]))
        if isinstance(checkpoint_record, dict) and checkpoint_record.get("path")
        else None
    )

    plugin = get_plugin(model_family)
    if plugin is None:
        emit("state", {"from": "preparing", "to": "failed", "failure": {"reason": "UNKNOWN"}})
        return 1

    issues = plugin.validate_config(config, facts)
    errors = [issue for issue in issues if issue.severity == "error"]
    for issue in issues:
        emit("log", {"level": issue.severity.upper(), "message": f"{issue.path}: {issue.message}"})
    if errors:
        emit(
            "state",
            {
                "from": "preparing",
                "to": "failed",
                "failure": {"reason": "DATA_ERROR", "details": errors[0].message},
            },
        )
        return 1

    device = resolve_device(Settings().device)
    ctx = WorkerRunContext(
        home=home,
        run_id=run_id,
        dataset_hash=dataset_hash,
        model_family=model_family,
        task_type=str(task_type) if task_type else None,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        facts=facts,
        config=config,
        output_dir=home / "runs" / run_id / "artifacts",
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
    )

    emit(
        "log",
        {"level": "INFO", "message": f"trainer {model_family} starting (device={device.kind})"},
    )
    emit("state", {"from": "preparing", "to": "running"})
    plugin.prepare(ctx)
    resume_from = record.get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from))
        if resume_path.is_file():
            emit("log", {"level": "INFO", "message": f"resuming from {resume_path.name}"})
            plugin.resume(ctx, resume_path)
        else:
            emit(
                "log",
                {"level": "WARNING", "message": f"resume checkpoint missing: {resume_path}"},
            )
    result = plugin.train(ctx)
    if ctx.should_stop():
        emit("state", {"from": "running", "to": "stopped"})
        emit("result", {"state": "stopped", "best": result.best})
        return 0
    evaluation: dict[str, Any] = plugin.evaluate(ctx) or {}
    emit("result", {"state": "completed", "best": result.best, "evaluation": evaluation})
    emit("state", {"from": "running", "to": "completed"})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trainhub-worker")
    parser.add_argument("--home", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    home = Path(args.home)
    run_id = args.run_id
    start_control_and_heartbeat(
        sys.stdin, sys.stdout, heartbeat_seconds=Settings().heartbeat_seconds
    )

    record = read_json(home / "runs" / run_id / "run.json")
    if not isinstance(record, dict):
        emit("state", {"from": "preparing", "to": "failed", "failure": {"reason": "DATA_ERROR"}})
        return 1
    try:
        return _run(record, home, run_id)
    except BaseException as exc:  # noqa: B036 - the worker must always report and exit
        reason, details = classify_failure(exc)
        emit("log", {"level": "ERROR", "message": details})
        emit(
            "state",
            {"from": "running", "to": "failed", "failure": {"reason": reason, "details": details}},
        )
        emit("result", {"state": "failed", "best": {}})
        return 1


if __name__ == "__main__":
    sys.exit(main())
