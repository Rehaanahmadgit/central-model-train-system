"""Detection worker: ``python -m app.inference.detect_worker --job job.json`` [Slice G].

The API process must never import torch (forking the verification pool after torch
exists can hang), so every inference runs here, in its own process. The job file
lists image files in a temp directory the API deletes afterwards; the result is
written to ``out`` as JSON. Weights come only from the run's ``best.ckpt``
(``weights_only=True``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.core.device import resolve_device
from app.storage.atomic import read_json


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    from app.training.plugins.rtdetr import load_best_model, predict_paths

    home = Path(str(job["home"]))
    run_dir = home / "runs" / str(job["run_id"])
    record = read_json(run_dir / "run.json")
    if not isinstance(record, dict):
        raise FileNotFoundError("run.json is missing")
    config = dict(record.get("config") or {})
    imgsz = int(config.get("imgsz", 512))
    report = read_json(home / "verifications" / str(record.get("verification_id")) / "report.json")
    derived = dict(report.get("derived") or {}) if isinstance(report, dict) else {}
    classes = list(derived.get("classes") or derived.get("class_names") or [])
    if not classes:
        raise ValueError("class names are missing from the verification report")

    requested = "cpu" if job.get("avoid_gpu") else str(job.get("device") or "auto")
    info = resolve_device(requested)
    device = torch.device("cuda" if info.is_gpu else "cpu")
    model, postprocessor = load_best_model(run_dir, len(classes), imgsz, device)
    items = list(job["items"])
    rows = predict_paths(
        model,
        postprocessor,
        [Path(str(item["path"])) for item in items],
        imgsz=imgsz,
        device=device,
        threshold=float(job["threshold"]),
        amp=bool(config.get("amp", False)) and info.is_gpu,
    )
    return {
        "device": {"kind": info.kind, "name": info.name},
        "class_names": classes,
        "imgsz": imgsz,
        "images": [{"key": item["key"], **row} for item, row in zip(items, rows, strict=True)],
    }


def main(argv: list[str]) -> int:
    job_path = Path(argv[argv.index("--job") + 1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out = Path(str(job["out"]))
    try:
        result: dict[str, Any] = {"ok": True, **run_job(job)}
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - subprocess entry
    raise SystemExit(main(sys.argv[1:]))
