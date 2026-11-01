"""Honest-evaluation tests for the RT-DETR plugin [doc 08 §6].

Fast (no training loop): the split loader, the evaluation-report shape and the
best-checkpoint reload are exercised directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from fixtures.make_datasets import build_yolo, materialize, png_bytes

from app.training.plugins.rtdetr import (
    RTDetrTrainer,
    build_eval_report,
    build_predictions_payload,
    match_counts,
    split_warning,
)
from app.verification.manifest import build_manifest, write_manifest


def _ctx(tmp_path: Path, raw: Path, *, seed: int = 42, num_classes: int = 2) -> Any:
    summary = build_manifest(raw, "yolo", sample_cap=500)
    manifest = tmp_path / "manifest.parquet"
    write_manifest(manifest, summary.rows)
    logs: list[str] = []

    def emit_log(level: str, message: str) -> None:
        logs.append(f"{level} {message}")

    ctx = SimpleNamespace(
        home=tmp_path,
        run_id="run_test",
        dataset_hash="hash",
        model_family="rtdetr",
        task_type="object_detection",
        dataset_root=raw,
        manifest_path=manifest,
        facts={},
        config={},
        output_dir=tmp_path / "artifacts",
        checkpoint_path=None,
        device=SimpleNamespace(is_gpu=False, kind="cpu"),
        seed=seed,
        emit_log=emit_log,
        emit_metric=lambda *args, **kwargs: None,
        emit_event=lambda *args, **kwargs: None,
        emit_system=lambda *args, **kwargs: None,
        save_checkpoint=lambda *args, **kwargs: None,
        should_pause=lambda: False,
        should_stop=lambda: False,
        wait_if_paused=lambda: None,
    )
    trainer = RTDetrTrainer()
    trainer._num_classes = num_classes
    trainer._class_names = ["cat", "dog"][:num_classes]
    return ctx, trainer, logs


def _split_dataset(with_test: bool) -> dict[str, bytes]:
    """YOLO layout with real directory splits (``train/``, ``val/``, ``test/``)."""
    entries: dict[str, bytes] = {}
    layout = [("train", 8), ("val", 4)] + ([("test", 3)] if with_test else [])
    index = 0
    for split, count in layout:
        for _ in range(count):
            entries[f"{split}/images/img{index}.png"] = png_bytes((10 + index, 40, 90))
            entries[f"{split}/labels/img{index}.txt"] = b"0 0.25 0.25 0.5 0.5\n"
            index += 1
    entries["data.yaml"] = b"names:\n  - cat\n  - dog\n"
    return entries


def test_test_split_images_never_enter_train_or_val(tmp_path: Path) -> None:
    raw = materialize(tmp_path / "raw", _split_dataset(with_test=True))
    ctx, trainer, _logs = _ctx(tmp_path, raw)

    train, val, test = trainer._load_samples(ctx)

    assert len(train) == 8
    assert len(val) == 4
    assert len(test) == 3
    trained = {path for path, _boxes in train} | {path for path, _boxes in val}
    assert not any("/test/" in path.as_posix() for path in trained)
    assert {path.name for path, _ in test} == {"img12.png", "img13.png", "img14.png"}


def test_unspecified_split_falls_back_to_deterministic_80_20(tmp_path: Path) -> None:
    entries = {
        f"images/img{index}.png": png_bytes((index * 5, 30, 60)) for index in range(20)
    }
    entries.update(
        {f"labels/img{index}.txt": b"1 0.5 0.5 0.2 0.2\n" for index in range(20)}
    )
    entries["data.yaml"] = b"names:\n  - cat\n  - dog\n"
    raw = materialize(tmp_path / "raw", entries)

    first_ctx, first, logs = _ctx(tmp_path, raw, seed=7)
    train_a, val_a, test_a = first._load_samples(first_ctx)
    second_ctx, second, _ = _ctx(tmp_path, raw, seed=7)
    train_b, val_b, _test_b = second._load_samples(second_ctx)

    assert len(train_a) == 16 and len(val_a) == 4
    assert val_a == val_b, "same seed must give the same split"
    assert train_a == train_b
    assert test_a == []
    assert any("deterministic 80/20" in line for line in logs)


def test_eval_report_has_val_and_test_sections() -> None:
    report = build_eval_report(
        ["cat", "dog"],
        {"mAP50": 0.9, "per_class_AP": {"cat": 0.95, "dog": 0.85}, "samples": 19},
        {"mAP50": 0.5, "per_class_AP": {"cat": 0.7, "dog": 0.3}, "samples": 10},
    )

    assert report["val"]["mAP50"] == 0.9
    assert report["test"]["mAP50"] == 0.5
    assert report["test"]["samples"] == 10
    assert report["test_reason"] is None
    # Backward compatibility: the top-level keys are the val numbers.
    assert report["mAP50"] == 0.9
    assert report["per_class_AP"] == {"cat": 0.95, "dog": 0.85}
    assert report["samples"] == 19
    assert report["task"] == "object_detection"


def test_eval_report_test_is_null_without_test_split() -> None:
    report = build_eval_report(
        ["cat"],
        {"mAP50": 0.42, "per_class_AP": {"cat": 0.42}, "samples": 5},
        None,
    )

    assert report["test"] is None
    assert report["test_reason"] == "no test split"
    assert report["mAP50"] == 0.42


def test_small_split_carries_warning() -> None:
    assert split_warning(19) is not None
    assert "19 images" in str(split_warning(19))
    assert split_warning(30) is None
    assert split_warning(500) is None


def test_evaluate_reloads_best_checkpoint(tmp_path: Path) -> None:
    """Evaluation must use the kept checkpoint, not the last epoch's weights."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True)
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    torch.save({"epoch": 3, "model_state": model.state_dict()}, checkpoints / "best.ckpt")

    logs: list[str] = []
    ctx = SimpleNamespace(
        output_dir=tmp_path / "artifacts",
        emit_log=lambda level, message: logs.append(f"{level} {message}"),
    )
    trainer = RTDetrTrainer()
    trainer._model = torch.nn.Linear(2, 2, bias=False)  # random init, must be replaced
    trainer._device = torch.device("cpu")

    trainer._load_best_weights(ctx)

    assert torch.allclose(trainer._model.weight, torch.eye(2))
    assert any("best weights" in line for line in logs)
    assert not trainer._model.training


def test_evaluate_without_best_checkpoint_keeps_current_weights(tmp_path: Path) -> None:
    trainer = RTDetrTrainer()
    trainer._model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        trainer._model.weight.copy_(torch.zeros(2, 2))
    logs: list[str] = []
    ctx = SimpleNamespace(
        output_dir=tmp_path / "artifacts",
        emit_log=lambda level, message: logs.append(f"{level} {message}"),
    )

    trainer._load_best_weights(ctx)

    assert torch.allclose(trainer._model.weight, torch.zeros(2, 2))
    assert any("final epoch weights" in line for line in logs)


def test_yolo_fixture_split_still_trains_only_on_train_rows(tmp_path: Path) -> None:
    """The shared fixture (train/val prefixes) keeps working after the split change."""
    raw = materialize(tmp_path / "raw", build_yolo(per_class=4, split=True))
    ctx, trainer, _logs = _ctx(tmp_path, raw, num_classes=2)

    train, val, test = trainer._load_samples(ctx)

    assert test == []
    assert train and val
    assert all("/val/" not in path.as_posix() for path, _ in train)


# --- B: matching counts (hand-computed known answer) ------------------------

# GT (normalized cxcywh, two classes):
#   A = class 0 at (0.25, 0.25), 0.2 x 0.2  -> xyxy 0.15,0.15,0.35,0.35
#   B = class 1 at (0.75, 0.75), 0.2 x 0.2  -> xyxy 0.65,0.65,0.85,0.85
#   C = class 1 at (0.25, 0.75), 0.2 x 0.2  -> xyxy 0.15,0.65,0.35,0.85
KNOWN_GT = [
    [0, 0.25, 0.25, 0.2, 0.2],
    [1, 0.75, 0.75, 0.2, 0.2],
    [1, 0.25, 0.75, 0.2, 0.2],
]


def test_match_counts_known_answer() -> None:
    # 1) exact class-0 box -> tp
    # 2) class 1's box predicted as class 0 -> wrong_class
    # 3) a box on empty background predicted as class 1 -> fp
    # C is never predicted -> fn
    pred = [
        [0, 0.9, 0.25, 0.25, 0.2, 0.2],
        [0, 0.6, 0.75, 0.75, 0.2, 0.2],
        [1, 0.5, 0.05, 0.05, 0.1, 0.1],
    ]

    counts = match_counts(KNOWN_GT, pred)

    assert counts == {"tp": 1, "fp": 1, "fn": 1, "wrong_class": 1}
    assert counts["fp"] + counts["fn"] + counts["wrong_class"] == 3  # error_score


def test_match_counts_perfect_predictions_have_no_errors() -> None:
    pred = [
        [0, 0.9, 0.25, 0.25, 0.2, 0.2],
        [1, 0.8, 0.75, 0.75, 0.2, 0.2],
        [1, 0.7, 0.25, 0.75, 0.2, 0.2],
    ]

    counts = match_counts(KNOWN_GT, pred)

    assert counts == {"tp": 3, "fp": 0, "fn": 0, "wrong_class": 0}


def test_match_counts_iou_threshold_is_half() -> None:
    # Shifted by 0.05 in x: intersection 0.15 x 0.2 = 0.03, union 0.05 -> IoU 0.6.
    just_above = match_counts(KNOWN_GT[:1], [[0, 0.9, 0.30, 0.25, 0.2, 0.2]])
    # Shifted by 0.08 in x: intersection 0.12 x 0.2 = 0.024, union 0.056 -> IoU 0.43.
    just_below = match_counts(KNOWN_GT[:1], [[0, 0.9, 0.33, 0.25, 0.2, 0.2]])

    assert just_above["tp"] == 1 and just_above["fp"] == 0
    assert just_below["tp"] == 0 and just_below["fp"] == 1 and just_below["fn"] == 1


def test_match_counts_empty_predictions_report_misses() -> None:
    assert match_counts(KNOWN_GT, []) == {"tp": 0, "fp": 0, "fn": 3, "wrong_class": 0}
    assert match_counts([], [[0, 0.9, 0.1, 0.1, 0.1, 0.1]]) == {
        "tp": 0,
        "fp": 1,
        "fn": 0,
        "wrong_class": 0,
    }


def test_predictions_payload_shape() -> None:
    image = {
        "split": "val",
        "path": "val/images/img0.png",
        "width": 640,
        "height": 480,
        "gt": [[0, 0.5, 0.5, 0.2, 0.2]],
        "pred": [[0, 0.87, 0.5, 0.5, 0.2, 0.2]],
        "counts": {"tp": 1, "fp": 0, "fn": 0, "wrong_class": 0},
        "error_score": 0,
    }

    payload = build_predictions_payload(["cat", "dog"], 0.25, [image], ["val", "test"])

    assert payload["class_names"] == ["cat", "dog"]
    assert payload["threshold"] == 0.25
    assert payload["match_iou"] == 0.5
    assert payload["splits"] == {"val": 1, "test": 0}
    entry = payload["images"][0]
    assert set(entry) == {"split", "path", "width", "height", "gt", "pred", "counts", "error_score"}
    assert len(entry["pred"][0]) == 6  # class_id, score, cx, cy, w, h
    assert len(entry["gt"][0]) == 5  # class_id, cx, cy, w, h


# --- B2: safe image serving -------------------------------------------------

RUN_ID = "run_20261101_120000_abcd"
DATASET_HASH = "a" * 64
IMAGES_PATH = "val/images/img0.png"


def _served_run(home: Path, *, listed: str = IMAGES_PATH) -> Path:
    """A finished run with predictions.json + one real dataset image."""
    run_dir = home / "runs" / RUN_ID
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"id": RUN_ID, "state": "completed", "dataset_hash": DATASET_HASH})
    )
    raw = home / "datasets" / DATASET_HASH / "raw"
    image = raw / IMAGES_PATH
    image.parent.mkdir(parents=True)
    image.write_bytes(png_bytes((1, 2, 3)))
    (raw / "unlisted.png").write_bytes(png_bytes((9, 9, 9)))
    (run_dir / "artifacts" / "predictions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_names": ["cat"],
                "threshold": 0.25,
                "images": [
                    {
                        "split": "val",
                        "path": listed,
                        "width": 16,
                        "height": 16,
                        "gt": [[0, 0.5, 0.5, 0.2, 0.2]],
                        "pred": [[0, 0.8, 0.5, 0.5, 0.2, 0.2]],
                        "counts": {"tp": 1, "fp": 0, "fn": 0, "wrong_class": 0},
                        "error_score": 0,
                    }
                ],
            }
        )
    )
    return raw


def test_predictions_endpoint_serves_the_file(client: TestClient, home: Path) -> None:
    _served_run(home)

    body = client.get(f"/api/v1/runs/{RUN_ID}/predictions").json()

    assert body["class_names"] == ["cat"]
    assert body["images"][0]["path"] == IMAGES_PATH
    assert body["images"][0]["counts"]["tp"] == 1


def test_predictions_endpoint_404_without_file(client: TestClient, home: Path) -> None:
    run_dir = home / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"id": RUN_ID, "state": "completed"}))

    response = client.get(f"/api/v1/runs/{RUN_ID}/predictions")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PREDICTIONS_NOT_FOUND"


def test_images_endpoint_serves_a_listed_image(client: TestClient, home: Path) -> None:
    _served_run(home)

    response = client.get(f"/api/v1/runs/{RUN_ID}/images", params={"path": IMAGES_PATH})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_images_endpoint_blocks_traversal(client: TestClient, home: Path) -> None:
    _served_run(home)

    for attempt in (
        "../../../../etc/passwd",
        "val/../../../../etc/passwd",
        "/etc/passwd",
        "val/images/../../../../etc/hostname",
    ):
        response = client.get(f"/api/v1/runs/{RUN_ID}/images", params={"path": attempt})
        assert response.status_code == 400, (attempt, response.status_code)
        assert response.json()["error"]["code"] == "INVALID_IMAGE_PATH"


def test_images_endpoint_refuses_unlisted_files(client: TestClient, home: Path) -> None:
    _served_run(home)

    response = client.get(
        f"/api/v1/runs/{RUN_ID}/images", params={"path": "unlisted.png"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "IMAGE_NOT_FOUND"


def test_images_endpoint_404_when_listed_file_is_gone(client: TestClient, home: Path) -> None:
    raw = _served_run(home, listed="val/images/missing.png")
    assert not (raw / "val/images/missing.png").exists()

    response = client.get(
        f"/api/v1/runs/{RUN_ID}/images", params={"path": "val/images/missing.png"}
    )

    assert response.status_code == 404


def test_images_endpoint_requires_a_path(client: TestClient, home: Path) -> None:
    _served_run(home)

    response = client.get(f"/api/v1/runs/{RUN_ID}/images")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE_PATH"


def test_prediction_images_are_token_aware(tmp_path: Path) -> None:
    from app.core.config import Settings
    from app.main import create_app

    settings = Settings(home=tmp_path / "home", access_token="shared-secret", log_level="DEBUG")
    with TestClient(create_app(settings)) as token_client:
        _served_run(settings.home)
        url = f"/api/v1/runs/{RUN_ID}/images"
        assert token_client.get(url, params={"path": IMAGES_PATH}).status_code == 401
        assert token_client.get(f"/api/v1/runs/{RUN_ID}/predictions").status_code == 401
        authorized = {"X-Access-Token": "shared-secret"}
        served = token_client.get(url, params={"path": IMAGES_PATH}, headers=authorized)
        assert served.status_code == 200


# --- B: slow end-to-end (CPU dev preset) -----------------------------------

@pytest.mark.slow
def test_dev_preset_run_writes_predictions(tmp_path: Path) -> None:
    """A real (tiny, CPU) train+evaluate pass produces predictions.json [B1]."""
    raw = materialize(tmp_path / "raw", _split_dataset(with_test=True))
    ctx, trainer, logs = _ctx(tmp_path, raw)
    ctx.config = {
        "epochs": 1,
        "batch_size": 2,
        "imgsz": 128,
        "lr": 1e-4,
        "freeze_backbone": True,
        "amp": False,
        "dev_preset": True,
        "score_threshold": 0.25,
    }
    ctx.facts = {"classes": ["cat", "dog"]}
    ctx.output_dir = tmp_path / "runs" / "run_x" / "artifacts"
    ctx.emit_metric = lambda *args, **kwargs: None

    trainer.prepare(ctx)
    trainer.train(ctx)
    evaluation = trainer.evaluate(ctx)

    predictions_path = ctx.output_dir / "predictions.json"
    assert predictions_path.is_file()
    payload = json.loads(predictions_path.read_text())
    assert payload["threshold"] == 0.25
    assert payload["class_names"] == ["cat", "dog"]
    # Both splits are represented: val (4 images) and test (3 images).
    assert payload["splits"] == {"val": 4, "test": 3}
    assert {image["split"] for image in payload["images"]} == {"val", "test"}
    for image in payload["images"]:
        assert image["path"].startswith(("val/", "test/"))
        assert len(image["gt"][0]) == 5
        assert image["counts"]["tp"] + image["counts"]["fn"] == len(image["gt"])
        assert image["error_score"] == (
            image["counts"]["fp"] + image["counts"]["fn"] + image["counts"]["wrong_class"]
        )
    # Same numbers reach the evaluation report for both splits.
    assert evaluation["val"]["samples"] == 4
    assert evaluation["test"]["samples"] == 3
    assert evaluation["test_reason"] is None
    assert any("predictions.json" in line for line in logs)
