"""RT-DETR r18vd detection plugin [doc 08 §1, §2].

Builds the vendored r18vd model, loads the official pretrained ``.pth``
(``torch.load(weights_only=True)``, ``["ema"]["module"]`` unwrap) with
classification-head adaptation, then trains on a YOLO-format dataset from
``manifest.parquet``. The loss is the exact official SetCriterion (vfl +
boxes with aux + denoising aux losses); metrics are ``train/loss`` and
``val/mAP50`` (manually computed, no extra dependency).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from PIL import Image

from app.storage.atomic import atomic_write_json
from app.training.base import Issue, RunContext, TrainerPlugin, TrainResult
from app.training.engine import EngineHooks, TrainEngine, load_best_checkpoint

# Head tensors whose shapes depend on the class count. ``denoising_class_embed``
# is (N+1, 256) (one extra background slot); the score heads are (N, 256)/(N,).
# This is the complete list of checkpoint keys containing 80/81 (verified
# 2026-10-22 against rtdetr_r18vd_dec3_6x_coco_from_paddle.pth) — nothing else
# must be touched during head adaptation.
HEAD_WEIGHT_KEYS = (
    "decoder.denoising_class_embed.weight",
    "decoder.enc_score_head.weight",
    "decoder.dec_score_head.0.weight",
    "decoder.dec_score_head.1.weight",
    "decoder.dec_score_head.2.weight",
)
HEAD_BIAS_KEYS = (
    "decoder.enc_score_head.bias",
    "decoder.dec_score_head.0.bias",
    "decoder.dec_score_head.1.bias",
    "decoder.dec_score_head.2.bias",
)
HEAD_KEYS = HEAD_WEIGHT_KEYS + HEAD_BIAS_KEYS

# Official r18vd_6x_coco criterion config (verified 2026-10-23).
WEIGHT_DICT = {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2}
LOSSES = ["vfl", "boxes"]
MATCHER_COSTS = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}

# Below this many images a split mAP is honest but noisy; the UI says so.
SMALL_VAL_THRESHOLD = 30
# Predictions kept in predictions.json (IoU 0.5 matching for the counts).
DEFAULT_SCORE_THRESHOLD = 0.25
MATCH_IOU = 0.5


def build_r18vd(
    num_classes: int, eval_spatial_size: list[int] | tuple[int, int] = (640, 640)
) -> torch.nn.Module:
    """r18vd model with the exact upstream config (configs/rtdetr/rtdetr_r18vd_6x_coco.yml).

    ``eval_spatial_size`` sizes the encoder's fixed positional embeddings, so it
    must match the input size used for inference/eval.
    """
    from app.vendor.rtdetr import RTDETR, HybridEncoder, PResNet, RTDETRTransformer

    spatial = list(eval_spatial_size)
    backbone = PResNet(  # type: ignore[no-untyped-call]
        depth=18,
        variant="d",
        return_idx=[1, 2, 3],
        num_stages=4,
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    )
    encoder = HybridEncoder(  # type: ignore[no-untyped-call]
        in_channels=[128, 256, 512],
        hidden_dim=256,
        expansion=0.5,
        feat_strides=[8, 16, 32],
        use_encoder_idx=[2],
        num_encoder_layers=1,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        pe_temperature=10000,
        depth_mult=1,
        act="silu",
        eval_spatial_size=spatial,
    )
    decoder = RTDETRTransformer(  # type: ignore[no-untyped-call]
        num_classes=num_classes,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_layers=3,
        num_denoising=100,
        eval_idx=-1,
        eval_spatial_size=spatial,
    )
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def _adapt_head(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Truncate/pad the checkpoint head to the target shape (first N classes)."""
    if tensor.shape == target.shape:
        return tensor
    out = torch.zeros_like(target)
    keep = min(tensor.shape[0], target.shape[0])
    out[:keep] = tensor[:keep]
    return out


def load_pretrained(model: torch.nn.Module, path: Path) -> int:
    """Load the official ``.pth`` and adapt the classification heads.

    Returns the number of adapted head tensors (must be ``len(HEAD_KEYS)``).
    Safe deserialization only: ``torch.load(..., weights_only=True)`` [doc 10 §4].
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"could not load pretrained checkpoint {path.name}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("ema"), dict):
        raise ValueError(
            f"pretrained checkpoint {path.name} is not an official RT-DETR checkpoint: "
            "expected {ema: {module: state_dict}}"
        )
    module = payload["ema"].get("module")
    if not isinstance(module, dict):
        raise ValueError(f"pretrained checkpoint {path.name} has no ema.module state dict")

    target = model.state_dict()
    adapted = 0
    for key, value in module.items():
        if key not in target:
            raise ValueError(f"checkpoint key {key!r} is not present in the model")
        if key in HEAD_KEYS:
            module[key] = _adapt_head(value, target[key])
            adapted += 1
        elif value.shape != target[key].shape:
            raise ValueError(
                f"non-head shape mismatch on {key!r}: checkpoint {tuple(value.shape)} "
                f"vs model {tuple(target[key].shape)}"
            )
    if adapted != len(HEAD_KEYS):
        raise ValueError(f"expected to adapt {len(HEAD_KEYS)} heads, adapted {adapted}")
    model.load_state_dict(module, strict=True)
    return adapted


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two xyxy boxes (pure Python, for mAP)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_map50(
    gt: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    pred: dict[int, list[tuple[int, float, tuple[float, float, float, float]]]],
) -> dict[str, Any]:
    """mAP@0.5, all-points interpolation (area under the PR curve).

    ``gt``: class -> [(image_id, xyxy)]; ``pred``: class -> [(image_id, score, xyxy)].
    A prediction is TP when its best IoU with an unmatched same-class GT in the
    same image is >= 0.5. Per-class AP uses the all-points integral; mAP is the
    mean over classes that have at least one GT box. Classes with no GT are
    excluded (standard). No dependency beyond stdlib.
    """
    aps: list[float] = []
    per_class: dict[int, float] = {}
    for class_id, gts in gt.items():
        if not gts:
            continue
        preds = sorted(pred.get(class_id, []), key=lambda item: -item[1])
        matched: set[int] = set()
        npos = len(gts)
        precisions: list[float] = []
        recalls: list[float] = []
        correct = 0
        for rank, (img, _score, box) in enumerate(preds):
            best_iou = 0.0
            best_j = -1
            for j, (gimg, gbox) in enumerate(gts):
                if gimg != img or j in matched:
                    continue
                iou = _iou(box, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.5:
                correct += 1
                matched.add(best_j)
            precisions.append(correct / (rank + 1))
            recalls.append(correct / npos)
        ap = 0.0
        prev_recall = 0.0
        for rank, recall in enumerate(recalls):
            ap += (recall - prev_recall) * max(precisions[rank:])
            prev_recall = recall
        aps.append(ap)
        per_class[class_id] = ap
    mean_ap = sum(aps) / len(aps) if aps else 0.0
    return {"mAP50": mean_ap, "per_class_AP": per_class, "num_classes_evaluated": len(aps)}


def _as_int(value: Any) -> int | None:
    """Config values may arrive as strings from an HTML form: coerce or fail."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return int(number) if number.is_integer() else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _image_size(path: Path) -> tuple[int, int]:
    """Real pixel size of an image (0, 0 when unreadable)."""
    try:
        with Image.open(path) as opened:
            return int(opened.width), int(opened.height)
    except OSError:
        return 0, 0


def _relative_to_root(path: Path, dataset_root: Path) -> str:
    """Image path relative to the dataset root, as the API expects it."""
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return path.name


def _cxcywh_to_xyxy(box: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    """Normalized cxcywh -> xyxy, ignoring any leading class id / score.

    GT rows are ``[class_id, cx, cy, w, h]`` and prediction rows are
    ``[class_id, score, cx, cy, w, h]``, so the box is always the last four.
    """
    cx, cy, w, h = (float(v) for v in box[-4:])
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def match_counts(
    gt: list[list[float]],
    pred: list[list[float]],
    iou_threshold: float = MATCH_IOU,
) -> dict[str, int]:
    """Count tp / fp / fn / wrong_class for one image (greedy, IoU 0.5, no deps).

    ``gt``: ``[class_id, cx, cy, w, h]`` normalized; ``pred``:
    ``[class_id, score, cx, cy, w, h]`` normalized. Predictions are visited by
    descending score; each one claims the best same-class unmatched GT with
    IoU >= ``iou_threshold`` (tp). A prediction left over that covers an
    unmatched GT of *another* class is ``wrong_class`` (a right box with the
    wrong label), everything else is fp. GT nobody claimed is fn. Every
    prediction lands in exactly one bucket (tp + fp + wrong_class == #pred) and
    every GT in exactly one (tp + fn == #gt), so error_score never double-counts.
    """
    gt_by_class: dict[int, list[list[float]]] = defaultdict(list)
    for box in gt:
        gt_by_class[int(box[0])].append(box)
    used: dict[int, set[int]] = defaultdict(set)

    tp = 0
    fp = 0
    wrong_class = 0
    leftover: list[list[float]] = []
    for prediction in sorted(pred, key=lambda item: -float(item[1])):
        class_id = int(prediction[0])
        pred_box = _cxcywh_to_xyxy(prediction)
        best_iou = 0.0
        best_index = -1
        for index, candidate in enumerate(gt_by_class.get(class_id, [])):
            if index in used[class_id]:
                continue
            iou = _iou(pred_box, _cxcywh_to_xyxy(candidate))
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index >= 0 and best_iou >= iou_threshold:
            used[class_id].add(best_index)
            tp += 1
        else:
            leftover.append(prediction)

    for prediction in leftover:
        class_id = int(prediction[0])
        pred_box = _cxcywh_to_xyxy(prediction)
        best_iou = 0.0
        best_class = -1
        best_index = -1
        for other_class, candidates in gt_by_class.items():
            if other_class == class_id:
                continue
            for index, candidate in enumerate(candidates):
                if index in used[other_class]:
                    continue
                iou = _iou(pred_box, _cxcywh_to_xyxy(candidate))
                if iou > best_iou:
                    best_iou, best_class, best_index = iou, other_class, index
        if best_class >= 0 and best_iou >= iou_threshold:
            used[best_class].add(best_index)
            wrong_class += 1
        else:
            fp += 1

    fn = sum(len(boxes) - len(used[class_id]) for class_id, boxes in gt_by_class.items())
    return {"tp": tp, "fp": fp, "fn": fn, "wrong_class": wrong_class}


def seed_everything(seed: int) -> None:
    """Make a run reproducible: python ``random``, numpy and torch.

    Weight init, dropout and the flip augmentation all draw from these RNGs.
    Without this, two runs with the same config and seed produced different
    losses (reproduced 2026-11-01: 36.829063 vs 36.801177 at epoch 1).
    GPU kernels are not bit-exact: determinism is asserted on CPU only.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in test env
        torch.cuda.manual_seed_all(seed)


def _load_image(path: Path, imgsz: int, flip: bool) -> torch.Tensor:
    """RGB, resize to imgsz, [0,1] float, optional horizontal flip (train)."""
    with Image.open(path) as opened:
        rgb = opened.convert("RGB").resize((imgsz, imgsz))
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if flip:
        tensor = torch.flip(tensor, dims=[2])
    return tensor


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def split_warning(sample_count: int) -> str | None:
    """Plain-words caveat for a split too small to give a stable mAP."""
    if sample_count >= SMALL_VAL_THRESHOLD:
        return None
    return f"only {sample_count} images: mAP50 will swing a lot between epochs and runs"


def build_predictions_payload(
    class_names: list[str],
    threshold: float,
    images: list[dict[str, Any]],
    split_names: list[str],
) -> dict[str, Any]:
    """predictions.json payload [B1]: per image boxes + error counts.

    ``images`` entries are already normalized and sorted; ``splits`` records how
    many images each split contributed so the UI can offer a split filter.
    """
    return {
        "schema_version": 1,
        "model_family": "rtdetr",
        "class_names": list(class_names),
        "threshold": threshold,
        "match_iou": MATCH_IOU,
        "splits": {
            name: sum(1 for image in images if image.get("split") == name) for name in split_names
        },
        "images": images,
    }


def build_eval_report(
    class_names: list[str],
    val: dict[str, Any],
    test: dict[str, Any] | None,
) -> dict[str, Any]:
    """evaluation.json payload: per-split numbers + backward-compatible val keys.

    ``val`` picks the best checkpoint, so it is optimistic; ``test`` is unseen.
    The top-level ``mAP50``/``per_class_AP``/``samples`` stay the val numbers so
    pre-test-split clients keep reading the same fields [doc 08 §6].
    """
    return {
        "task": "object_detection",
        "class_names": list(class_names),
        "val": dict(val),
        "test": dict(test) if test is not None else None,
        "test_reason": None if test is not None else "no test split",
        "mAP50": val["mAP50"],
        "per_class_AP": dict(val["per_class_AP"]),
        "samples": val["samples"],
    }


class RTDetrTrainer(TrainerPlugin):
    family = "rtdetr"
    supported_tasks = ["object_detection"]
    supported_formats = ["yolo"]
    description = (
        "RT-DETR r18vd object detector (needs a GPU in full mode; dev_preset runs on CPU)."
    )
    needs_gpu = True
    hparam_schema = {
        "type": "object",
        "properties": {
            "epochs": {"type": "integer", "minimum": 1, "default": 1},
            "batch_size": {"type": "integer", "minimum": 1, "default": 1},
            "imgsz": {"type": "integer", "minimum": 128, "maximum": 1536, "default": 512},
            "lr": {"type": "number", "minimum": 1e-6, "default": 0.0001},
            "amp": {"type": "boolean", "default": False},
            "freeze_backbone": {"type": "boolean", "default": True},
            "clip_max_norm": {"type": "number", "minimum": 0, "default": 0.1},
            "score_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.25,
                "description": "Predictions below this confidence are dropped from "
                "artifacts/predictions.json (mAP itself always uses all queries).",
            },
            "dev_preset": {
                "type": "boolean",
                "default": True,
                "description": "Tiny CPU preset (1 epoch); disable for full GPU mode.",
            },
        },
    }
    default_hparams = {
        "epochs": 1,
        "batch_size": 1,
        "imgsz": 512,
        "lr": 0.0001,
        "amp": False,
        "freeze_backbone": True,
        "clip_max_norm": 0.1,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "dev_preset": True,
    }

    def __init__(self) -> None:
        # Seeded in prepare(); a fixed default keeps direct helper use (tests,
        # predictions) deterministic too.
        self._rng = random.Random(0)

    # -- plugin interface ---------------------------------------------------

    def validate_config(self, config: dict[str, Any], facts: dict[str, Any]) -> list[Issue]:
        del facts
        issues: list[Issue] = []
        epochs = _as_int(config.get("epochs", 1))
        if epochs is None:
            issues.append(Issue("epochs", "must be an integer"))
        elif epochs < 1:
            issues.append(Issue("epochs", "must be at least 1"))
        batch_size = _as_int(config.get("batch_size", 1))
        if batch_size is None:
            issues.append(Issue("batch_size", "must be an integer"))
        elif batch_size < 1:
            issues.append(Issue("batch_size", "must be at least 1"))
        imgsz = _as_int(config.get("imgsz", 512))
        if imgsz is None:
            issues.append(Issue("imgsz", "must be an integer"))
        elif not 128 <= imgsz <= 1536:
            issues.append(Issue("imgsz", "must be between 128 and 1536"))
        lr = _as_float(config.get("lr", 0.0001))
        if lr is None:
            issues.append(Issue("lr", "must be a number"))
        elif lr <= 0:
            issues.append(Issue("lr", "must be positive"))
        threshold = _as_float(config.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
        if threshold is None:
            issues.append(Issue("score_threshold", "must be a number"))
        elif not 0.0 <= threshold <= 1.0:
            issues.append(Issue("score_threshold", "must be between 0 and 1"))
        return issues

    def prepare(self, ctx: RunContext) -> None:
        from app.vendor.rtdetr import HungarianMatcher, RTDETRPostProcessor, SetCriterion

        # The verification report stores the sorted class-name list under
        # "classes" (doc 06 §6); "class_names" is a legacy alias.
        classes = list(ctx.facts.get("classes") or ctx.facts.get("class_names") or [])
        if not classes:
            raise ValueError("detection needs class names from verification (classes empty)")
        self._num_classes = len(classes)
        self._class_names = classes
        self._imgsz = int(ctx.config.get("imgsz", 512))
        self._batch_size = int(ctx.config.get("batch_size", 1))
        self._epochs = int(ctx.config.get("epochs", 1))
        self._lr = float(ctx.config.get("lr", 0.0001))
        self._clip = float(ctx.config.get("clip_max_norm", 0.1))
        self._amp = bool(ctx.config.get("amp", False))
        self._device = torch.device("cuda" if ctx.device.is_gpu else "cpu")
        # Reproducibility: same seed + same config -> same numbers [S1].
        self._seed = int(getattr(ctx, "seed", 42) or 42)
        seed_everything(self._seed)
        self._rng = random.Random(self._seed)

        self._train_samples, self._val_samples, self._test_samples = self._load_samples(ctx)
        if not self._train_samples:
            raise ValueError("no labelled detection boxes in the manifest (train split empty)")
        if len(self._val_samples) < SMALL_VAL_THRESHOLD:
            ctx.emit_log(
                "WARNING",
                f"val split has only {len(self._val_samples)} images "
                f"(< {SMALL_VAL_THRESHOLD}): mAP50 will swing a lot between epochs and runs; "
                "the number is honest but not precise",
            )

        self._model = build_r18vd(self._num_classes, eval_spatial_size=(self._imgsz, self._imgsz))
        freeze_backbone = bool(ctx.config.get("freeze_backbone", True))
        checkpoint = getattr(ctx, "checkpoint_path", None)
        if freeze_backbone:
            if checkpoint is None:
                ctx.emit_log(
                    "WARNING",
                    "freeze_backbone=True but no pretrained checkpoint was loaded: a frozen "
                    "random backbone cannot learn — load pretrained weights or disable "
                    "freeze_backbone",
                )
            for name, parameter in self._model.named_parameters():
                if name.startswith("backbone"):
                    parameter.requires_grad_(False)

        if checkpoint is not None:
            load_pretrained(self._model, Path(checkpoint))
        self._model.to(self._device)

        backbone = [
            parameter
            for name, parameter in self._model.named_parameters()
            if name.startswith("backbone") and parameter.requires_grad
        ]
        head = [
            parameter
            for name, parameter in self._model.named_parameters()
            if not name.startswith("backbone") and parameter.requires_grad
        ]
        # Official optimizer: backbone lr 1e-5, head lr 1e-4, wd 1e-4.
        groups: list[dict[str, Any]] = []
        if backbone:
            groups.append({"params": backbone, "lr": self._lr * 0.1})
        if head:
            groups.append({"params": head, "lr": self._lr})
        self._optimizer = torch.optim.AdamW(
            groups, lr=self._lr, betas=(0.9, 0.999), weight_decay=1e-4
        )
        self._matcher = HungarianMatcher(  # type: ignore[no-untyped-call]
            weight_dict=MATCHER_COSTS, use_focal_loss=True, alpha=0.25, gamma=2.0
        )
        self._criterion = SetCriterion(  # type: ignore[no-untyped-call]
            self._matcher,
            weight_dict=WEIGHT_DICT,
            losses=LOSSES,
            alpha=0.75,
            gamma=2.0,
            num_classes=self._num_classes,
        )
        self._postprocessor = RTDETRPostProcessor(
            num_classes=self._num_classes, use_focal_loss=True, num_top_queries=300
        )
        self._engine = TrainEngine(
            hooks=EngineHooks(
                batches=lambda: _batches(self._train_samples, self._batch_size),
                train_step=self._train_batch,
                evaluate=self._evaluate_map,
                set_train_mode=self._set_train_mode,
                model_state=self._model.state_dict,
                load_model_state=self._model.load_state_dict,
                optimizer_state=self._optimizer.state_dict,
                load_optimizer_state=self._optimizer.load_state_dict,
                export_best=self._export_best,
            ),
            epochs=self._epochs,
        )
        freeze = bool(ctx.config.get("freeze_backbone", True))
        frozen = "backbone frozen" if freeze else "full model"
        test_note = (
            f"{len(self._test_samples)} held-out test"
            if self._test_samples
            else "no test split (final test mAP will be null)"
        )
        ctx.emit_log(
            "INFO",
            f"rtdetr r18vd prepared: {len(self._train_samples)} train / "
            f"{len(self._val_samples)} val / {test_note} images, {self._num_classes} classes, "
            f"{frozen}, imgsz {self._imgsz}, amp {self._amp}",
        )

    def train(self, ctx: RunContext) -> TrainResult:
        return self._engine.run(ctx)

    def evaluate(self, ctx: RunContext) -> dict[str, Any]:
        """Final detection metrics from the best weights: val + held-out test [doc 08 §6]."""
        self._load_best_weights(ctx)
        names = list(getattr(self, "_class_names", []))
        val = self._eval_split(self._val_samples, names)
        test = self._eval_split(self._test_samples, names) if self._test_samples else None
        result = build_eval_report(names, val, test)
        try:
            self._write_predictions(ctx, names)
        except Exception as exc:  # the metrics are already computed: never lose them
            ctx.emit_log(
                "WARNING",
                f"could not write predictions.json ({type(exc).__name__}: {exc}); "
                "the Predictions tab will be empty for this run",
            )
        test_note = (
            f"test/mAP50={test['mAP50']:.4f} on {test['samples']} images"
            if test is not None
            else "test=mAP50 not measured (no test split)"
        )
        ctx.emit_log(
            "INFO",
            f"evaluation (best weights): val/mAP50={val['mAP50']:.4f} on "
            f"{val['samples']} images, {test_note}",
        )
        return result

    def resume(self, ctx: RunContext, checkpoint_path: Path) -> None:
        self._engine.resume(ctx, checkpoint_path, self._device)

    # -- internals -----------------------------------------------------------

    def _load_samples(self, ctx: RunContext) -> tuple[list[Any], list[Any], list[Any]]:
        """Read manifest.parquet; boxes are [class_id, cx, cy, w, h] normalized.

        Returns ``(train, val, test)``. ``test`` is held out: never trained on
        and never used to pick the best checkpoint [doc 06 §3]. ``unspecified``
        rows are treated as train; a dataset with no ``val`` rows is split 80/20
        here, shuffled with ``ctx.seed`` so the result is deterministic.
        """
        table = pq.read_table(ctx.manifest_path)
        paths = table.column("path").to_pylist()
        splits = table.column("split").to_pylist()
        boxes_col = table.column("boxes").to_pylist()
        train: list[tuple[Path, list[list[float]]]] = []
        val: list[tuple[Path, list[list[float]]]] = []
        test: list[tuple[Path, list[list[float]]]] = []
        dropped = 0
        for path, split, boxes_raw in zip(paths, splits, boxes_col, strict=False):
            if split not in ("train", "val", "test", "unspecified"):
                continue  # unknown split label: do not guess
            full = ctx.dataset_root / path
            if not full.is_file():
                continue
            boxes: list[list[float]] = []
            if boxes_raw:
                try:
                    for box in json.loads(boxes_raw):
                        if len(box) != 5:
                            continue
                        cid = box[0]
                        coords = [float(v) for v in box[1:]]
                        if not (isinstance(cid, int) and 0 <= cid < self._num_classes):
                            continue
                        if not all(0.0 <= v <= 1.0 for v in coords):
                            continue
                        boxes.append([cid, coords[0], coords[1], coords[2], coords[3]])
                except (ValueError, TypeError):
                    continue
            if not boxes:
                dropped += 1
                continue
            if split == "val":
                val.append((full, boxes))
            elif split == "test":
                test.append((full, boxes))
            else:  # train / unspecified -> train by default
                train.append((full, boxes))
        if not train and not val and not test:
            return [], [], []
        if not val and len(train) > 1:
            # Deterministic 80/20 fallback: same dataset + seed -> same split.
            shuffled = list(train)
            random.Random(ctx.seed).shuffle(shuffled)
            boundary = min(len(shuffled) - 1, max(1, round(0.8 * len(shuffled))))
            train, val = shuffled[:boundary], shuffled[boundary:]
            ctx.emit_log(
                "INFO",
                f"no val rows in the manifest: {len(train)} train / {len(val)} val "
                f"(deterministic 80/20, seed {ctx.seed})",
            )
        if dropped:
            ctx.emit_log("INFO", f"dropped {dropped} images with no valid boxes")
        return train, val, test

    def _collate(
        self, batch: list[tuple[Path, list[list[float]]]]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        images: list[torch.Tensor] = []
        targets: list[dict[str, torch.Tensor]] = []
        for _path, boxes in batch:
            flip = self._model.training and self._rng.random() < 0.5
            image = _load_image(_path, self._imgsz, flip)
            images.append(image)
            labels: list[int] = []
            coords: list[list[float]] = []
            for cid, cx, cy, w, h in boxes:
                labels.append(int(cid))
                flipped_cx = 1.0 - float(cx) if flip else float(cx)
                coords.append([flipped_cx, float(cy), float(w), float(h)])
            targets.append(
                {
                    "labels": torch.tensor(labels, dtype=torch.int64, device=self._device),
                    "boxes": torch.tensor(coords, dtype=torch.float32, device=self._device),
                }
            )
        return torch.stack(images).to(self._device), targets

    def _train_batch(self, batch: list[tuple[Path, list[list[float]]]]) -> float:
        images, targets = self._collate(batch)
        enabled = self._amp and self._device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.float16, enabled=enabled):
            outputs = self._model(images, targets)
            losses = self._criterion(outputs, targets)
            loss = sum(losses.values())
        loss.backward()
        if self._clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        return float(loss.item())

    def _eval_map_dict(
        self,
        samples: list[tuple[Path, list[list[float]]]] | None = None,
    ) -> dict[str, Any]:
        """Run the postprocessor over ``samples`` (default: val) and compute mAP@0.5."""
        if samples is None:
            samples = self._val_samples
        self._model.eval()
        gt: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
        pred: dict[int, list[tuple[int, float, tuple[float, float, float, float]]]] = defaultdict(
            list
        )
        image_id = 0
        enabled = self._amp and self._device.type == "cuda"
        with torch.no_grad():
            for batch in _batches(samples, self._batch_size):
                images, targets = self._collate(batch)
                with torch.autocast("cuda", dtype=torch.float16, enabled=enabled):
                    outputs = self._model(images)
                orig_sizes = torch.full(
                    (len(batch), 2), self._imgsz, dtype=torch.float32, device=self._device
                )
                results = self._postprocessor(outputs, orig_sizes)
                for result, target in zip(results, targets, strict=False):
                    labels = result["labels"].tolist()
                    scores = result["scores"].tolist()
                    boxes = result["boxes"].tolist()
                    for label, score, box in zip(labels, scores, boxes, strict=False):
                        pred[int(label)].append(
                            (
                                image_id,
                                float(score),
                                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                            )
                        )
                    for label, box in zip(
                        target["labels"].tolist(), target["boxes"].tolist(), strict=False
                    ):
                        cx, cy, w, h = box
                        xyxy = (
                            (cx - w / 2) * self._imgsz,
                            (cy - h / 2) * self._imgsz,
                            (cx + w / 2) * self._imgsz,
                            (cy + h / 2) * self._imgsz,
                        )
                        gt[int(label)].append((image_id, xyxy))
                    image_id += 1
        return compute_map50(gt, pred)

    def _evaluate_map(self) -> float:
        return float(self._eval_map_dict()["mAP50"])

    def _eval_split(
        self, samples: list[tuple[Path, list[list[float]]]], names: list[str]
    ) -> dict[str, Any]:
        """mAP50 + per-class AP (keyed by class name) for one split."""
        raw = self._eval_map_dict(samples)
        metrics: dict[str, Any] = {
            "mAP50": round(float(raw["mAP50"]), 4),
            "per_class_AP": {
                names[cid]: round(float(ap), 4)
                for cid, ap in raw["per_class_AP"].items()
                if cid < len(names)
            },
            "samples": len(samples),
        }
        if len(samples) < SMALL_VAL_THRESHOLD:
            metrics["warning"] = split_warning(len(samples))
        return metrics

    def _write_predictions(self, ctx: RunContext, names: list[str]) -> None:
        """Per-image GT/prediction boxes for the UI gallery -> artifacts/predictions.json.

        Runs on the best weights (``evaluate`` already reloaded them). Boxes are
        normalized to the image, so the UI can draw them over the original pixels:
        the network sees the whole image stretched to ``imgsz`` (no padding), so
        its pixel boxes divided by ``imgsz`` are the same normalized space as the
        manifest GT. No flip happens here — ``_collate`` only flips in train mode.
        """
        threshold = float(ctx.config.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
        splits: list[tuple[str, list[tuple[Path, list[list[float]]]]]] = [
            ("val", self._val_samples)
        ]
        if self._test_samples:
            splits.append(("test", self._test_samples))
        images: list[dict[str, Any]] = []
        for split_name, samples in splits:
            images.extend(self._predict_split(ctx, split_name, samples, threshold))
        payload = build_predictions_payload(names, threshold, images, [name for name, _ in splits])
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(ctx.output_dir / "predictions.json", payload)
        total_errors = sum(int(image["error_score"]) for image in images)
        ctx.emit_log(
            "INFO",
            f"predictions.json: {len(images)} images (score >= {threshold}), "
            f"{total_errors} errors to inspect",
        )

    def _predict_split(
        self,
        ctx: RunContext,
        split_name: str,
        samples: list[tuple[Path, list[list[float]]]],
        threshold: float,
    ) -> list[dict[str, Any]]:
        self._model.eval()
        records: list[dict[str, Any]] = []
        enabled = self._amp and self._device.type == "cuda"
        with torch.no_grad():
            for batch in _batches(samples, self._batch_size):
                images, _targets = self._collate(batch)
                with torch.autocast("cuda", dtype=torch.float16, enabled=enabled):
                    outputs = self._model(images)
                sizes = torch.full(
                    (len(batch), 2), self._imgsz, dtype=torch.float32, device=self._device
                )
                results = self._postprocessor(outputs, sizes)
                for (path, boxes), result in zip(batch, results, strict=False):
                    gt = [[float(box[0]), *[float(v) for v in box[1:5]]] for box in boxes]
                    predictions: list[list[float]] = []
                    for label, score, raw in zip(
                        result["labels"].tolist(),
                        result["scores"].tolist(),
                        result["boxes"].tolist(),
                        strict=False,
                    ):
                        if float(score) < threshold:
                            continue
                        x1, y1, x2, y2 = (float(v) / self._imgsz for v in raw)
                        predictions.append(
                            [
                                float(int(label)),
                                round(float(score), 4),
                                round(_clamp01((x1 + x2) / 2), 4),
                                round(_clamp01((y1 + y2) / 2), 4),
                                round(_clamp01(x2 - x1), 4),
                                round(_clamp01(y2 - y1), 4),
                            ]
                        )
                    counts = match_counts(gt, predictions)
                    width, height = _image_size(path)
                    records.append(
                        {
                            "split": split_name,
                            "path": _relative_to_root(path, ctx.dataset_root),
                            "width": width,
                            "height": height,
                            "gt": [[int(box[0]), *box[1:5]] for box in gt],
                            "pred": predictions,
                            "counts": counts,
                            "error_score": counts["fp"] + counts["fn"] + counts["wrong_class"],
                        }
                    )
        records.sort(key=lambda item: (-int(item["error_score"]), str(item["path"])))
        return records

    def _load_best_weights(self, ctx: RunContext) -> None:
        """Evaluate the checkpoint that was actually kept, not the last epoch."""
        device = getattr(self, "_device", "cpu")
        if load_best_checkpoint(ctx, device, self._model.load_state_dict) is not None:
            self._model.eval()

    def _set_train_mode(self) -> None:
        self._model.train()

    def _export_best(self, checkpoints: Path) -> None:
        # Ultralytics-compatible copy of the best weights (no optimizer state)
        # so the export can be loaded by the upstream tooling [doc 08 §6].
        torch.save({"ema": {"module": self._model.state_dict()}}, checkpoints / "weights.pth")
