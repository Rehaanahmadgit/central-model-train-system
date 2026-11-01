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
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from PIL import Image

from app.training.base import Issue, RunContext, TrainerPlugin, TrainResult

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
        "dev_preset": True,
    }

    # -- plugin interface ---------------------------------------------------

    def validate_config(self, config: dict[str, Any], facts: dict[str, Any]) -> list[Issue]:
        del facts
        issues: list[Issue] = []
        if int(config.get("epochs", 1)) < 1:
            issues.append(Issue("epochs", "must be at least 1"))
        if int(config.get("batch_size", 1)) < 1:
            issues.append(Issue("batch_size", "must be at least 1"))
        if not 128 <= int(config.get("imgsz", 512)) <= 1536:
            issues.append(Issue("imgsz", "must be between 128 and 1536"))
        if float(config.get("lr", 0.0001)) <= 0:
            issues.append(Issue("lr", "must be positive"))
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

        self._train_samples, self._val_samples = self._load_samples(ctx)
        if not self._train_samples:
            raise ValueError("no labelled detection boxes in the manifest (train split empty)")

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
        self._start_epoch = 1
        self._best_map = -1.0
        self._best_epoch = 0
        freeze = bool(ctx.config.get("freeze_backbone", True))
        frozen = "backbone frozen" if freeze else "full model"
        ctx.emit_log(
            "INFO",
            f"rtdetr r18vd prepared: {len(self._train_samples)} train / "
            f"{len(self._val_samples)} val images, {self._num_classes} classes, "
            f"{frozen}, imgsz {self._imgsz}, amp {self._amp}",
        )

    def train(self, ctx: RunContext) -> TrainResult:
        epoch = self._start_epoch - 1
        for epoch in range(self._start_epoch, self._epochs + 1):
            if ctx.should_stop():
                break
            self._model.train()
            running_loss = 0.0
            seen = 0
            for batch in _batches(self._train_samples, self._batch_size):
                if ctx.should_pause():
                    ctx.wait_if_paused()
                if ctx.should_stop():
                    break
                loss = self._train_batch(batch)
                if math.isnan(loss) or math.isinf(loss):
                    raise ValueError(f"nan/inf loss at epoch {epoch}")
                running_loss += loss
                seen += 1
            if ctx.should_stop():
                break
            # Epoch-mean train loss, one point per epoch (step == epoch) [doc 08 §6].
            train_loss = running_loss / max(seen, 1)
            ctx.emit_metric("train/loss", float(train_loss), epoch, epoch, "train")
            val_map = self._evaluate_map()
            if math.isnan(val_map):
                raise ValueError(f"nan validation mAP at epoch {epoch}")
            ctx.emit_metric("val/mAP50", val_map, epoch, epoch, "val")
            improved = val_map > self._best_map
            ctx.emit_log(
                "INFO",
                f"epoch {epoch}/{self._epochs} train/loss={train_loss:.4f} "
                f"val/mAP50={val_map:.4f}",
            )
            self._save(ctx, epoch, is_best=improved)
            if improved:
                self._best_map = val_map
                self._best_epoch = epoch
        return TrainResult(
            best={
                "metric": "val/mAP50",
                "value": round(max(self._best_map, 0.0), 4),
                "epoch": self._best_epoch,
            },
            epochs=epoch,
            note=f"val/mAP50={max(self._best_map, 0.0):.4f}",
        )

    def evaluate(self, ctx: RunContext) -> dict[str, Any]:
        """Final detection metrics: mAP50 + per-class AP [doc 05 §5, doc 08 §6]."""
        result = self._eval_map_dict()
        names = getattr(self, "_class_names", [])
        result["class_names"] = names
        result["per_class_AP"] = {
            names[cid]: round(ap, 4)
            for cid, ap in result["per_class_AP"].items()
            if cid < len(names)
        }
        result["samples"] = len(self._val_samples)
        return result

    def resume(self, ctx: RunContext, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        if isinstance(payload, dict) and "model_state" in payload:
            self._model.load_state_dict(payload["model_state"])
            self._optimizer.load_state_dict(payload["optimizer_state"])
            self._start_epoch = int(payload.get("epoch", 0)) + 1
            self._best_map = float(payload.get("best_map", -1.0))
            self._best_epoch = int(payload.get("best_epoch", 0))
            ctx.emit_log(
                "INFO", f"resumed from {checkpoint_path.name} at epoch {self._start_epoch}"
            )
        else:
            self._model.load_state_dict(payload)
            ctx.emit_log("INFO", f"loaded weights from {checkpoint_path.name} (no optimizer state)")

    # -- internals -----------------------------------------------------------

    def _load_samples(self, ctx: RunContext) -> tuple[list[Any], list[Any]]:
        """Read manifest.parquet; boxes are [class_id, cx, cy, w, h] normalized."""
        table = pq.read_table(ctx.manifest_path)
        paths = table.column("path").to_pylist()
        splits = table.column("split").to_pylist()
        boxes_col = table.column("boxes").to_pylist()
        train: list[tuple[Path, list[list[float]]]] = []
        val: list[tuple[Path, list[list[float]]]] = []
        dropped = 0
        for path, split, boxes_raw in zip(paths, splits, boxes_col, strict=False):
            if split not in ("train", "val", "unspecified"):
                continue  # "test" is held out and never trained on [doc 06 §3]
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
            else:  # train / unspecified -> train by default
                train.append((full, boxes))
        if not train and not val:
            return [], []
        if not val:
            boundary = max(1, int(0.8 * len(train)))
            val = train[boundary:]
            train = train[:boundary]
        if dropped:
            ctx.emit_log("INFO", f"dropped {dropped} images with no valid boxes")
        return train, val

    def _collate(
        self, batch: list[tuple[Path, list[list[float]]]]
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        images: list[torch.Tensor] = []
        targets: list[dict[str, torch.Tensor]] = []
        for _path, boxes in batch:
            flip = self._model.training and random.random() < 0.5
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

    def _eval_map_dict(self) -> dict[str, Any]:
        """Run the postprocessor over the val split and compute mAP@0.5."""
        self._model.eval()
        gt: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
        pred: dict[int, list[tuple[int, float, tuple[float, float, float, float]]]] = defaultdict(
            list
        )
        image_id = 0
        enabled = self._amp and self._device.type == "cuda"
        with torch.no_grad():
            for batch in _batches(self._val_samples, self._batch_size):
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

    def _save(self, ctx: RunContext, epoch: int, *, is_best: bool) -> None:
        checkpoints = ctx.output_dir.parent / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch,
            "model_state": self._model.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "best_map": self._best_map,
            "best_epoch": self._best_epoch,
        }
        last = checkpoints / "last.ckpt"
        torch.save(payload, last)
        ctx.save_checkpoint(last, is_best=False, metrics={"epoch": epoch})
        if is_best:
            best = checkpoints / "best.ckpt"
            torch.save(payload, best)
            # Ultralytics-compatible copy of the best weights (no optimizer state)
            # so the export can be loaded by the upstream tooling [doc 08 §6].
            torch.save({"ema": {"module": self._model.state_dict()}}, checkpoints / "weights.pth")
            ctx.save_checkpoint(best, is_best=True, metrics={"epoch": epoch})
