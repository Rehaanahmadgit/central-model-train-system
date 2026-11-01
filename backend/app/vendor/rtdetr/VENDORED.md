# Vendored code — RT-DETR (official)

Source repository: https://github.com/lyuwenyu/RT-DETR
Commit: `df2a96c8c3a098567a1149bf0248da6efe6d495d`
License: Apache-2.0 (see `LICENSE`, copied verbatim from upstream).

## What was copied (model only, forward path)

| Vendored file | Upstream path | Purpose |
|---|---|---|
| `rtdetr.py` | `rtdetr_pytorch/src/zoo/rtdetr/rtdetr.py` | `RTDETR` top-level module |
| `rtdetr_decoder.py` | `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py` | `RTDETRTransformer` decoder |
| `hybrid_encoder.py` | `rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py` | `HybridEncoder` |
| `denoising.py` | `rtdetr_pytorch/src/zoo/rtdetr/denoising.py` | denoising helpers (imported by decoder) |
| `utils.py` | `rtdetr_pytorch/src/zoo/rtdetr/utils.py` | deformable attention + activations |
| `box_ops.py` | `rtdetr_pytorch/src/zoo/rtdetr/box_ops.py` | box utilities (imported by denoising) |
| `presnet.py` | `rtdetr_pytorch/src/nn/backbone/presnet.py` | `PResNet` backbone (r18vd) |
| `common.py` | `rtdetr_pytorch/src/nn/backbone/common.py` | backbone layers (imported by presnet) |
| `matcher.py` | `rtdetr_pytorch/src/zoo/rtdetr/matcher.py` | `HungarianMatcher` (S14b) |
| `rtdetr_criterion.py` | `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py` | `SetCriterion` (S14b) |
| `rtdetr_postprocessor.py` | `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py` | `RTDETRPostProcessor` (S14b) |

## What was changed

- Removed `from src.core import register` and the `@register` decorators
  (upstream's config-registry system; TrainHub builds the model directly).
- Flattened the two upstream subpackages into this single package so the
  original relative imports (`from .utils import ...`, `from .common import ...`)
  keep working without the `src.` namespace.
- `box_ops.py`: inlined `box_area` (BSD-3-Clause, from torchvision.ops.boxes)
  so the image does not need a torchvision dependency.
- `rtdetr_criterion.py`: removed the `torchvision` import
  (`torchvision.ops.sigmoid_focal_loss` re-implemented locally as
  `_sigmoid_focal_loss`) and replaced `src.misc.dist` with single-node stubs
  (`_is_dist_available_and_initialized` → False, `_get_world_size` → 1).
- `rtdetr_postprocessor.py`: replaced `torchvision.ops.box_convert` with the
  vendored `box_cxcywh_to_xyxy` (no torchvision dependency).
- `matcher.py`: unchanged apart from the registry import (depends on
  `scipy.optimize.linear_sum_assignment`, declared in the `training` extra).
- No functional/algorithmic changes to the model/loss code.

## Deliberately NOT vendored

Training/eval infrastructure (`data/`, `solver/`, `misc/`, `optim/`,
`core/config*`) — TrainHub supplies its own training loop, data loading and
metrics (S14b). EMA (`src/solver/det_solver.py`) is deferred.
