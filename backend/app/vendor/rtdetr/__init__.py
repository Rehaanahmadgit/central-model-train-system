"""Vendored RT-DETR model + training heads (Apache-2.0).

See VENDORED.md for provenance. Model (forward) plus the S14b training pieces
(HungarianMatcher, SetCriterion, RTDETRPostProcessor); data loading, the
training loop and metrics live in ``app.training.plugins.rtdetr``.
"""

from .common import ConvNormLayer, FrozenBatchNorm2d, get_activation as _nn_act
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .presnet import PResNet
from .rtdetr import RTDETR
from .rtdetr_criterion import SetCriterion
from .rtdetr_decoder import RTDETRTransformer
from .rtdetr_postprocessor import RTDETRPostProcessor

__all__ = [
    "RTDETR",
    "PResNet",
    "HybridEncoder",
    "RTDETRTransformer",
    "HungarianMatcher",
    "SetCriterion",
    "RTDETRPostProcessor",
    "ConvNormLayer",
    "FrozenBatchNorm2d",
]
