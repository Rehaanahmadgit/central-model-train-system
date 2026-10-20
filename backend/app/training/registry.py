"""Plugin registry [doc 08 §1].

Adding a model = adding a file under ``training/plugins/``, a row in METADATA
and a row in ``_PLUGIN_CLASS``. Nothing else changes.

Static metadata lives here (not on the plugin classes) so ``plugin_catalog()``
— which powers ``GET /models`` and the wizard's model cards — never pays the
torch/sklearn import cost. The heavy module is imported only when a run
actually trains with that family.
"""

from __future__ import annotations

import importlib
from typing import Any

from app.training.base import TrainerPlugin

# Static, import-free metadata [doc 08 §1, doc 05 §2].
METADATA: dict[str, dict[str, Any]] = {
    "random_forest": {
        "family": "random_forest",
        "supported_tasks": ["tabular_classification", "tabular_regression"],
        "supported_formats": ["tabular"],
        "description": "Classic tabular learner; trains on CPU in seconds. No pretrained weights.",
        "needs_gpu": False,
        "hparam_schema": {
            "type": "object",
            "properties": {
                "n_estimators": {"type": "integer", "minimum": 5, "default": 100},
                "max_depth": {"type": ["integer", "null"], "default": None},
            },
        },
        "default_hparams": {"n_estimators": 100, "max_depth": None, "dev_preset": False},
    },
    "cnn": {
        "family": "cnn",
        "supported_tasks": ["image_classification"],
        "supported_formats": ["image_folder", "csv_path_label"],
        "description": "Tiny from-scratch CNN (CPU or GPU). dev_preset: 2 epochs, 32 px, batch 4.",
        "needs_gpu": False,
        "hparam_schema": {
            "type": "object",
            "properties": {
                "epochs": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2,
                    "description": "Number of training epochs.",
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 4,
                    "description": "Images per gradient step.",
                },
                "imgsz": {
                    "type": "integer",
                    "minimum": 16,
                    "maximum": 1024,
                    "default": 32,
                    "description": "Input image size in pixels (square).",
                },
                "lr": {
                    "type": "number",
                    "minimum": 1e-6,
                    "default": 0.001,
                    "description": "Learning rate for Adam.",
                },
                "patience": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 3,
                    "description": "Early stopping patience in epochs; 0 disables early stopping.",
                },
                "min_delta": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                    "description": "Minimum val/accuracy improvement to count as progress.",
                },
            },
        },
        "default_hparams": {
            "epochs": 2,
            "batch_size": 4,
            "imgsz": 32,
            "lr": 0.001,
            "patience": 3,
            "min_delta": 0.0,
        },
    },
}

_PLUGIN_MODULES = {
    "random_forest": "app.training.plugins.random_forest",
    "cnn": "app.training.plugins.cnn",
}

_PLUGIN_CLASS = {
    "random_forest": "RandomForestTrainer",
    "cnn": "CNNTrainer",
}

_cache: dict[str, TrainerPlugin] = {}


def _load(family: str) -> TrainerPlugin:
    """Import the heavy plugin module once; used only when training starts."""
    if family in _cache:
        return _cache[family]
    module = importlib.import_module(_PLUGIN_MODULES[family])
    plugin: TrainerPlugin = getattr(module, _PLUGIN_CLASS[family])()
    _cache[family] = plugin
    return plugin


def all_plugins() -> list[TrainerPlugin]:
    return [_load(family) for family in _PLUGIN_MODULES]


def get_plugin(family: str) -> TrainerPlugin | None:
    if family not in _PLUGIN_MODULES:
        return None
    return _load(family)


def plugin_catalog() -> list[dict[str, Any]]:
    """The /models endpoint payload [doc 05 §2]; no heavy imports."""
    return [dict(METADATA[family]) for family in METADATA]
