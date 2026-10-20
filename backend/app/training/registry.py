"""Plugin registry [doc 08 §1].

Adding a model = adding a file under ``training/plugins/`` and a row here. No
other code changes. Plugins are imported lazily so a tabular run never pays
the torch import cost and vice versa.
"""

from __future__ import annotations

import importlib
from typing import Any

from app.training.base import TrainerPlugin

_PLUGIN_MODULES: dict[str, str] = {
    "random_forest": "app.training.plugins.random_forest",
    "cnn": "app.training.plugins.cnn",
}

# Trainer class name inside each plugin module.
_PLUGIN_CLASS = {
    "random_forest": "RandomForestTrainer",
    "cnn": "CNNTrainer",
}

_cache: dict[str, TrainerPlugin] = {}


def _load(family: str) -> TrainerPlugin:
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
    """The /models endpoint payload [doc 05 §2]."""
    return [
        {
            "family": plugin.family,
            "supported_tasks": plugin.supported_tasks,
            "supported_formats": plugin.supported_formats,
            "description": plugin.description,
            "needs_gpu": plugin.needs_gpu,
            "hparam_schema": plugin.hparam_schema,
            "default_hparams": plugin.default_hparams,
        }
        for plugin in all_plugins()
    ]
