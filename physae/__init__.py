"""PhysAE package exposing a modular interface for the original project.

This module only imports the lightweight configuration helpers eagerly so that
sub-packages depending on optional third-party libraries (PyTorch, Optuna,
Matplotlib, …) can still be imported individually without immediately raising
``ImportError``.  Heavier objects are exposed through a lazy lookup performed in
``__getattr__`` which mirrors the behaviour that users expect from
``from physae import build_data_and_model`` while keeping import side-effects to
a minimum.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from . import config

__all__ = [
    "config",
    "PlotAndMetricsCallback",
    "UpdateEpochInDataset",
    "SpectraDataset",
    "evaluate_and_plot",
    "build_data_and_model",
    "prepare_training_environment",
    "instantiate_model",
    "TrainingEnvironment",
    "PhysicallyInformedAE",
    "optimise_stage",
    "train_stage_A",
    "train_stage_B1",
    "train_stage_B2",
    "train_stage_custom",
    "validate_data_config",
    "validate_stage_config",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "PlotAndMetricsCallback": (".callbacks", "PlotAndMetricsCallback"),
    "UpdateEpochInDataset": (".callbacks", "UpdateEpochInDataset"),
    "SpectraDataset": (".dataset", "SpectraDataset"),
    "evaluate_and_plot": (".evaluation", "evaluate_and_plot"),
    "build_data_and_model": (".factory", "build_data_and_model"),
    "prepare_training_environment": (".factory", "prepare_training_environment"),
    "instantiate_model": (".factory", "instantiate_model"),
    "TrainingEnvironment": (".factory", "TrainingEnvironment"),
    "PhysicallyInformedAE": (".model", "PhysicallyInformedAE"),
    "optimise_stage": (".optimization", "optimise_stage"),
    "train_stage_A": (".training", "train_stage_A"),
    "train_stage_B1": (".training", "train_stage_B1"),
    "train_stage_B2": (".training", "train_stage_B2"),
    "train_stage_custom": (".training", "train_stage_custom"),
    "validate_data_config": (".validation", "validate_data_config"),
    "validate_stage_config": (".validation", "validate_stage_config"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper around importlib
    if name in _LAZY_IMPORTS:
        module_name, attribute = _LAZY_IMPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial helper
    return sorted(set(__all__))
