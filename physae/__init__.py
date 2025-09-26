"""PhysAE package exposing a modular interface for the original project."""

from . import config
from .callbacks import PlotAndMetricsCallback, UpdateEpochInDataset
from .dataset import SpectraDataset
from .evaluation import evaluate_and_plot
from .factory import build_data_and_model
from .model import PhysicallyInformedAE
from .training import train_stage_A, train_stage_B1, train_stage_B2, train_stage_custom

__all__ = [
    "config",
    "PlotAndMetricsCallback",
    "UpdateEpochInDataset",
    "SpectraDataset",
    "evaluate_and_plot",
    "build_data_and_model",
    "PhysicallyInformedAE",
    "train_stage_A",
    "train_stage_B1",
    "train_stage_B2",
    "train_stage_custom",
]
