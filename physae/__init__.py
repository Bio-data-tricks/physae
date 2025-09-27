"""PhysAE package exposing a modular interface for the original project."""

__version__ = "0.1.0"

from . import config
from .dataset import SpectraDataset
from .factory import build_data_and_model
from .model import PhysicallyInformedAE
from .models import available_encoders, available_refiners, register_encoder, register_refiner
from .optimization import optimise_stage
from .pipeline import expand_stages, optimise_stages, train_stages
from .training import train_stage_A, train_stage_B1, train_stage_B2, train_stage_custom

__all__ = [
    "config",
    "SpectraDataset",
    "build_data_and_model",
    "PhysicallyInformedAE",
    "optimise_stage",
    "optimise_stages",
    "train_stages",
    "expand_stages",
    "train_stage_A",
    "train_stage_B1",
    "train_stage_B2",
    "train_stage_custom",
    "available_encoders",
    "available_refiners",
    "register_encoder",
    "register_refiner",
    "evaluate_and_plot",
    "PlotAndMetricsCallback",
    "UpdateEpochInDataset",
]


def __getattr__(name: str):
    if name in {"PlotAndMetricsCallback", "UpdateEpochInDataset"}:
        from .callbacks import PlotAndMetricsCallback, UpdateEpochInDataset

        return {"PlotAndMetricsCallback": PlotAndMetricsCallback, "UpdateEpochInDataset": UpdateEpochInDataset}[name]
    if name == "evaluate_and_plot":
        from .evaluation import evaluate_and_plot

        return evaluate_and_plot
    raise AttributeError(name)
