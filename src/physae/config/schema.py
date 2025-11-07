"""Configuration schemas for PhysAE experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class NoiseConfig:
    """Noise parameters for synthetic spectra generation."""

    multiplicative_std: float = 0.05
    additive_std: float = 0.01
    drift_strength: float = 0.05
    spike_probability: float = 0.05


@dataclass
class DataConfig:
    """Dataset and dataloader parameters."""

    num_points: int = 1024
    train_samples: int = 128
    val_samples: int = 32
    test_samples: int = 32
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    parameter_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "baseline0": [0.0, 0.3],
            "baseline1": [-0.1, 0.1],
            "baseline2": [-0.05, 0.05],
            "peak_height": [0.3, 1.0],
            "peak_width": [5.0, 35.0],
            "peak_position": [0.15, 0.85],
        }
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass
class OptimizerConfig:
    """Optimizer hyper-parameters."""

    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    input_points: int = 1024
    encoder_channels: List[int] = field(default_factory=lambda: [8, 16, 32])
    decoder_channels: List[int] = field(default_factory=lambda: [16, 8])
    latent_dim: int = 32
    predict_parameters: List[str] = field(
        default_factory=lambda: [
            "baseline0",
            "baseline1",
            "baseline2",
            "peak_height",
            "peak_width",
            "peak_position",
        ]
    )
    dropout: float = 0.05


@dataclass
class TrainerConfig:
    """Training loop configuration."""

    max_epochs: int = 5
    gradient_clip_norm: Optional[float] = 1.0
    device: str = "cpu"
    log_every_n_steps: int = 10
    precision: str = "float32"
    accumulate_grad_batches: int = 1


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""

    project: str = "physae"
    run_name: str = "debug"
    enable_wandb: bool = False
    output_dir: Path = Path("outputs")
    resume_run_id: Optional[str] = None


@dataclass
class InferenceConfig:
    """Configuration for inference services."""

    checkpoint_path: Path = Path("outputs/latest.ckpt")
    device: str = "cpu"
    batch_size: int = 16


@dataclass
class ExperimentConfig:
    """Top-level configuration grouping all sub-configurations."""

    experiment_name: str = "physae_experiment"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
