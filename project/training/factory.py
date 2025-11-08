"""Utilities to mirror ``physae.py``'s data/model preparation pipeline."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader

from config import params as params_cfg
from config.training_config import TrainingConfig
from data.dataset import SpectraDataset
from models.autoencoder import PhysicallyInformedAE


def _default_num_workers() -> int:
    """Replicate the platform-dependent worker heuristic from ``physae.py``."""
    import sys

    return 0 if sys.platform == "win32" else 4


def prepare_datasets(
    config: TrainingConfig,
    *,
    poly_freq_CH4,
    transitions_dict: Dict[str, Iterable],
    tipspy=None,
    freeze_train_parameters: bool = False,
    freeze_val_parameters: bool = False,
    freeze_val_noise: bool = False,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders matching ``physae.py`` defaults."""

    dataset_cfg = config.dataset_kwargs()

    # Align global normalisation with the training ranges as the monolithic
    # script did inside ``build_data_and_model``.
    params_cfg.NORM_PARAMS.clear()
    params_cfg.NORM_PARAMS.update(dataset_cfg["train_ranges"])

    if num_workers is None:
        num_workers = _default_num_workers()

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset = SpectraDataset(
        n_samples=dataset_cfg["n_train"],
        num_points=dataset_cfg["n_points"],
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=dataset_cfg["train_ranges"],
        strict_check=True,
        with_noise=True,
        noise_profile=dataset_cfg["noise_train"],
        freeze_parameters=freeze_train_parameters,
        freeze_noise=False,
        tipspy=tipspy,
    )

    val_dataset = SpectraDataset(
        n_samples=dataset_cfg["n_val"],
        num_points=dataset_cfg["n_points"],
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=dataset_cfg["val_ranges"],
        strict_check=True,
        with_noise=True,
        noise_profile=dataset_cfg["noise_val"],
        freeze_parameters=freeze_val_parameters,
        freeze_noise=freeze_val_noise,
        tipspy=tipspy,
    )

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader


def prepare_model(
    config: TrainingConfig,
    *,
    transitions_dict: Dict[str, Iterable],
    poly_freq_CH4,
    tipspy=None,
    stage: str | None = None,
) -> PhysicallyInformedAE:
    """Instantiate :class:`PhysicallyInformedAE` with faithful defaults."""

    model_kwargs = config.model_kwargs()
    if stage is not None:
        model_kwargs.update(config.stage_overrides(stage))

    model = PhysicallyInformedAE(
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        tipspy=tipspy,
        **model_kwargs,
    )

    # Match optimisation defaults from the monolithic script.
    model.hparams.optimizer = "lion"
    model.hparams.betas = (0.9, 0.99)
    model.weight_decay = 1e-4

    print("\n" + "=" * 70)
    print("🔍 Vérification des poids de loss")
    print("=" * 70)
    print(f"w_pw_raw          = {model.w_pw_raw}")
    print(f"w_spectral_angle  = {model.w_spectral_angle}")
    print(f"w_peak            = {model.w_peak}")
    print(f"w_params          = {model.w_params}")
    print(f"use_relobralo_top = {model.use_relobralo_top}")
    print(f"peak_loss         = {type(model.peak_weighted_loss).__name__}")
    if model.use_relobralo_top and getattr(model, "relo_top", None) is not None:
        print(f"relo_top          = {type(model.relo_top).__name__} (actif)")
    print("=" * 70 + "\n")

    return model


def build_data_and_model(
    config: TrainingConfig,
    *,
    transitions_dict: Dict[str, Iterable],
    poly_freq_CH4,
    tipspy=None,
    freeze_train_parameters: bool = False,
    freeze_val_parameters: bool = False,
    freeze_val_noise: bool = False,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    stage: str | None = None,
) -> Tuple[PhysicallyInformedAE, DataLoader, DataLoader]:
    """Convenience wrapper mirroring ``physae.py``'s helper."""

    train_loader, val_loader = prepare_datasets(
        config,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        tipspy=tipspy,
        freeze_train_parameters=freeze_train_parameters,
        freeze_val_parameters=freeze_val_parameters,
        freeze_val_noise=freeze_val_noise,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = prepare_model(
        config,
        transitions_dict=transitions_dict,
        poly_freq_CH4=poly_freq_CH4,
        tipspy=tipspy,
        stage=stage,
    )

    return model, train_loader, val_loader
