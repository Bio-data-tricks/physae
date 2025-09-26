"""Factory helpers to build datasets and the PhysAE model."""

from __future__ import annotations

import sys
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import SpectraDataset
from .model import PhysicallyInformedAE
from .physics import parse_csv_transitions


def build_data_and_model(
    *,
    seed: int = 42,
    n_points: int = 800,
    n_train: int = 50000,
    n_val: int = 5000,
    batch_size: int = 16,
    train_ranges: Dict[str, Tuple[float, float]] | None = None,
    val_ranges: Dict[str, Tuple[float, float]] | None = None,
    noise_train: Dict | None = None,
    noise_val: Dict | None = None,
    predict_list: list | None = None,
    film_list: list | None = None,
    lrs: Tuple[float, float] = (1e-4, 1e-5),
):
    pl.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_windows = sys.platform == "win32"
    num_workers = 0 if is_windows else 4
    poly_freq_CH4 = [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]
    transitions_ch4_str = """6;1;3085.861015;1.013E-19;0.06;0.078;219.9411;0.73;-0.00712;0.0;0.0221;0.96;0.584;1.12
6;1;3085.832038;1.693E-19;0.0597;0.078;219.9451;0.73;-0.00712;0.0;0.0222;0.91;0.173;1.11
6;1;3085.893769;1.011E-19;0.0602;0.078;219.9366;0.73;-0.00711;0.0;0.0184;1.14;-0.516;1.37
6;1;3086.030985;1.659E-19;0.0595;0.078;219.9197;0.73;-0.00711;0.0;0.0193;1.17;-0.204;0.97
6;1;3086.071879;1.000E-19;0.0585;0.078;219.9149;0.73;-0.00703;0.0;0.0232;1.09;-0.0689;0.82
6;1;3086.085994;6.671E-20;0.055;0.078;219.9133;0.70;-0.00610;0.0;0.0300;0.54;0.00;0.0"""
    transitions_dict = {"CH4": parse_csv_transitions(transitions_ch4_str)}
    default_val = {
        "sig0": (3085.43, 3085.46),
        "dsig": (0.001521, 0.00154),
        "mf_CH4": (2e-6, 20e-6),
        "baseline0": (0.99, 1.01),
        "baseline1": (-0.0004, -0.0003),
        "baseline2": (-4.0565e-08, -3.07117e-08),
        "P": (400, 600),
        "T": (273.15 + 30, 273.15 + 40),
    }
    expand_factors = {
        "_default": 1.0,
        "sig0": 5.0,
        "dsig": 7.0,
        "mf_CH4": 2.0,
        "baseline0": 1,
        "baseline1": 3.0,
        "baseline2": 8.0,
        "P": 2.0,
        "T": 2.0,
    }
    default_train = config.map_ranges(default_val, config.expand_interval, per_param=expand_factors)
    lo, hi = default_train["mf_CH4"]
    default_train["mf_CH4"] = (max(lo, config.LOG_FLOOR), max(hi, config.LOG_FLOOR * 10))
    lo, hi = default_val["mf_CH4"]
    default_val["mf_CH4"] = (max(lo, config.LOG_FLOOR), max(hi, config.LOG_FLOOR * 10))
    val_ranges = val_ranges or default_val
    train_ranges = train_ranges or default_train
    config.assert_subset(val_ranges, train_ranges, "VAL", "TRAIN")
    config.set_norm_params(train_ranges)
    noise_train = noise_train or dict(
        std_add_range=(0, 1e-2),
        std_mult_range=(0, 1e-2),
        p_drift=0.1,
        drift_sigma_range=(10.0, 120.0),
        drift_amp_range=(0.004, 0.05),
        p_fringes=0.1,
        n_fringes_range=(1, 2),
        fringe_freq_range=(0.3, 50.0),
        fringe_amp_range=(0.001, 0.015),
        p_spikes=0.1,
        spikes_count_range=(1, 6),
        spike_amp_range=(0.002, 1),
        spike_width_range=(1.0, 20.0),
        clip=(0.0, 1.2),
    )
    noise_val = noise_val or dict(
        std_add_range=(0, 1e-5),
        std_mult_range=(0, 1e-5),
        p_drift=0,
        drift_sigma_range=(20.0, 120.0),
        drift_amp_range=(0.0, 0.01),
        p_fringes=0,
        n_fringes_range=(1, 2),
        fringe_freq_range=(0.5, 10.0),
        fringe_amp_range=(0.0, 0.004),
        p_spikes=0.0,
        spikes_count_range=(1, 2),
        spike_amp_range=(0.0, 0.01),
        spike_width_range=(1.0, 3.0),
        clip=(0.0, 1.2),
    )
    dataset_train = SpectraDataset(
        n_samples=n_train,
        num_points=n_points,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=train_ranges,
        strict_check=True,
        with_noise=True,
        noise_profile=noise_train,
        freeze_noise=False,
    )
    dataset_val = SpectraDataset(
        n_samples=n_val,
        num_points=n_points,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=val_ranges,
        strict_check=True,
        with_noise=True,
        noise_profile=noise_val,
        freeze_noise=True,
    )
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    predict_list = predict_list or ["sig0", "dsig", "mf_CH4", "P", "T", "baseline1", "baseline2"]
    film_list = film_list or []
    model = PhysicallyInformedAE(
        n_points=n_points,
        param_names=config.PARAMS,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        lr=lrs[0],
        alpha_param=0.3,
        alpha_phys=0.7,
        head_mode="multi",
        predict_params=predict_list,
        film_params=film_list,
        refine_steps=1,
        refine_delta_scale=0.1,
        refine_target="noisy",
        refine_warmup_epochs=30,
        freeze_base_epochs=20,
        base_lr=lrs[0],
        refiner_lr=lrs[1],
        baseline_fix_enable=False,
        baseline_fix_sideband=50,
        baseline_fix_degree=2,
        baseline_fix_weight=1.0,
        baseline_fix_in_warmup=False,
        recon_max1=True,
        corr_mode="none",
        corr_savgol_win=15,
        corr_savgol_poly=3,
    )
    return model, train_loader, val_loader
