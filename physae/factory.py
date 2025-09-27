"""Factory helpers to build datasets and the PhysAE model."""

from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from . import config
from .config_loader import coerce_sequence, load_data_config, merge_dicts
from .dataset import SpectraDataset
from .model import PhysicallyInformedAE
from .physics import parse_csv_transitions


def _to_tuple_map(mapping: Dict[str, Tuple[float, float]] | Dict[str, list] | None) -> Dict[str, Tuple[float, float]]:
    if mapping is None:
        return {}
    converted: Dict[str, Tuple[float, float]] = {}
    for key, value in mapping.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Interval for '{key}' must be a length-2 sequence.")
        converted[key] = (float(value[0]), float(value[1]))
    return converted


def build_data_and_model(
    *,
    seed: int | None = None,
    n_points: int | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
    batch_size: int | None = None,
    train_ranges: Dict[str, Tuple[float, float]] | None = None,
    val_ranges: Dict[str, Tuple[float, float]] | None = None,
    noise_train: Dict | None = None,
    noise_val: Dict | None = None,
    predict_list: list | None = None,
    film_list: list | None = None,
    lrs: Tuple[float, float] | None = None,
    config_path: str | None = None,
    config_name: str = "default",
    config_overrides: Dict[str, Any] | None = None,
):
    data_config = load_data_config(config_path, name=config_name)
    if config_overrides:
        data_config = merge_dicts(data_config, config_overrides)

    seed = int(seed if seed is not None else data_config.get("seed", 42))
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
    config_val_ranges = _to_tuple_map(data_config.get("val_ranges"))
    if config_val_ranges:
        default_val.update(config_val_ranges)

    expand_factors_default = {
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
    expand_factors = dict(expand_factors_default)
    expand_factors.update(data_config.get("train_ranges_expand", {}))
    base_ranges = _to_tuple_map(data_config.get("train_ranges_base")) or dict(default_val)
    default_train = config.map_ranges(base_ranges, config.expand_interval, per_param=expand_factors)
    direct_train = _to_tuple_map(data_config.get("train_ranges"))

    lo, hi = default_train["mf_CH4"]
    default_train["mf_CH4"] = (max(lo, config.LOG_FLOOR), max(hi, config.LOG_FLOOR * 10))
    lo, hi = default_val["mf_CH4"]
    default_val["mf_CH4"] = (max(lo, config.LOG_FLOOR), max(hi, config.LOG_FLOOR * 10))
    val_ranges = val_ranges or config_val_ranges or default_val
    if train_ranges is None:
        train_ranges = direct_train or default_train
    config.assert_subset(val_ranges, train_ranges, "VAL", "TRAIN")
    config.set_norm_params(train_ranges)
    noise_defaults = {
        "train": {
            "std_add_range": (0, 1e-2),
            "std_mult_range": (0, 1e-2),
            "p_drift": 0.1,
            "drift_sigma_range": (10.0, 120.0),
            "drift_amp_range": (0.004, 0.05),
            "p_fringes": 0.1,
            "n_fringes_range": (1, 2),
            "fringe_freq_range": (0.3, 50.0),
            "fringe_amp_range": (0.001, 0.015),
            "p_spikes": 0.1,
            "spikes_count_range": (1, 6),
            "spike_amp_range": (0.002, 1),
            "spike_width_range": (1.0, 20.0),
            "clip": (0.0, 1.2),
        },
        "val": {
            "std_add_range": (0, 1e-5),
            "std_mult_range": (0, 1e-5),
            "p_drift": 0,
            "drift_sigma_range": (20.0, 120.0),
            "drift_amp_range": (0.0, 0.01),
            "p_fringes": 0,
            "n_fringes_range": (1, 2),
            "fringe_freq_range": (0.5, 10.0),
            "fringe_amp_range": (0.0, 0.004),
            "p_spikes": 0.0,
            "spikes_count_range": (1, 2),
            "spike_amp_range": (0.0, 0.01),
            "spike_width_range": (1.0, 3.0),
            "clip": (0.0, 1.2),
        },
    }

    def _normalise_noise(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Convert list based intervals to tuples preserving element types."""

        result: Dict[str, Any] = {}
        for key, value in cfg.items():
            if isinstance(value, list):
                result[key] = tuple(value)
            else:
                result[key] = value
        return result

    noise_cfg = data_config.get("noise", {})
    noise_train_cfg = noise_cfg.get("train")
    noise_val_cfg = noise_cfg.get("val")
    noise_train = noise_train or _normalise_noise(noise_train_cfg or noise_defaults["train"])
    noise_val = noise_val or _normalise_noise(noise_val_cfg or noise_defaults["val"])
    n_points = int(n_points if n_points is not None else data_config.get("n_points", 800))
    n_train = int(n_train if n_train is not None else data_config.get("n_train", 50000))
    n_val = int(n_val if n_val is not None else data_config.get("n_val", 5000))
    batch_size = int(batch_size if batch_size is not None else data_config.get("batch_size", 16))
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
    predict_list = predict_list or coerce_sequence(data_config.get("predict_list")) or [
        "sig0",
        "dsig",
        "mf_CH4",
        "P",
        "T",
        "baseline1",
        "baseline2",
    ]
    film_list = film_list or coerce_sequence(data_config.get("film_list"))
    lrs = lrs or tuple(float(v) for v in data_config.get("lrs", (1e-4, 1e-5)))

    model_cfg = data_config.get("model", {}) or {}
    encoder_cfg = model_cfg.get("encoder", {}) or {}
    refiner_cfg = model_cfg.get("refiner", {}) or {}
    optimizer_cfg = model_cfg.get("optimizer", {}) or {}
    scheduler_cfg = model_cfg.get("scheduler", {}) or {}
    shared_head_hidden_scale = float(model_cfg.get("shared_head_hidden_scale", 0.5))
    encoder_name = str(encoder_cfg.get("name", "efficientnet"))
    encoder_params = dict(encoder_cfg.get("params", {}))
    for key in ("width_mult", "depth_mult", "expand_ratio_scale", "se_ratio", "norm_groups"):
        if key in encoder_cfg and key not in encoder_params:
            encoder_params[key] = encoder_cfg[key]
    encoder_width_mult = float(encoder_params.get("width_mult", 1.0))
    encoder_depth_mult = float(encoder_params.get("depth_mult", 1.0))
    encoder_expand_ratio_scale = float(encoder_params.get("expand_ratio_scale", 1.0))
    encoder_se_ratio = float(encoder_params.get("se_ratio", 0.25))
    encoder_norm_groups = int(encoder_params.get("norm_groups", 8))

    refiner_name = str(refiner_cfg.get("name", "efficientnet"))
    refiner_params = dict(refiner_cfg.get("params", {}))
    for key in ("width_mult", "depth_mult", "expand_ratio_scale", "se_ratio", "norm_groups", "hidden_scale"):
        if key in refiner_cfg and key not in refiner_params:
            refiner_params[key] = refiner_cfg[key]
    refiner_width_mult = float(refiner_params.get("width_mult", 1.0))
    refiner_depth_mult = float(refiner_params.get("depth_mult", 1.0))
    refiner_expand_ratio_scale = float(refiner_params.get("expand_ratio_scale", 1.0))
    refiner_se_ratio = float(refiner_params.get("se_ratio", 0.25))
    refiner_norm_groups = int(refiner_params.get("norm_groups", 8))
    refiner_hidden_scale = float(refiner_params.get("hidden_scale", 0.5))

    betas_cfg = optimizer_cfg.get("betas", (0.9, 0.999))
    if (
        isinstance(betas_cfg, (list, tuple))
        and len(betas_cfg) == 2
        and all(isinstance(v, (int, float)) for v in betas_cfg)
    ):
        optimizer_betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    else:
        optimizer_betas = (0.9, 0.999)
    scheduler_t_max = scheduler_cfg.get("T_max")
    if scheduler_t_max is not None:
        scheduler_t_max = int(scheduler_t_max)

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
        encoder_width_mult=encoder_width_mult,
        encoder_depth_mult=encoder_depth_mult,
        encoder_expand_ratio_scale=encoder_expand_ratio_scale,
        encoder_se_ratio=encoder_se_ratio,
        encoder_norm_groups=encoder_norm_groups,
        encoder_name=encoder_name,
        encoder_config=encoder_params,
        shared_head_hidden_scale=shared_head_hidden_scale,
        refiner_encoder_width_mult=refiner_width_mult,
        refiner_encoder_depth_mult=refiner_depth_mult,
        refiner_encoder_expand_ratio_scale=refiner_expand_ratio_scale,
        refiner_encoder_se_ratio=refiner_se_ratio,
        refiner_encoder_norm_groups=refiner_norm_groups,
        refiner_hidden_scale=refiner_hidden_scale,
        refiner_name=refiner_name,
        refiner_config=refiner_params,
        optimizer_name=str(optimizer_cfg.get("name", "adamw")),
        optimizer_betas=optimizer_betas,
        optimizer_weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
        scheduler_eta_min=float(scheduler_cfg.get("eta_min", 1e-9)),
        scheduler_T_max=scheduler_t_max,
    )
    return model, train_loader, val_loader
