"""Factory helpers to build datasets and PhysAE training environments."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from . import config
from .config_loader import coerce_sequence, load_data_config, merge_dicts
from .dataset import SpectraDataset
from .model import PhysicallyInformedAE
from .physics import parse_csv_transitions


_POLY_FREQ_CH4 = (-2.3614803e-07, 1.2103413e-10, -3.1617856e-14)
_TRANSITIONS_CH4_STR = """6;1;3085.861015;1.013E-19;0.06;0.078;219.9411;0.73;-0.00712;0.0;0.0221;0.96;0.584;1.12
6;1;3085.832038;1.693E-19;0.0597;0.078;219.9451;0.73;-0.00712;0.0;0.0222;0.91;0.173;1.11
6;1;3085.893769;1.011E-19;0.0602;0.078;219.9366;0.73;-0.00711;0.0;0.0184;1.14;-0.516;1.37
6;1;3086.030985;1.659E-19;0.0595;0.078;219.9197;0.73;-0.00711;0.0;0.0193;1.17;-0.204;0.97
6;1;3086.071879;1.000E-19;0.0585;0.078;219.9149;0.73;-0.00703;0.0;0.0232;1.09;-0.0689;0.82
6;1;3086.085994;6.671E-20;0.055;0.078;219.9133;0.70;-0.00610;0.0;0.0300;0.54;0.00;0.0"""


def _base_transitions_dict() -> Dict[str, list]:
    """Return a fresh copy of the default transition table."""

    return {"CH4": parse_csv_transitions(_TRANSITIONS_CH4_STR)}


def _to_tuple_map(
    mapping: Dict[str, Tuple[float, float]] | Dict[str, list] | None,
) -> Dict[str, Tuple[float, float]]:
    if mapping is None:
        return {}
    converted: Dict[str, Tuple[float, float]] = {}
    for key, value in mapping.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Interval for '{key}' must be a length-2 sequence.")
        converted[key] = (float(value[0]), float(value[1]))
    return converted


def _normalise_noise(cfg: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, list):
            result[key] = tuple(float(v) for v in value)
        else:
            result[key] = value
    return result


@dataclass
class TrainingEnvironment:
    """Container bundling dataloaders and model kwargs for training."""

    train_loader: DataLoader
    val_loader: DataLoader
    model_kwargs: Dict[str, Any]
    seed: int

    def spawn_model(self, *, seed: int | None = None) -> PhysicallyInformedAE:
        """Instantiate a new :class:`PhysicallyInformedAE` with optional reseed."""

        if seed is not None:
            try:  # pragma: no cover - dependency on external library state
                pl.seed_everything(seed)
            except Exception:
                pass
        return instantiate_model(self.model_kwargs)

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary useful for notebook displays."""

        return {
            "train_samples": len(self.train_loader.dataset),
            "val_samples": len(self.val_loader.dataset),
            "batch_size": self.train_loader.batch_size,
            "n_points": self.model_kwargs.get("n_points"),
            "predict_params": list(self.model_kwargs.get("predict_params", [])),
            "film_params": list(self.model_kwargs.get("film_params", [])),
        }


def prepare_training_environment(
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
) -> TrainingEnvironment:
    """Build dataloaders and model kwargs according to the YAML configuration."""

    data_config = load_data_config(config_path, name=config_name)
    if config_overrides:
        data_config = merge_dicts(data_config, config_overrides)

    seed_value = int(seed if seed is not None else data_config.get("seed", 42))
    pl.seed_everything(seed_value)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_windows = sys.platform == "win32"
    num_workers = 0 if is_windows else 4

    poly_freq_CH4 = list(_POLY_FREQ_CH4)
    transitions_dict = _base_transitions_dict()

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
        "baseline0": 1.0,
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

    model_transitions = {
        mol: tuple(dict(item) for item in transitions)
        for mol, transitions in transitions_dict.items()
    }
    model_kwargs = {
        "n_points": n_points,
        "param_names": config.PARAMS,
        "poly_freq_CH4": tuple(poly_freq_CH4),
        "transitions_dict": model_transitions,
        "lr": lrs[0],
        "alpha_param": 0.3,
        "alpha_phys": 0.7,
        "head_mode": "multi",
        "predict_params": list(predict_list),
        "film_params": list(film_list) if film_list is not None else None,
        "refine_steps": 1,
        "refine_delta_scale": 0.1,
        "refine_target": "noisy",
        "refine_warmup_epochs": 30,
        "freeze_base_epochs": 20,
        "base_lr": lrs[0],
        "refiner_lr": lrs[1] if len(lrs) > 1 else lrs[0],
        "baseline_fix_enable": False,
        "baseline_fix_sideband": 50,
        "baseline_fix_degree": 2,
        "baseline_fix_weight": 1.0,
        "baseline_fix_in_warmup": False,
        "recon_max1": True,
        "corr_mode": "none",
        "corr_savgol_win": 15,
        "corr_savgol_poly": 3,
    }

    return TrainingEnvironment(train_loader, val_loader, model_kwargs, seed_value)


def instantiate_model(source: Mapping[str, Any] | TrainingEnvironment) -> PhysicallyInformedAE:
    """Instantiate a :class:`PhysicallyInformedAE` from saved kwargs or an environment."""

    if isinstance(source, TrainingEnvironment):
        return instantiate_model(source.model_kwargs)

    kwargs = copy.deepcopy(dict(source))
    kwargs.setdefault("param_names", config.PARAMS)
    poly_freq = kwargs.get("poly_freq_CH4", _POLY_FREQ_CH4)
    kwargs["poly_freq_CH4"] = list(poly_freq)
    transitions_dict = kwargs.get("transitions_dict", {})
    kwargs["transitions_dict"] = {
        mol: [dict(item) for item in transitions]
        for mol, transitions in transitions_dict.items()
    }
    return PhysicallyInformedAE(**kwargs)


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
    """Backwards-compatible helper returning a model alongside the dataloaders."""

    env = prepare_training_environment(
        seed=seed,
        n_points=n_points,
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        train_ranges=train_ranges,
        val_ranges=val_ranges,
        noise_train=noise_train,
        noise_val=noise_val,
        predict_list=predict_list,
        film_list=film_list,
        lrs=lrs,
        config_path=config_path,
        config_name=config_name,
        config_overrides=config_overrides,
    )
    model = instantiate_model(env)
    return model, env.train_loader, env.val_loader


__all__ = [
    "TrainingEnvironment",
    "prepare_training_environment",
    "instantiate_model",
    "build_data_and_model",
]
