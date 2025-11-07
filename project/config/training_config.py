"""Structured training configuration derived from ``physae.py``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Tuple

from .params import LOG_FLOOR, LOG_SCALE_PARAMS, PARAMS

DEFAULT_PREDICT_PARAMS = ["sig0", "dsig", "P", "T", "mf_CH4", "baseline1", "baseline2"]


def _default_val_ranges() -> dict[str, tuple[float, float]]:
    return {
        "sig0": (3085.43, 3085.46),
        "dsig": (0.001521, 0.00154),
        "mf_CH4": (2e-6, 20e-6),
        "mf_H2O": (5e-5, 3e-4),
        "baseline0": (0.999999, 1.00001),
        "baseline1": (-4.0e-4, -3.0e-4),
        "baseline2": (-4.0565e-8, -3.07117e-8),
        "P": (450.0, 550.0),
        "T": (273.15 + 32, 273.15 + 37),
    }


def _expand_interval(a: float, b: float, factor: float) -> tuple[float, float]:
    center = 0.5 * (a + b)
    half = 0.5 * (b - a) * float(factor)
    return float(center - half), float(center + half)


def _map_ranges(
    base: Mapping[str, tuple[float, float]],
    *,
    per_param: Mapping[str, float] | None = None,
    default_factor: float = 1.0,
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for key, (lo, hi) in base.items():
        factor = default_factor
        if per_param:
            factor = per_param.get(key, per_param.get("_default", default_factor))
        result[key] = _expand_interval(lo, hi, factor)
    return result


def _apply_log_floor(ranges: MutableMapping[str, tuple[float, float]]) -> MutableMapping[str, tuple[float, float]]:
    for name in LOG_SCALE_PARAMS:
        if name in ranges:
            lo, hi = ranges[name]
            ranges[name] = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR * 10))
    return ranges


def _default_noise_profile(train: bool) -> dict:
    if train:
        return {
            "std_add_range": (0.0, 5e-3),
            "std_mult_range": (0.0, 5e-3),
            "p_drift": 0.2,
            "drift_sigma_range": (1.0, 100.0),
            "drift_amp_range": (1e-5, 5e-3),
            "p_fringes": 0.2,
            "n_fringes_range": (1, 2),
            "fringe_freq_range": (0.3, 50.0),
            "fringe_amp_range": (1e-3, 1.5e-3),
            "p_spikes": 0.2,
            "spikes_count_range": (1, 2),
            "spike_amp_range": (1e-3, 1.0),
            "spike_width_range": (1.0, 200.0),
            "clip": (0.0, 1.5),
            "complex": {
                "probability": 0.7,
                "noise_types": [
                    "gaussian",
                    "shot",
                    "flicker",
                    "etaloning",
                    "glitches",
                ],
                "noise_type_weights": [0.25, 0.15, 0.2, 0.15, 0.25],
                "noise_level_range": (0.4, 1.6),
                "max_rel_to_line": 0.10,
                "mode": "blend",
                "blend_alpha": 0.35,
                "clip": (0.0, 1.5),
            },
        }
    return {
        "std_add_range": (0.0, 1e-3),
        "std_mult_range": (0.0, 1e-3),
        "p_drift": 0.0,
        "drift_sigma_range": (1.0, 100.0),
        "drift_amp_range": (1e-5, 5e-3),
        "p_fringes": 0.0,
        "n_fringes_range": (1, 2),
        "fringe_freq_range": (0.3, 50.0),
        "fringe_amp_range": (1e-5, 1.5e-3),
        "p_spikes": 0.0,
        "spikes_count_range": (1, 2),
        "spike_amp_range": (1e-4, 1.0),
        "spike_width_range": (1.0, 200.0),
        "clip": (0.0, 1.5),
        "complex": {
            "probability": 0.0,
            "noise_types": ["gaussian", "shot", "flicker", "etaloning", "glitches"],
            "noise_level_range": (0.2, 1.0),
            "max_rel_to_line": 0.08,
            "mode": "replace",
            "clip": (0.0, 1.5),
        },
    }


@dataclass
class TrainingConfig:
    """Container aggregating dataset, model and optimisation defaults."""

    seed: int = 42
    n_points: int = 800
    n_train: int = 500_000
    n_val: int = 5_000
    batch_size: int = 32
    train_ranges: dict[str, tuple[float, float]] | None = None
    val_ranges: dict[str, tuple[float, float]] | None = None
    noise_train: dict | None = None
    noise_val: dict | None = None
    predict_params: list[str] = field(default_factory=lambda: list(DEFAULT_PREDICT_PARAMS))
    learning_rates: Tuple[float, float] = (1e-4, 1e-5)

    backbone_variant: str = "s"
    refiner_variant: str = "s"
    backbone_width_mult: float = 1.0
    backbone_depth_mult: float = 0.4
    refiner_width_mult: float = 1.0
    refiner_depth_mult: float = 1.0
    backbone_stem_channels: int | None = None
    refiner_stem_channels: int | None = None
    backbone_drop_path: float = 0.0
    refiner_drop_path: float = 0.0
    backbone_se_ratio: float = 0.25
    refiner_se_ratio: float = 0.25
    refiner_feature_pool: str = "avg"
    refiner_shared_hidden_scale: float = 0.5

    huber_beta: float = 0.002
    w_pw_raw: float = 1.0
    w_spectral_angle: float = 0.5
    w_peak: float = 0.0
    w_params: float = 1.0
    use_relobralo_top: bool = True
    pw_cfg: dict | None = None

    scheduler_T_0: int = 10
    scheduler_T_mult: float = 1.2
    scheduler_eta_min: float = 1e-9
    scheduler_decay_factor: float = 0.75
    scheduler_warmup_epochs: int = 5

    qtpy_dir: str | None = None

    def resolved_val_ranges(self) -> dict[str, tuple[float, float]]:
        base = _default_val_ranges() if self.val_ranges is None else dict(self.val_ranges)
        return _apply_log_floor(base)

    def resolved_train_ranges(self) -> dict[str, tuple[float, float]]:
        if self.train_ranges is not None:
            return _apply_log_floor(dict(self.train_ranges))

        per_param = {
            "_default": 1.0,
            "sig0": 5.0,
            "dsig": 3.0,
            "mf_CH4": 2.0,
            "mf_H2O": 2.0,
            "baseline0": 1.0,
            "baseline1": 3.0,
            "baseline2": 8.0,
            "P": 2.0,
            "T": 2.0,
        }
        ranges = _map_ranges(self.resolved_val_ranges(), per_param=per_param)
        return _apply_log_floor(ranges)

    def resolved_noise_profiles(self) -> tuple[dict, dict]:
        train_noise = dict(_default_noise_profile(train=True))
        val_noise = dict(_default_noise_profile(train=False))
        if self.noise_train is not None:
            train_noise.update(self.noise_train)
        if self.noise_val is not None:
            val_noise.update(self.noise_val)
        return train_noise, val_noise

    def model_kwargs(self) -> dict:
        return {
            "n_points": self.n_points,
            "param_names": PARAMS,
            "predict_params": list(self.predict_params),
            "lr": self.learning_rates[0],
            "base_lr": self.learning_rates[0],
            "refiner_lr": self.learning_rates[1],
            "huber_beta": self.huber_beta,
            "w_pw_raw": self.w_pw_raw,
            "w_spectral_angle": self.w_spectral_angle,
            "w_peak": self.w_peak,
            "w_params": self.w_params,
            "use_relobralo_top": self.use_relobralo_top,
            "pw_cfg": self.pw_cfg,
            "backbone_variant": self.backbone_variant,
            "refiner_variant": self.refiner_variant,
            "backbone_width_mult": self.backbone_width_mult,
            "backbone_depth_mult": self.backbone_depth_mult,
            "refiner_width_mult": self.refiner_width_mult,
            "refiner_depth_mult": self.refiner_depth_mult,
            "backbone_stem_channels": self.backbone_stem_channels,
            "refiner_stem_channels": self.refiner_stem_channels,
            "backbone_drop_path": self.backbone_drop_path,
            "refiner_drop_path": self.refiner_drop_path,
            "backbone_se_ratio": self.backbone_se_ratio,
            "refiner_se_ratio": self.refiner_se_ratio,
            "refiner_feature_pool": self.refiner_feature_pool,
            "refiner_shared_hidden_scale": self.refiner_shared_hidden_scale,
            "scheduler_T_0": self.scheduler_T_0,
            "scheduler_T_mult": self.scheduler_T_mult,
            "scheduler_eta_min": self.scheduler_eta_min,
            "scheduler_decay_factor": self.scheduler_decay_factor,
            "scheduler_warmup_epochs": self.scheduler_warmup_epochs,
        }

    def dataset_kwargs(self) -> dict:
        train_noise, val_noise = self.resolved_noise_profiles()
        return {
            "seed": self.seed,
            "n_points": self.n_points,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "batch_size": self.batch_size,
            "train_ranges": self.resolved_train_ranges(),
            "val_ranges": self.resolved_val_ranges(),
            "noise_train": train_noise,
            "noise_val": val_noise,
        }

    def stage_overrides(self, stage: str) -> dict:
        """Return scheduler/loss overrides matching ``physae.py`` for a stage."""

        stage = stage.upper()
        if stage == "A":
            return {"use_relobralo_top": False}
        if stage == "DEN":
            return {"use_denoiser": True, "denoiser_lr": 1e-4}
        if stage in {"B1", "B2"}:
            return {"refine_steps": 1 if stage == "B1" else 2}
        return {}

    def as_dict(self) -> dict:
        """Serialize the config (including resolved ranges/noise) to a dict."""

        train_noise, val_noise = self.resolved_noise_profiles()
        return {
            "seed": self.seed,
            "n_points": self.n_points,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "batch_size": self.batch_size,
            "train_ranges": self.resolved_train_ranges(),
            "val_ranges": self.resolved_val_ranges(),
            "noise_train": train_noise,
            "noise_val": val_noise,
            "predict_params": list(self.predict_params),
            "learning_rates": self.learning_rates,
            "scheduler": {
                "T_0": self.scheduler_T_0,
                "T_mult": self.scheduler_T_mult,
                "eta_min": self.scheduler_eta_min,
                "decay_factor": self.scheduler_decay_factor,
                "warmup_epochs": self.scheduler_warmup_epochs,
            },
            "loss_weights": {
                "w_pw_raw": self.w_pw_raw,
                "w_spectral_angle": self.w_spectral_angle,
                "w_peak": self.w_peak,
                "w_params": self.w_params,
                "use_relobralo_top": self.use_relobralo_top,
            },
            "architecture": {
                "backbone_variant": self.backbone_variant,
                "refiner_variant": self.refiner_variant,
                "backbone_width_mult": self.backbone_width_mult,
                "backbone_depth_mult": self.backbone_depth_mult,
                "refiner_width_mult": self.refiner_width_mult,
                "refiner_depth_mult": self.refiner_depth_mult,
                "backbone_drop_path": self.backbone_drop_path,
                "refiner_drop_path": self.refiner_drop_path,
                "backbone_se_ratio": self.backbone_se_ratio,
                "refiner_se_ratio": self.refiner_se_ratio,
                "refiner_feature_pool": self.refiner_feature_pool,
                "refiner_shared_hidden_scale": self.refiner_shared_hidden_scale,
            },
            "qtpy_dir": self.qtpy_dir,
        }
