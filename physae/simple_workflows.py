"""High-level helpers to launch PhysAE training and optimisation without the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import optuna

from .config_loader import load_stage_config, merge_dicts
from .pipeline import (
    StageOptimisationResult,
    StageTrainingResult,
    expand_stages,
    optimise_stages,
    train_stages,
)


@dataclass
class StageRunConfig:
    """Configuration for a single stage training run."""

    stage: str
    data_config_path: str | None = None
    data_config_name: str = "default"
    data_overrides: Dict[str, Any] = field(default_factory=dict)
    stage_config_path: str | None = None
    stage_overrides: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    ckpt_dir: str | None = None
    ckpt_in: str | None = None
    ckpt_out: str | None = None


@dataclass
class StageSequenceConfig:
    """Configuration to train a sequence of stages (e.g. A -> B1 -> B2)."""

    stages: Iterable[str]
    data_config_path: str | None = None
    data_config_name: str = "default"
    data_overrides: Dict[str, Any] = field(default_factory=dict)
    stage_config_path: str | None = None
    stage_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    ckpt_dir: str | None = None
    fine_tune_only: bool = False


@dataclass
class OptunaRunConfig:
    """Configuration for an Optuna optimisation sweep."""

    stages: Iterable[str]
    n_trials: int = 20
    metric: str = "val_loss"
    direction: str = "minimize"
    data_config_path: str | None = None
    data_config_name: str = "default"
    data_overrides: Dict[str, Any] = field(default_factory=dict)
    stage_config_path: str | None = None
    stage_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sampler: str | None = None
    pruner: str | None = None
    storage: str | None = None
    study_name: str | None = None
    show_progress: bool = False
    output_dir: str | None = None
    save_figures: bool = True


def _load_stage_params(stage: str, *, config_path: str | None, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    stage_cfg = load_stage_config(stage, path=config_path)
    params = merge_dicts(stage_cfg, overrides)
    params.pop("optuna", None)
    return params


def run_single_stage(config: StageRunConfig) -> Tuple[Dict[str, float], Path | None]:
    """Train a single stage and return metrics and the last checkpoint path."""

    stage = config.stage
    stage_params = _load_stage_params(stage, config_path=config.stage_config_path, overrides=config.stage_overrides)
    if config.ckpt_in:
        stage_params["ckpt_in"] = config.ckpt_in
    if config.ckpt_out:
        stage_params["ckpt_out"] = config.ckpt_out
    stage_params.setdefault("stage_name", stage)
    stage_map = {stage: stage_params}
    training_result = train_stages(
        [stage],
        stage_params=stage_map,
        data_config_path=config.data_config_path,
        data_config_name=config.data_config_name,
        data_overrides=config.data_overrides,
        ckpt_dir=config.ckpt_dir,
        trainer_kwargs=config.trainer_kwargs,
    )
    metrics = training_result.metrics.get(stage, {})
    return metrics, training_result.last_checkpoint


def run_stage_sequence(config: StageSequenceConfig) -> StageTrainingResult:
    """Train several stages sequentially (A, B1, B2)."""

    stage_map: Dict[str, Dict[str, Any]] = {}
    for stage in expand_stages(config.stages):
        overrides = config.stage_overrides.get(stage) or config.stage_overrides.get(stage.upper()) or {}
        stage_map[stage] = _load_stage_params(stage, config_path=config.stage_config_path, overrides=overrides)
    return train_stages(
        config.stages,
        stage_params=stage_map,
        data_config_path=config.data_config_path,
        data_config_name=config.data_config_name,
        data_overrides=config.data_overrides,
        ckpt_dir=config.ckpt_dir,
        trainer_kwargs=config.trainer_kwargs,
        fine_tune_only=config.fine_tune_only,
    )


def _resolve_sampler(name: str | None) -> optuna.samplers.BaseSampler | None:
    if name is None:
        return None
    lower = name.lower()
    if lower == "tpe":
        return optuna.samplers.TPESampler()
    if lower == "random":
        return optuna.samplers.RandomSampler()
    if lower == "cmaes":
        return optuna.samplers.CmaEsSampler()
    raise ValueError(f"Sampler inconnu: {name}")


def _resolve_pruner(name: str | None) -> optuna.pruners.BasePruner | None:
    if name is None:
        return None
    lower = name.lower()
    if lower == "median":
        return optuna.pruners.MedianPruner()
    if lower == "successivehalving":
        return optuna.pruners.SuccessiveHalvingPruner()
    if lower == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Pruner inconnu: {name}")


def run_optuna(config: OptunaRunConfig) -> Tuple[Dict[str, StageOptimisationResult], Dict[str, Any]]:
    """Launch an Optuna sweep for the requested stages and return the studies."""

    sampler = _resolve_sampler(config.sampler)
    pruner = _resolve_pruner(config.pruner)
    return optimise_stages(
        config.stages,
        n_trials=config.n_trials,
        metric=config.metric,
        direction=config.direction,
        data_config_path=config.data_config_path,
        data_config_name=config.data_config_name,
        data_overrides=config.data_overrides,
        stage_config_path=config.stage_config_path,
        stage_overrides=config.stage_overrides,
        sampler=sampler,
        pruner=pruner,
        storage=config.storage,
        study_name=config.study_name,
        show_progress_bar=config.show_progress,
        output_dir=config.output_dir,
        save_figures=config.save_figures,
    )


__all__ = [
    "StageRunConfig",
    "StageSequenceConfig",
    "OptunaRunConfig",
    "run_single_stage",
    "run_stage_sequence",
    "run_optuna",
]
