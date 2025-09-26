"""Optuna-based utilities to optimise PhysAE training stages."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence

import optuna

from .config_loader import load_data_config, load_stage_config, merge_dicts
from .factory import build_data_and_model
from .training import train_stage_custom


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: Mapping[str, Any]) -> Any:
    kind = spec.get("type", "float")
    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if step is not None:
            return trial.suggest_float(name, low, high, step=float(step), log=log)
        return trial.suggest_float(name, low, high, log=log)
    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        if log:
            return trial.suggest_int(name, low, high, log=log)
        return trial.suggest_int(name, low, high, step=step)
    if kind == "categorical":
        choices = list(spec["choices"])
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unknown Optuna parameter type '{kind}' for '{name}'.")


def _assign_nested(target: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    if not keys:
        return
    node = target
    for key in keys[:-1]:
        existing = node.get(key)
        if not isinstance(existing, dict):
            existing = {}
            node[key] = existing
        node = existing
    node[keys[-1]] = value


def optimise_stage(
    stage: str,
    *,
    n_trials: int = 20,
    metric: str = "val_loss",
    direction: str = "minimize",
    data_config_path: str | None = None,
    data_config_name: str = "default",
    data_overrides: Optional[Mapping[str, Any]] = None,
    stage_config_path: str | None = None,
    stage_overrides: Optional[Mapping[str, Any]] = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    show_progress_bar: bool = False,
) -> optuna.study.Study:
    """Optimise a training stage using Optuna."""

    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be either 'minimize' or 'maximize'.")

    data_overrides = dict(data_overrides or {})
    if data_overrides:
        # Validate overrides early by ensuring the base configuration can be loaded.
        load_data_config(data_config_path, name=data_config_name)

    def objective(trial: optuna.Trial) -> float:
        stage_cfg = load_stage_config(stage, path=stage_config_path)
        if stage_overrides:
            stage_cfg = merge_dicts(stage_cfg, stage_overrides)
        search_space = stage_cfg.pop("optuna", {})
        params: Dict[str, Any] = {}
        params.update(stage_cfg)
        trial_data_overrides = copy.deepcopy(data_overrides)
        for param_name, spec in search_space.items():
            value = _suggest_from_spec(trial, param_name, spec)
            if param_name.startswith("data."):
                _assign_nested(trial_data_overrides, param_name.split(".")[1:], value)
            elif param_name.startswith("model."):
                _assign_nested(trial_data_overrides, ["model", *param_name.split(".")[1:]], value)
            else:
                params[param_name] = value
        model, train_loader, val_loader = build_data_and_model(
            config_path=data_config_path,
            config_name=data_config_name,
            config_overrides=trial_data_overrides or None,
        )
        _, metrics = train_stage_custom(
            model,
            train_loader,
            val_loader,
            **params,
            return_metrics=True,
        )
        if metric not in metrics:
            raise KeyError(
                f"Metric '{metric}' not found in trainer callback metrics: {list(metrics)}",
            )
        score = float(metrics[metric])
        trial.set_user_attr("metrics", metrics)
        return score

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


__all__ = ["optimise_stage"]
