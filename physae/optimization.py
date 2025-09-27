"""Optuna-based utilities to optimise PhysAE training stages."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import optuna
import pandas as pd
import yaml

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


def _ensure_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _save_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _serialise(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            result[key] = _serialise(value)
        elif isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def _save_optuna_figures(study: optuna.study.Study, directory: Path) -> None:
    from optuna.visualization import matplotlib as vis

    directory.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, Callable[[optuna.study.Study], Any]] = {
        "optimization_history.png": vis.plot_optimization_history,
        "slice.png": vis.plot_slice,
        "parallel_coordinates.png": vis.plot_parallel_coordinate,
        "param_importances.png": vis.plot_param_importances,
    }
    for filename, fn in figures.items():
        try:
            fig = fn(study)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"! Impossible de générer {filename}: {exc}")
            continue
        fig.tight_layout()
        fig_path = directory / filename
        fig.savefig(fig_path, dpi=200)
        print(f"✓ Figure Optuna sauvegardée: {fig_path}")
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # pragma: no cover - optional cleanup
            pass


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
    output_dir: str | Path | None = None,
    save_figures: bool = True,
) -> optuna.study.Study:
    """Optimise a training stage using Optuna."""

    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be either 'minimize' or 'maximize'.")

    data_overrides = dict(data_overrides or {})
    if data_overrides:
        # Validate overrides early by ensuring the base configuration can be loaded.
        load_data_config(data_config_path, name=data_config_name)

    artifacts_dir = _ensure_path(output_dir)
    stage_dir: Path | None = None
    if artifacts_dir is not None:
        stage_dir = artifacts_dir / f"stage_{stage.upper()}"
        stage_dir.mkdir(parents=True, exist_ok=True)

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
        trial.set_user_attr("stage_params", copy.deepcopy(params))
        trial.set_user_attr("data_overrides", copy.deepcopy(trial_data_overrides))
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
    if stage_dir is not None:
        best_trial = study.best_trial
        summary = {
            "stage": stage,
            "metric": metric,
            "direction": direction,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_metrics": best_trial.user_attrs.get("metrics", {}),
        }
        _write_json(stage_dir / "summary.json", summary)
        stage_params = best_trial.user_attrs.get("stage_params", {})
        data_overrides_best = best_trial.user_attrs.get("data_overrides", {})
        _save_yaml(stage_dir / "best_stage_params.yaml", _serialise(stage_params))
        _save_yaml(stage_dir / "best_data_overrides.yaml", _serialise(data_overrides_best))
        records = []
        for trial in study.trials:
            records.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "state": str(trial.state),
                    "params": trial.params,
                    "metrics": trial.user_attrs.get("metrics", {}),
                }
            )
        df = pd.DataFrame.from_records(records)
        df.to_csv(stage_dir / "trials.csv", index=False)
        print(f"✓ Résultats Optuna sauvegardés dans {stage_dir}")
        if save_figures:
            _save_optuna_figures(study, stage_dir)
    return study


__all__ = ["optimise_stage"]
