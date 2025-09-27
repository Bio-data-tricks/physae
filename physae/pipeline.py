"""High-level orchestration helpers for optimisation, training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import optuna

from .config_loader import merge_dicts
from .factory import build_data_and_model
from .optimization import optimise_stage
from .training import train_stage_custom


def _expand_stage_alias(stage: str) -> list[str]:
    lookup = {
        "A": ["A"],
        "B": ["B1", "B2"],
        "B1": ["B1"],
        "B2": ["B2"],
        "ALL": ["A", "B1", "B2"],
    }
    key = stage.strip().upper()
    if key not in lookup:
        raise ValueError(f"Stage inconnu: {stage!r}")
    return lookup[key]


def expand_stages(stages: Iterable[str]) -> list[str]:
    """Expand logical stage groups (``B`` -> ``B1``/``B2``)."""

    ordered: list[str] = []
    for stage in stages:
        ordered.extend(_expand_stage_alias(stage))
    seen = set()
    unique: list[str] = []
    for stage in ordered:
        if stage not in seen:
            unique.append(stage)
            seen.add(stage)
    return unique


def _select_stage_overrides(overrides: Mapping[str, Mapping[str, Any]] | None, stage: str) -> Dict[str, Any]:
    if not overrides:
        return {}
    return dict(overrides.get(stage, overrides.get(stage.upper(), {})))


@dataclass(slots=True)
class StageOptimisationResult:
    stage: str
    study: optuna.study.Study
    stage_params: Dict[str, Any]
    data_overrides: Dict[str, Any]
    metrics: Dict[str, float]
    artifact_dir: Optional[Path]


def optimise_stages(
    stages: Iterable[str],
    *,
    n_trials: int = 20,
    metric: str = "val_loss",
    direction: str = "minimize",
    data_config_path: str | Path | None = None,
    data_config_name: str = "default",
    data_overrides: Optional[Mapping[str, Any]] = None,
    stage_config_path: str | Path | None = None,
    stage_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    show_progress_bar: bool = False,
    output_dir: str | Path | None = None,
    save_figures: bool = True,
) -> Tuple[Dict[str, StageOptimisationResult], Dict[str, Any]]:
    """Run Optuna optimisation sequentially for the requested stages.

    Returns a tuple ``(results, cumulative_overrides)`` where ``results`` is a mapping from
    stage name to :class:`StageOptimisationResult` and ``cumulative_overrides`` aggregates the
    dataset/model overrides inferred from the best trials.
    """

    cumulative_overrides: Dict[str, Any] = dict(data_overrides or {})
    stage_results: Dict[str, StageOptimisationResult] = {}
    for stage in expand_stages(stages):
        overrides = _select_stage_overrides(stage_overrides, stage)
        study = optimise_stage(
            stage,
            n_trials=n_trials,
            metric=metric,
            direction=direction,
            data_config_path=data_config_path,
            data_config_name=data_config_name,
            data_overrides=cumulative_overrides,
            stage_config_path=stage_config_path,
            stage_overrides=overrides,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            show_progress_bar=show_progress_bar,
            output_dir=output_dir,
            save_figures=save_figures,
        )
        best_trial = study.best_trial
        stage_params = dict(best_trial.user_attrs.get("stage_params", {}))
        data_best = dict(best_trial.user_attrs.get("data_overrides", {}))
        metrics_best = dict(best_trial.user_attrs.get("metrics", {}))
        artifact_dir = None
        if output_dir is not None:
            artifact_dir = Path(output_dir) / f"stage_{stage.upper()}"
        cumulative_overrides = merge_dicts(cumulative_overrides, data_best)
        stage_results[stage] = StageOptimisationResult(
            stage=stage,
            study=study,
            stage_params=stage_params,
            data_overrides=data_best,
            metrics=metrics_best,
            artifact_dir=artifact_dir,
        )
    return stage_results, cumulative_overrides


def _normalise_stage_params(params: Mapping[str, Any], stage: str, ckpt_out: Optional[Path]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = dict(params)
    resolved.setdefault("stage_name", stage)
    if ckpt_out is not None:
        resolved["ckpt_out"] = str(ckpt_out)
    return resolved


def _prepare_trainer_kwargs(kwargs: Optional[Mapping[str, Any]], default_dir: Optional[Path]) -> Dict[str, Any]:
    if not kwargs and default_dir is None:
        return {}
    merged: Dict[str, Any] = dict(kwargs or {})
    if default_dir is not None:
        merged.setdefault("default_root_dir", str(default_dir))
    return merged


def train_stages(
    stages: Iterable[str],
    *,
    stage_params: Mapping[str, Mapping[str, Any]],
    data_config_path: str | Path | None = None,
    data_config_name: str = "default",
    data_overrides: Optional[Mapping[str, Any]] = None,
    ckpt_dir: str | Path | None = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    fine_tune_only: bool = False,
) -> Tuple[Any, Dict[str, Dict[str, float]], Optional[Path]]:
    """Sequentially train the requested stages with the provided parameters."""

    stages_expanded = expand_stages(stages)
    required = [stage for stage in stages_expanded if not (fine_tune_only and stage.upper() != "B2")]
    missing = [stage for stage in required if stage not in stage_params]
    if missing:
        raise KeyError(f"Paramètres manquants pour les stages: {', '.join(missing)}")
    overrides = dict(data_overrides or {})
    model, (train_loader, val_loader), _ = build_data_and_model(
        config_path=data_config_path,
        config_name=data_config_name,
        config_overrides=overrides or None,
    )
    ckpt_directory = Path(ckpt_dir) if ckpt_dir is not None else None
    metrics: Dict[str, Dict[str, float]] = {}
    prev_ckpt: Optional[Path] = None
    trainer_opts = _prepare_trainer_kwargs(trainer_kwargs, ckpt_directory)
    for stage in stages_expanded:
        if fine_tune_only and stage.upper() != "B2":
            continue
        ckpt_out = None
        if ckpt_directory is not None:
            ckpt_directory.mkdir(parents=True, exist_ok=True)
            ckpt_out = ckpt_directory / f"stage_{stage.upper()}.ckpt"
        params = _normalise_stage_params(stage_params[stage], stage, ckpt_out)
        if prev_ckpt is not None:
            params.setdefault("ckpt_in", str(prev_ckpt))
        trained, stage_metrics = train_stage_custom(
            model,
            train_loader,
            val_loader,
            **params,
            return_metrics=True,
            trainer_kwargs=trainer_opts,
        )
        model = trained
        metrics[stage] = stage_metrics
        if ckpt_out is not None:
            prev_ckpt = ckpt_out
        elif "ckpt_out" in params:
            prev_ckpt = Path(params["ckpt_out"])
    return model, metrics, prev_ckpt


__all__ = ["StageOptimisationResult", "optimise_stages", "train_stages", "expand_stages"]

