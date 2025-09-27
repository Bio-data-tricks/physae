#!/usr/bin/env python
"""Optuna helper script for PhysAE stages with ready-to-use presets."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable

from physae.parameter_catalog import set_by_path
from physae.simple_workflows import OptunaRunConfig, run_optuna


def _parse_stage_list(value: str) -> Iterable[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_overrides(values: list[str]) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Format invalide pour l'override: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if ":" not in key:
            raise ValueError(
                "Chaque override Optuna doit préciser le stage: ex. B1:model.encoder.params.width_mult=1.2"
            )
        stage, param = [part.strip() for part in key.split(":", 1)]
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        stage_dict = overrides.setdefault(stage, {})
        set_by_path(stage_dict, param, value)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        type=_parse_stage_list,
        default=["A"],
        help="Stages à optimiser (séparés par des virgules, ex: A,B ou A,B1,B2)",
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Nombre d'essais Optuna par stage")
    parser.add_argument("--metric", type=str, default="val_loss", help="Nom de la métrique à optimiser")
    parser.add_argument(
        "--direction", type=str, choices=["minimize", "maximize"], default="minimize", help="Direction"
    )
    parser.add_argument("--sampler", type=str, default="tpe", help="Sampler Optuna (tpe, random, cmaes)")
    parser.add_argument(
        "--pruner",
        type=str,
        default=None,
        help="Pruner Optuna (median, successivehalving, hyperband)",
    )
    parser.add_argument("--storage", type=str, default=None, help="URL de stockage Optuna")
    parser.add_argument("--study-name", type=str, default=None, help="Nom de l'étude Optuna")
    parser.add_argument("--output-dir", type=str, default="optuna_runs", help="Répertoire de sortie")
    parser.add_argument(
        "--data-override",
        action="append",
        default=[],
        metavar="cle=valeur",
        help="Overrides JSON pour la configuration des données",
    )
    parser.add_argument(
        "--stage-override",
        action="append",
        default=[],
        metavar="stage:cle=valeur",
        help="Overrides JSON spécifiques à un stage",
    )
    parser.add_argument("--no-figures", action="store_true", help="Ne pas générer les graphiques Optuna")
    parser.add_argument("--show-progress", action="store_true", help="Afficher la barre de progression Optuna")
    return parser


def _parse_data_overrides(values: list[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Format invalide: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        set_by_path(data, key, value)
    return data


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    stage_overrides = _parse_overrides(args.stage_override)
    data_overrides = _parse_data_overrides(args.data_override)

    config = OptunaRunConfig(
        stages=args.stages,
        n_trials=args.n_trials,
        metric=args.metric,
        direction=args.direction,
        data_overrides=data_overrides,
        stage_overrides=stage_overrides,
        sampler=args.sampler,
        pruner=args.pruner,
        storage=args.storage,
        study_name=args.study_name,
        show_progress=args.show_progress,
        output_dir=args.output_dir,
        save_figures=not args.no_figures,
    )
    results, cumulative = run_optuna(config)
    print("\nOptimisation terminée.")
    for stage, result in results.items():
        print(
            f"Stage {stage}: meilleur {args.metric}={result.study.best_value:.6f} (essais={len(result.study.trials)})"
        )
    if cumulative:
        print("\nOverrides cumulés pour les données:")
        print(json.dumps(cumulative, indent=2))


if __name__ == "__main__":
    main()
