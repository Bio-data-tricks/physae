#!/usr/bin/env python
"""Simple command-line helper to train PhysAE stages with explicit parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from physae.parameter_catalog import list_data_parameters, list_stage_parameters, set_by_path
from physae.simple_workflows import StageRunConfig, run_single_stage


def _parse_overrides(values: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Format invalide pour l'override: {item!r} (attendu cle=valeur)")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(f"Clé vide dans l'override: {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        set_by_path(overrides, key, value)
    return overrides


def _print_parameters(stage: str) -> None:
    print("\nParamètres des données:")
    for info in list_data_parameters():
        print(f"  - {info.key} (défaut={info.default!r}, type={info.value_type})")
    print(f"\nParamètres du stage {stage}:")
    for info in list_stage_parameters(stage):
        print(f"  - {info.key} (défaut={info.default!r}, type={info.value_type})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["A", "B1", "B2"], help="Stage à entraîner")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Répertoire de sortie des checkpoints")
    parser.add_argument("--ckpt-in", type=str, default=None, help="Checkpoint à charger avant l'entraînement")
    parser.add_argument("--ckpt-out", type=str, default=None, help="Chemin de sauvegarde du checkpoint final")
    parser.add_argument(
        "--data-override",
        action="append",
        default=[],
        metavar="cle=valeur",
        help="Override pour le YAML des données (utiliser JSON pour les valeurs complexes)",
    )
    parser.add_argument(
        "--stage-override",
        action="append",
        default=[],
        metavar="cle=valeur",
        help="Override pour la configuration du stage",
    )
    parser.add_argument(
        "--trainer-arg",
        action="append",
        default=[],
        metavar="cle=valeur",
        help="Arguments supplémentaires passés au Trainer Lightning",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="Affiche tous les paramètres modifiables et quitte",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_params:
        _print_parameters(args.stage)
        return

    data_overrides = _parse_overrides(args.data_override)
    stage_overrides = _parse_overrides(args.stage_override)
    trainer_kwargs = _parse_overrides(args.trainer_arg)

    config = StageRunConfig(
        stage=args.stage,
        data_overrides=data_overrides,
        stage_overrides=stage_overrides,
        trainer_kwargs=trainer_kwargs,
        ckpt_dir=args.ckpt_dir,
        ckpt_in=args.ckpt_in,
        ckpt_out=args.ckpt_out,
    )
    metrics, last_ckpt = run_single_stage(config)
    print("\nEntraînement terminé. Métriques:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")
    if last_ckpt is not None:
        print(f"Dernier checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
