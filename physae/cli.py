"""Command-line interface to orchestrate optimisation, training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import optuna
import pandas as pd
import torch

from .config_loader import load_yaml_file, merge_dicts
from .factory import build_data_and_model
from .normalization import unnorm_param_torch
from .pipeline import expand_stages, optimise_stages, train_stages


def _load_mapping(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    data = load_yaml_file(path)
    if not isinstance(data, Mapping):
        raise TypeError(f"Le fichier {path} doit contenir un dictionnaire.")
    return dict(data)


def _resolve_sampler(name: Optional[str]) -> optuna.samplers.BaseSampler | None:
    if not name:
        return None
    name = name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler()
    if name == "random":
        return optuna.samplers.RandomSampler()
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler()
    raise ValueError(f"Sampler inconnu: {name}")


def _resolve_pruner(name: Optional[str]) -> optuna.pruners.BasePruner | None:
    if not name:
        return None
    name = name.lower()
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "successivehalving":
        return optuna.pruners.SuccessiveHalvingPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Pruner inconnu: {name}")


def _stage_dir(root: Path, stage: str) -> Path:
    return root / f"stage_{stage.upper()}"


def _save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _parse_devices(value: Optional[str]) -> Any:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.lower() == "auto":
        return "auto"
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        try:
            return [int(part) for part in parts]
        except ValueError:
            return parts
    try:
        return int(value)
    except ValueError:
        return value


def cmd_optimise(args: argparse.Namespace) -> None:
    data_overrides = _load_mapping(args.data_overrides)
    stage_overrides = _load_mapping(args.stage_overrides) if args.stage_overrides else None
    sampler = _resolve_sampler(args.sampler)
    pruner = _resolve_pruner(args.pruner)
    results, cumulative_overrides = optimise_stages(
        args.stages,
        n_trials=args.n_trials,
        metric=args.metric,
        direction=args.direction,
        data_config_path=args.data_config_path,
        data_config_name=args.data_config_name,
        data_overrides=data_overrides,
        stage_config_path=args.stage_config_path,
        stage_overrides=stage_overrides,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
        show_progress_bar=args.show_progress,
        output_dir=args.output_dir,
        save_figures=not args.no_figures,
    )
    for stage, result in results.items():
        print(
            f"Stage {stage}: best {args.metric}={result.study.best_value:.6f} | "
            f"params enregistrés dans {result.artifact_dir or 'N/A'}"
        )
    if args.output_dir:
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        _save_json(root / "cumulative_data_overrides.json", cumulative_overrides)
    print("Optimisation terminée.")


def _load_stage_results(studies_dir: Path, stages: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    params: Dict[str, Dict[str, Any]] = {}
    for stage in expand_stages(stages):
        stage_path = _stage_dir(studies_dir, stage)
        cfg_path = stage_path / "best_stage_params.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Paramètres optimisés introuvables pour {stage} ({cfg_path})")
        params[stage] = load_yaml_file(cfg_path)
    return params


def _load_stage_data_overrides(studies_dir: Path, stages: Iterable[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for stage in expand_stages(stages):
        stage_path = _stage_dir(studies_dir, stage)
        data_path = stage_path / "best_data_overrides.yaml"
        if data_path.exists():
            overrides = merge_dicts(overrides, load_yaml_file(data_path))
    return overrides


def cmd_train(args: argparse.Namespace) -> None:
    if not args.studies_dir:
        raise ValueError("--studies-dir est requis pour l'entraînement.")
    studies_dir = Path(args.studies_dir)
    stage_params = _load_stage_results(studies_dir, args.stages)
    data_overrides = _load_stage_data_overrides(studies_dir, args.stages)
    extra_overrides = _load_mapping(args.data_overrides)
    if extra_overrides:
        data_overrides = merge_dicts(data_overrides, extra_overrides)
    trainer_kwargs = _load_mapping(args.trainer_kwargs)
    if args.accelerator:
        trainer_kwargs["accelerator"] = args.accelerator
    devices = _parse_devices(args.devices)
    if devices is not None:
        trainer_kwargs["devices"] = devices
    if args.num_nodes:
        trainer_kwargs["num_nodes"] = args.num_nodes
    if args.strategy:
        trainer_kwargs["strategy"] = args.strategy
    if args.precision:
        trainer_kwargs["precision"] = args.precision
    train_result = train_stages(
        args.stages,
        stage_params=stage_params,
        data_config_path=args.data_config_path,
        data_config_name=args.data_config_name,
        data_overrides=data_overrides,
        ckpt_dir=args.ckpt_dir,
        trainer_kwargs=trainer_kwargs,
        fine_tune_only=args.fine_tune_only,
    )
    print("Entraînement terminé. Statistiques:")
    for stage, stage_metrics in train_result.metrics.items():
        print(f"  - {stage}: {json.dumps(stage_metrics, indent=2)}")
    if args.metrics_out:
        _save_json(Path(args.metrics_out), train_result.metrics)
    if train_result.last_checkpoint is not None:
        print(f"Dernier checkpoint: {train_result.last_checkpoint}")


def cmd_infer(args: argparse.Namespace) -> None:
    overrides = _load_mapping(args.data_overrides)
    model, (_, val_loader), _ = build_data_and_model(
        config_path=args.data_config_path,
        config_name=args.data_config_name,
        config_overrides=overrides or None,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, Mapping) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    eps = float(args.eps)
    predictions = []
    param_names = list(getattr(model, "predict_params", []))
    if not param_names:
        raise RuntimeError("Le modèle ne définit pas de paramètres prédictibles.")
    batch_limit = args.batch_limit if args.batch_limit is not None else float("inf")
    batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= batch_limit:
                break
            noisy = batch["noisy_spectra"].to(device)
            params_true = batch["params"].to(device)
            provided = {name: params_true[:, model.name_to_idx[name]] for name in model.provided_params}
            outputs = model.infer(noisy, provided, refine=not args.no_refine)
            pred = outputs["y_phys_full"].detach().cpu()
            true = torch.stack(
                [unnorm_param_torch(name, params_true[:, model.name_to_idx[name]]) for name in param_names],
                dim=1,
            ).cpu()
            denom = torch.clamp(true.abs(), min=eps)
            err_pct = 100.0 * (pred - true).abs() / denom
            for i in range(pred.size(0)):
                row: Dict[str, Any] = {"sample": len(predictions)}
                for j, name in enumerate(param_names):
                    row[f"{name}_pred"] = float(pred[i, j])
                    row[f"{name}_true"] = float(true[i, j])
                    row[f"{name}_errpct"] = float(err_pct[i, j])
                predictions.append(row)
            batches += 1
            if args.max_batches is not None and batches >= args.max_batches:
                break
    if not predictions:
        print("Aucune prédiction générée (vérifier batch_limit / max_batches).");
        return
    df = pd.DataFrame(predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    summary = {
        name: {
            "mean_errpct": float(df[f"{name}_errpct"].mean()),
            "median_errpct": float(df[f"{name}_errpct"].median()),
        }
        for name in param_names
    }
    if args.metrics_out:
        _save_json(Path(args.metrics_out), summary)
    print(f"Prédictions sauvegardées dans {output_path}")
    for name, stats in summary.items():
        print(f"  - {name}: mean={stats['mean_errpct']:.3f}% | median={stats['median_errpct']:.3f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Outils de pilotage PhysAE")
    subparsers = parser.add_subparsers(dest="command", required=True)

    opt_parser = subparsers.add_parser("optimise", help="Optimiser un ou plusieurs stages avec Optuna")
    opt_parser.add_argument("--stages", nargs="+", default=["A"], help="Stages à optimiser (A, B, B1, B2, ALL)")
    opt_parser.add_argument("--n-trials", type=int, default=20)
    opt_parser.add_argument("--metric", default="val_loss")
    opt_parser.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    opt_parser.add_argument("--data-config-path")
    opt_parser.add_argument("--data-config-name", default="default")
    opt_parser.add_argument("--data-overrides")
    opt_parser.add_argument("--stage-config-path")
    opt_parser.add_argument("--stage-overrides")
    opt_parser.add_argument("--sampler", help="Sampler Optuna (tpe, random, cmaes)")
    opt_parser.add_argument("--pruner", help="Pruner Optuna (median, successivehalving, hyperband)")
    opt_parser.add_argument("--storage")
    opt_parser.add_argument("--study-name")
    opt_parser.add_argument("--show-progress", action="store_true")
    opt_parser.add_argument("--output-dir")
    opt_parser.add_argument("--no-figures", action="store_true", help="Désactiver la génération des figures Optuna")
    opt_parser.set_defaults(func=cmd_optimise)

    train_parser = subparsers.add_parser("train", help="Réentraîner les stages avec les meilleurs hyperparamètres")
    train_parser.add_argument("--stages", nargs="+", default=["A", "B"], help="Stages à entraîner")
    train_parser.add_argument("--studies-dir", required=True, help="Répertoire contenant les résultats Optuna")
    train_parser.add_argument("--data-config-path")
    train_parser.add_argument("--data-config-name", default="default")
    train_parser.add_argument("--data-overrides")
    train_parser.add_argument("--trainer-kwargs", help="Fichier YAML de configuration additionnelle du Trainer")
    train_parser.add_argument("--accelerator")
    train_parser.add_argument("--devices")
    train_parser.add_argument("--num-nodes", type=int)
    train_parser.add_argument("--strategy")
    train_parser.add_argument("--precision")
    train_parser.add_argument("--ckpt-dir")
    train_parser.add_argument("--metrics-out")
    train_parser.add_argument("--fine-tune-only", action="store_true", help="N'exécuter que le fine-tuning global (B2)")
    train_parser.set_defaults(func=cmd_train)

    infer_parser = subparsers.add_parser("infer", help="Réaliser de l'inférence à partir d'un checkpoint entraîné")
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument("--metrics-out")
    infer_parser.add_argument("--device")
    infer_parser.add_argument("--data-config-path")
    infer_parser.add_argument("--data-config-name", default="default")
    infer_parser.add_argument("--data-overrides")
    infer_parser.add_argument("--no-refine", action="store_true")
    infer_parser.add_argument("--eps", type=float, default=1e-12)
    infer_parser.add_argument("--batch-limit", type=int, help="Nombre maximum de batchs à traiter")
    infer_parser.add_argument("--max-batches", type=int, help="Limite stricte en nombre de batchs")
    infer_parser.set_defaults(func=cmd_infer)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

