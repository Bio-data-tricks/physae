"""Training entry-point mirroring the original ``physae.py`` workflow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.data_config import (  # noqa: E402
    load_noise_profile,
    load_parameter_ranges,
    load_transitions,
)
from config.training_config import TrainingConfig  # noqa: E402
from physics.tips import Tips2021QTpy, find_qtpy_dir  # noqa: E402
from training.callbacks import (  # noqa: E402
    LossCurvePlotCallback,
    PT_PredVsExp_VisuCallback,
    StageAwarePlotCallback,
    UpdateEpochInDataset,
    UpdateEpochInValDataset,
)
from training.factory import build_data_and_model  # noqa: E402
from training.stages import (  # noqa: E402
    train_refiner_idx,
    train_stage_A,
    train_stage_B1,
    train_stage_B2,
    train_stage_DENOISER,
)


def _default_transitions_path() -> Path:
    return REPO_ROOT / "config" / "data" / "transitions_sample.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PhysicallyInformedAE with stage-aware defaults",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage configuration
    parser.add_argument(
        "--stage",
        choices=["A", "B1", "B2", "DEN", "REFINER"],
        default="A",
        help="Training stage to execute",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the default number of epochs for the selected stage",
    )
    parser.add_argument(
        "--refiner-index",
        type=int,
        default=0,
        help="Index of the refiner block to fine-tune when stage=REFINER",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=0.12,
        help="Residual scaling factor for refiner fine-tuning",
    )
    parser.add_argument(
        "--stage-a-exp-params",
        nargs="*",
        default=None,
        help="Parameters kept at experimental ground truth during Stage A",
    )
    parser.add_argument(
        "--refiner-exp-params",
        nargs="*",
        default=None,
        help="Parameters treated as experimental during refiner stages",
    )

    # Dataset configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-points", type=int, default=800, help="Spectral points")
    parser.add_argument("--train-samples", type=int, default=500_000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=5_000, help="Validation samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (None reproduces physae.py heuristic)",
    )
    parser.add_argument(
        "--static-train-params",
        action="store_true",
        help="Freeze parameter draws for the training dataset",
    )
    parser.add_argument(
        "--static-val-params",
        action="store_true",
        help="Freeze parameter draws for the validation dataset",
    )
    parser.add_argument(
        "--freeze-val-noise",
        action="store_true",
        help="Keep validation noise realisations fixed across epochs",
    )

    # Configuration files
    parser.add_argument(
        "--train-parameters-config",
        type=str,
        help="YAML file overriding training parameter ranges",
    )
    parser.add_argument(
        "--val-parameters-config",
        type=str,
        help="YAML file overriding validation parameter ranges",
    )
    parser.add_argument(
        "--noise-train-config",
        type=str,
        help="YAML file overriding training noise profile",
    )
    parser.add_argument(
        "--noise-val-config",
        type=str,
        help="YAML file overriding validation noise profile",
    )
    parser.add_argument(
        "--transitions-config",
        type=str,
        help="YAML file describing spectroscopic transitions",
    )
    parser.add_argument(
        "--qtpy-dir",
        type=str,
        default=None,
        help="Directory containing QTpy partition function tables",
    )

    # Logging & checkpoints
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory used by ModelCheckpoint",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Root directory for diagnostic figures",
    )
    parser.add_argument("--ckpt-in", type=str, help="Checkpoint to load before training")
    parser.add_argument(
        "--ckpt-out",
        type=str,
        help="Checkpoint path saved after training (in addition to callbacks)",
    )

    # Trainer arguments
    parser.add_argument("--accelerator", type=str, help="Lightning accelerator override")
    parser.add_argument("--devices", type=int, help="Number of devices to use")
    parser.add_argument("--precision", type=str, help="Numerical precision override")
    parser.add_argument("--strategy", type=str, help="Training strategy (ddp, etc.)")
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        help="Limit number of training batches (debug)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        help="Limit number of validation batches (debug)",
    )
    parser.add_argument(
        "--enable-progress-bar",
        action="store_true",
        help="Display Lightning progress bar",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic training",
    )

    return parser.parse_args()


def _load_parameter_ranges(path: Optional[str]) -> Optional[Dict[str, tuple[float, float]]]:
    if path is None:
        return None
    return load_parameter_ranges(path, update_globals=False)


def _load_noise(path: Optional[str]) -> Optional[Dict[str, object]]:
    if path is None:
        return None
    return load_noise_profile(path)


def _load_transitions(path: Optional[str]) -> tuple[Dict[str, Iterable], Dict[str, Iterable]]:
    cfg_path = Path(path) if path else _default_transitions_path()
    return load_transitions(cfg_path, include_poly_freq=True)


def _create_callbacks(
    stage: str,
    model,
    val_loader,
    log_dir: Path,
    checkpoint_dir: Path,
) -> list:
    stage_tag = f"Stage{stage.upper()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "loss_curves").mkdir(parents=True, exist_ok=True)
    stage_fig_dir = log_dir / "figures" / stage_tag.lower()
    stage_fig_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / stage_tag.lower()).mkdir(parents=True, exist_ok=True)

    callbacks = [
        StageAwarePlotCallback(
            val_loader=val_loader,
            param_names=model.param_names,
            num_examples=1,
            save_dir=stage_fig_dir,
            stage_tag=stage_tag,
            refine=stage.upper() not in {"A", "DEN"},
            use_gt_for_provided=True,
            recon_PT="pred",
            max_val_batches=10,
        ),
        PT_PredVsExp_VisuCallback(
            val_loader=val_loader,
            save_dir=stage_fig_dir / "pt_compare",
            num_examples=1,
            tag=f"{stage_tag}_PT",
            use_gt_for_provided=True,
        ),
        LossCurvePlotCallback(
            save_path=log_dir / "loss_curves" / f"{stage_tag.lower()}_curves.png"
        ),
        UpdateEpochInDataset(),
        UpdateEpochInValDataset(),
        ModelCheckpoint(
            dirpath=checkpoint_dir / stage_tag.lower(),
            filename=f"physae-{stage_tag.lower()}-{{epoch:03d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return callbacks


def _trainer_kwargs_from_args(args: argparse.Namespace) -> dict:
    trainer_kwargs: Dict[str, object] = {}
    for key in ("accelerator", "devices", "precision", "strategy"):
        value = getattr(args, key)
        if value is not None:
            trainer_kwargs[key] = value
    for key in ("limit_train_batches", "limit_val_batches"):
        value = getattr(args, key)
        if value is not None:
            trainer_kwargs[key] = value
    if args.deterministic:
        trainer_kwargs["deterministic"] = True
    return trainer_kwargs


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed)

    config = TrainingConfig(
        seed=args.seed,
        n_points=args.num_points,
        n_train=args.train_samples,
        n_val=args.val_samples,
        batch_size=args.batch_size,
    )

    train_ranges = _load_parameter_ranges(args.train_parameters_config)
    if train_ranges is not None:
        config.train_ranges = train_ranges
    val_ranges = _load_parameter_ranges(args.val_parameters_config)
    if val_ranges is not None:
        config.val_ranges = val_ranges

    noise_train = _load_noise(args.noise_train_config)
    if noise_train is not None:
        config.noise_train = noise_train
    noise_val = _load_noise(args.noise_val_config)
    if noise_val is not None:
        config.noise_val = noise_val

    transitions_dict, poly_freq_map = _load_transitions(args.transitions_config)
    poly_freq_CH4 = poly_freq_map.get("CH4")

    tipspy = None
    try:
        qtpy_dir = find_qtpy_dir(args.qtpy_dir or "./QTpy")
        tipspy = Tips2021QTpy(qtpy_dir, device="cpu")
        print(f"✓ TIPS2021 chargé depuis: {qtpy_dir}")
    except Exception as exc:  # pragma: no cover - best effort informative message
        print(f"⚠️ Impossible de charger TIPS2021 ({exc}). La simulation physique utilisera None.")

    model, train_loader, val_loader = build_data_and_model(
        config,
        transitions_dict=transitions_dict,
        poly_freq_CH4=poly_freq_CH4,
        tipspy=tipspy,
        freeze_train_parameters=args.static_train_params,
        freeze_val_parameters=args.static_val_params,
        freeze_val_noise=args.freeze_val_noise,
        num_workers=args.num_workers,
        stage=args.stage,
    )

    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    callbacks = _create_callbacks(args.stage, model, val_loader, log_dir, checkpoint_dir)

    stage_kwargs = dict(
        callbacks=callbacks,
        enable_progress_bar=args.enable_progress_bar,
        ckpt_in=args.ckpt_in,
        ckpt_out=args.ckpt_out,
        **_trainer_kwargs_from_args(args),
    )
    if args.epochs is not None:
        stage_kwargs["epochs"] = args.epochs

    stage = args.stage.upper()
    if stage == "A":
        if args.stage_a_exp_params:
            stage_kwargs["use_exp_params"] = args.stage_a_exp_params
        train_stage_A(model, train_loader, val_loader, **stage_kwargs)
    elif stage == "B1":
        train_stage_B1(model, train_loader, val_loader, **stage_kwargs)
    elif stage == "B2":
        train_stage_B2(model, train_loader, val_loader, **stage_kwargs)
    elif stage == "DEN":
        train_stage_DENOISER(model, train_loader, val_loader, **stage_kwargs)
    else:
        if args.refiner_exp_params:
            stage_kwargs["use_exp_params_for_resid"] = args.refiner_exp_params
        stage_kwargs.setdefault("refiner_lr", 1e-4)
        stage_kwargs.setdefault("delta_scale", args.delta_scale)
        train_refiner_idx(
            model,
            train_loader,
            val_loader,
            k=int(args.refiner_index),
            **stage_kwargs,
        )

    print("✅ Entraînement terminé")


if __name__ == "__main__":
    main()
