"""Distributed Optuna orchestration for PhysAE hyper-parameter search.

This module launches Optuna workers that coordinate through Optuna's journal
storage (file based, SLURM friendly). Each worker executes the training
pipeline from :mod:`physae_train` while disabling checkpoints and visual
artefacts so the workflow remains filesystem-light and portable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import optuna
import torch
import pytorch_lightning as pl

import physae_train as pt


# --------------------------------------------------------------------------------------
# Signal handling for graceful shutdown / SLURM preemption
# --------------------------------------------------------------------------------------
_SHOULD_TERMINATE = False


def _signal_handler(signum: int, frame) -> None:  # type: ignore[override]
    global _SHOULD_TERMINATE
    _SHOULD_TERMINATE = True


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def _get_worker_rank() -> str:
    for key in ("SLURM_PROCID", "LOCAL_RANK", "RANK"):
        val = os.environ.get(key)
        if val is not None:
            return str(val)
    return str(os.getpid())


def _setup_logging(log_dir: Path, log_level: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_rank = _get_worker_rank()
    log_path = log_dir / f"worker_{worker_rank}.log"

    logger = logging.getLogger("optuna_phisae")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove pre-existing handlers to avoid duplicated logs on resume.
    if logger.hasHandlers():
        for hdl in list(logger.handlers):
            logger.removeHandler(hdl)
            hdl.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# --------------------------------------------------------------------------------------
# Serialization helper for Optuna user attrs
# --------------------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --------------------------------------------------------------------------------------
# Hyper-parameter search space definition
# --------------------------------------------------------------------------------------

def _suggest_hyperparameters(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Sample the search space for a given Optuna trial."""

    # Dataset & model construction hyperparameters.
    init_base_lr = trial.suggest_float("init_base_lr", 5e-5, 3e-4, log=True)
    init_refiner_lr = trial.suggest_float("init_refiner_lr", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24])

    hparams: Dict[str, Any] = {
        "seed_offset": trial.suggest_int("seed_offset", 0, 10_000),
        "build": {
            "batch_size": int(batch_size),
            "lrs": (init_base_lr, init_refiner_lr),
            "backbone_variant": trial.suggest_categorical("backbone_variant", ["s", "m"]),
            "refiner_variant": trial.suggest_categorical("refiner_variant", ["s", "m"]),
            "backbone_width_mult": trial.suggest_float("backbone_width_mult", 0.75, 1.25),
            "backbone_depth_mult": trial.suggest_float("backbone_depth_mult", 0.75, 1.25),
            "refiner_width_mult": trial.suggest_float("refiner_width_mult", 0.75, 1.5),
            "refiner_depth_mult": trial.suggest_float("refiner_depth_mult", 0.75, 1.5),
            "backbone_drop_path": trial.suggest_float("backbone_drop_path", 0.0, 0.2),
            "refiner_drop_path": trial.suggest_float("refiner_drop_path", 0.0, 0.2),
            "refiner_feature_pool": trial.suggest_categorical("refiner_feature_pool", ["avg", "avgmax"]),
            "refiner_shared_hidden_scale": trial.suggest_float(
                "refiner_shared_hidden_scale", 0.4, 1.0
            ),
            "huber_beta": trial.suggest_float("huber_beta", 5e-4, 3e-3, log=True),
        },
        "stage_A": {
            "epochs": trial.suggest_int("stageA_epochs", 15, 30),
            "base_lr": trial.suggest_float("stageA_base_lr", 5e-5, 3e-4, log=True),
            "refiner_lr": trial.suggest_float("stageA_refiner_lr", 1e-6, 5e-5, log=True),
        },
        "stage_B1": {
            "epochs": trial.suggest_int("stageB1_epochs", 8, 20),
            "refiner_lr": trial.suggest_float("stageB1_refiner_lr", 5e-5, 5e-4, log=True),
            "delta_scale": trial.suggest_float("stageB1_delta_scale", 0.05, 0.2),
            "refine_steps": trial.suggest_int("stageB1_refine_steps", 1, 3),
        },
        "stage_B2": {
            "epochs": trial.suggest_int("stageB2_epochs", 10, 25),
            "base_lr": trial.suggest_float("stageB2_base_lr", 1e-5, 1e-4, log=True),
            "refiner_lr": trial.suggest_float("stageB2_refiner_lr", 5e-6, 5e-5, log=True),
            "delta_scale": trial.suggest_float("stageB2_delta_scale", 0.05, 0.15),
            "refine_steps": trial.suggest_int("stageB2_refine_steps", 1, 3),
        },
    }
    return hparams


# --------------------------------------------------------------------------------------
# Lightning callback to capture validation metrics during stages
# --------------------------------------------------------------------------------------


class _MetricRecorder(pl.Callback):
    """Capture the latest value for a metric logged by Lightning."""

    def __init__(self, metric_name: str = "val_loss") -> None:
        super().__init__()
        self.metric_name = metric_name
        self.history: List[float] = []

    def _maybe_record(self, trainer: pl.Trainer) -> None:
        if trainer.sanity_checking:
            return
        if self.metric_name not in trainer.callback_metrics:
            return
        value = trainer.callback_metrics[self.metric_name]
        try:
            value_f = float(value.detach().cpu().item())  # type: ignore[attr-defined]
        except AttributeError:
            value_f = float(value)
        self.history.append(value_f)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._maybe_record(trainer)

    @property
    def last_value(self) -> Optional[float]:
        return self.history[-1] if self.history else None


# --------------------------------------------------------------------------------------
# Trial execution (single worker)
# --------------------------------------------------------------------------------------


def _set_global_seeds(base_seed: int) -> None:
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    pl.seed_everything(base_seed, workers=True)


def _ensure_no_termination() -> None:
    if _SHOULD_TERMINATE:
        raise optuna.TrialPruned("Termination requested via signal")


def _run_stages(
    trial: optuna.trial.Trial,
    model: pt.PhysicallyInformedAE,
    train_loader,
    val_loader,
    trainer_kwargs: Dict[str, Any],
    stage_params: Dict[str, Any],
    stage_name: str,
    logger: logging.Logger,
    step_index: int,
) -> Tuple[pt.PhysicallyInformedAE, Optional[float]]:
    """Run a single training stage and report metrics to Optuna."""

    _ensure_no_termination()
    metric_cb = _MetricRecorder("val_loss")
    callbacks = [
        pt.UpdateEpochInDataset(),
        pt.AdvanceDistributedSamplerEpoch(),
        metric_cb,
    ]
    stage_kwargs = dict(trainer_kwargs)
    stage_kwargs.update(stage_params)
    stage_kwargs.setdefault("callbacks", callbacks)
    stage_kwargs["callbacks"] = callbacks
    stage_kwargs.setdefault("ckpt_in", None)
    stage_kwargs.setdefault("ckpt_out", None)
    stage_kwargs.setdefault("enable_progress_bar", False)

    logger.info("Starting stage %s with params: %s", stage_name, json.dumps(stage_params))
    stage_fn = {
        "A": pt.train_stage_A,
        "B1": pt.train_stage_B1,
        "B2": pt.train_stage_B2,
    }[stage_name]
    model = stage_fn(model, train_loader, val_loader, **stage_kwargs)

    metric = metric_cb.last_value
    if metric is not None:
        logger.info("Stage %s finished with %s=%.6f", stage_name, metric_cb.metric_name, metric)
        trial.report(metric, step=step_index)
        if trial.should_prune():
            logger.warning("Trial %s pruned after stage %s", trial.number, stage_name)
            raise optuna.TrialPruned(f"Pruned at stage {stage_name}")
    else:
        logger.warning("Stage %s completed without metric '%s'", stage_name, metric_cb.metric_name)
    return model, metric


def _run_single_trial(
    trial: optuna.trial.Trial,
    args: argparse.Namespace,
    logger: logging.Logger,
    hparams: Dict[str, Any],
) -> float:
    _ensure_no_termination()

    seed_base = (args.seed or 0) + int(hparams.get("seed_offset", 0)) + trial.number
    _set_global_seeds(seed_base)
    torch.set_float32_matmul_precision("high")

    # Prepare trial-specific directories (no checkpoints or figures kept).
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trial_dir = output_root / f"trial_{trial.number:05d}"
    if trial_dir.exists():
        # Clean up any previous partial results.
        for item in trial_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                # We avoid recursive deletion to stay conservative.
                pass
    trial_dir.mkdir(exist_ok=True)

    logger.info("Running trial %s", trial.number)

    build_params = dict(hparams["build"])
    build_params.setdefault("seed", seed_base)
    if args.n_train is not None:
        build_params["n_train"] = int(args.n_train)
    if args.n_val is not None:
        build_params["n_val"] = int(args.n_val)
    if args.n_points is not None:
        build_params["n_points"] = int(args.n_points)

    trial.set_user_attr("hyperparameters", _to_jsonable(hparams))

    pt.get_master_addr_and_port()

    model, train_loader, val_loader = pt.build_data_and_model(**build_params)
    train_loader, val_loader = pt.ensure_distributed_samplers(train_loader, val_loader)

    trainer_kwargs = pt.trainer_common_kwargs()
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer_kwargs.update(
        {
            "accelerator": accelerator,
            "devices": 1,
            "num_nodes": 1,
            "logger": False,
            "default_root_dir": str(trial_dir),
            "enable_progress_bar": False,
        }
    )

    metrics: List[Tuple[str, Optional[float]]] = []

    model, metric_a = _run_stages(
        trial,
        model,
        train_loader,
        val_loader,
        trainer_kwargs,
        hparams["stage_A"],
        "A",
        logger,
        step_index=0,
    )
    metrics.append(("stage_A", metric_a))

    model, metric_b1 = _run_stages(
        trial,
        model,
        train_loader,
        val_loader,
        trainer_kwargs,
        hparams["stage_B1"],
        "B1",
        logger,
        step_index=1,
    )
    metrics.append(("stage_B1", metric_b1))

    model, metric_b2 = _run_stages(
        trial,
        model,
        train_loader,
        val_loader,
        trainer_kwargs,
        hparams["stage_B2"],
        "B2",
        logger,
        step_index=2,
    )
    metrics.append(("stage_B2", metric_b2))

    final_metric = metric_b2 if metric_b2 is not None else float("inf")

    for stage_name, value in metrics:
        if value is not None:
            trial.set_user_attr(f"metric_{stage_name}", value)
    trial.set_user_attr("seed", seed_base)

    # Explicitly clear CUDA cache between trials.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Trial %s completed with final metric %.6f", trial.number, final_metric)
    _ensure_no_termination()
    return final_metric


# --------------------------------------------------------------------------------------
# Optuna orchestration
# --------------------------------------------------------------------------------------


def _create_sampler(name: str, seed: Optional[int]) -> optuna.samplers.BaseSampler:
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    # Default to TPE sampler.
    return optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=8,
        seed=seed,
    )


def _create_pruner(name: str) -> optuna.pruners.BasePruner:
    if name == "nop":
        return optuna.pruners.NopPruner()
    if name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    return optuna.pruners.MedianPruner(n_warmup_steps=1)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna orchestration for PhysAE")
    parser.add_argument("--storage", required=True, help="Path to the Optuna journal file")
    parser.add_argument("--study-name", required=True, help="Optuna study name")
    parser.add_argument("--output-dir", default="./optuna_runs", help="Directory for per-trial outputs")
    parser.add_argument("--log-dir", default="./optuna_logs", help="Directory for worker logs")
    parser.add_argument("--sampler", default="tpe", choices=["tpe", "random"], help="Optuna sampler")
    parser.add_argument("--pruner", default="median", choices=["median", "nop", "successive_halving"], help="Optuna pruner")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--trials-per-worker", type=int, default=0, help="Maximum trials for this worker (0 = unlimited)")
    parser.add_argument("--max-trials", type=int, default=0, help="Global maximum number of completed trials (0 = unlimited)")
    parser.add_argument("--timeout", type=int, default=0, help="Global timeout in seconds (0 = none)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...)")
    parser.add_argument("--sleep-on-error", type=float, default=5.0,
                        help="Seconds to sleep before retrying after recoverable errors")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Override the number of training samples (default from physae_train)")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Override the number of validation samples")
    parser.add_argument("--n-points", type=int, default=None,
                        help="Override the number of spectral points per sample")
    parser.add_argument("--stageA-epochs", type=int, default=None,
                        help="Force the number of epochs for stage A")
    parser.add_argument("--stageB1-epochs", type=int, default=None,
                        help="Force the number of epochs for stage B1")
    parser.add_argument("--stageB2-epochs", type=int, default=None,
                        help="Force the number of epochs for stage B2")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    log_dir = Path(args.log_dir)
    logger = _setup_logging(log_dir, args.log_level)
    logger.info("Worker starting with PID %s", os.getpid())

    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(storage_path)))

    sampler = _create_sampler(args.sampler, args.seed)
    pruner = _create_pruner(args.pruner)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    callbacks: List[Any] = []
    if args.max_trials and args.max_trials > 0:
        callbacks.append(
            optuna.study.MaxTrialsCallback(
                args.max_trials,
                states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
            )
        )

    trials_limit = args.trials_per_worker if args.trials_per_worker > 0 else None
    timeout = args.timeout if args.timeout and args.timeout > 0 else None

    def _objective(trial: optuna.trial.Trial) -> float:
        _ensure_no_termination()
        params = _suggest_hyperparameters(trial)
        if args.stageA_epochs is not None:
            params["stage_A"]["epochs"] = int(args.stageA_epochs)
        if args.stageB1_epochs is not None:
            params["stage_B1"]["epochs"] = int(args.stageB1_epochs)
        if args.stageB2_epochs is not None:
            params["stage_B2"]["epochs"] = int(args.stageB2_epochs)
        try:
            return _run_single_trial(trial, args, logger, params)
        except optuna.TrialPruned:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            logger.exception("CUDA OOM on trial %s", trial.number)
            raise optuna.TrialPruned("CUDA OOM") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error on trial %s: %s", trial.number, exc)
            time.sleep(max(0.0, float(args.sleep_on_error)))
            raise

    try:
        study.optimize(
            _objective,
            n_trials=trials_limit,
            timeout=timeout,
            callbacks=callbacks,
        )
    except optuna.exceptions.OptunaError as exc:
        logger.error("Optuna error: %s", exc)
        return 1
    finally:
        logger.info("Worker shutting down")

    if study.best_trial is not None:
        logger.info("Best trial so far: #%s with value %.6f", study.best_trial.number, study.best_trial.value)

    return 0


if __name__ == "__main__":
    sys.exit(main())
