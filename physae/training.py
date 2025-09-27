"""High-level training utilities for staged PhysAE optimisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .config_loader import load_stage_config, merge_dicts
from .model import PhysicallyInformedAE


def _freeze_all(model: PhysicallyInformedAE) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _set_trainable_heads(model: PhysicallyInformedAE, names: Optional[List[str]]) -> None:
    if model.head_mode == "single":
        for param in model.out_head.parameters():
            param.requires_grad_(True)
        return
    desired = set(names) if names is not None else set(model.predict_params)
    for name, head in model.out_heads.items():
        required = name in desired
        for param in head.parameters():
            param.requires_grad_(required)


def _apply_stage_freeze(
    model: PhysicallyInformedAE,
    *,
    train_base: bool,
    train_heads: bool,
    train_film: bool,
    train_refiner: bool,
    heads_subset: Optional[List[str]],
) -> None:
    _freeze_all(model)
    if train_base:
        for module in (model.backbone, model.shared_head):
            for param in module.parameters():
                param.requires_grad_(True)
    if model.head_mode == "single":
        if train_heads:
            for param in model.out_head.parameters():
                param.requires_grad_(True)
    else:
        _set_trainable_heads(model, heads_subset if train_heads else [])
    if model.film is not None and train_film:
        for param in model.film.parameters():
            param.requires_grad_(True)
    if train_refiner:
        for param in model.refiner.parameters():
            param.requires_grad_(True)


def _load_weights_if_any(model: PhysicallyInformedAE, ckpt_in: Optional[str]) -> None:
    if ckpt_in:
        state = torch.load(ckpt_in, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ weights chargés depuis: {ckpt_in}")


def _save_checkpoint(trainer: pl.Trainer, ckpt_out: Optional[str]) -> None:
    if ckpt_out:
        trainer.save_checkpoint(ckpt_out)
        print(f"✓ checkpoint sauvegardé: {ckpt_out}")


def train_stage_custom(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    stage_name: str,
    epochs: int,
    base_lr: float,
    refiner_lr: float,
    train_base: bool,
    train_heads: bool,
    train_film: bool,
    train_refiner: bool,
    refine_steps: int,
    delta_scale: float,
    use_film: Optional[bool] = None,
    film_subset: Optional[List[str]] = None,
    heads_subset: Optional[List[str]] = None,
    baseline_fix_enable: Optional[bool] = None,
    callbacks: Optional[list] = None,
    accelerator: Optional[str] = None,
    ckpt_in: Optional[str] = None,
    ckpt_out: Optional[str] = None,
    enable_progress_bar: bool = False,
    optimizer: Optional[str] = None,
    optimizer_weight_decay: Optional[float] = None,
    optimizer_beta1: Optional[float] = None,
    optimizer_beta2: Optional[float] = None,
    scheduler_eta_min: Optional[float] = None,
    scheduler_T_max: Optional[int] = None,
    return_metrics: bool = False,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> PhysicallyInformedAE | Tuple[PhysicallyInformedAE, Dict[str, float]]:
    print(f"\n===== Stage {stage_name} =====")
    _load_weights_if_any(model, ckpt_in)
    try:
        from pytorch_lightning.callbacks.progress import TQDMProgressBar

        if callbacks is not None and not enable_progress_bar:
            callbacks = [cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)]
    except Exception:
        pass
    model.base_lr = float(base_lr)
    model.refiner_lr = float(refiner_lr)
    model.set_stage_mode(stage_name, refine_steps=refine_steps, delta_scale=delta_scale)
    if use_film is not None:
        model.set_film_usage(bool(use_film))
    if film_subset is not None:
        model.set_film_subset(film_subset)
    if baseline_fix_enable is not None:
        model.baseline_fix_enable = bool(baseline_fix_enable)
    if optimizer is not None:
        model.optimizer_name = str(optimizer).lower()
    if optimizer_weight_decay is not None:
        model.optimizer_weight_decay = float(optimizer_weight_decay)
    beta1, beta2 = model.optimizer_betas
    if optimizer_beta1 is not None:
        beta1 = float(optimizer_beta1)
    if optimizer_beta2 is not None:
        beta2 = float(optimizer_beta2)
    model.optimizer_betas = (beta1, beta2)
    if scheduler_eta_min is not None:
        model.scheduler_eta_min = float(scheduler_eta_min)
    if scheduler_T_max is not None:
        model.scheduler_T_max = int(scheduler_T_max)
    _apply_stage_freeze(
        model,
        train_base=train_base,
        train_heads=train_heads,
        train_film=train_film,
        train_refiner=train_refiner,
        heads_subset=heads_subset,
    )
    resolved_accelerator = accelerator or ("gpu" if torch.cuda.is_available() else "cpu")
    trainer_args: Dict[str, Any] = {
        "max_epochs": epochs,
        "accelerator": resolved_accelerator,
        "enable_progress_bar": enable_progress_bar,
        "log_every_n_steps": 1,
        "callbacks": callbacks or [],
    }
    if trainer_kwargs:
        trainer_args.update(trainer_kwargs)
    default_root_dir = trainer_args.get("default_root_dir")
    if default_root_dir is not None:
        trainer_args["default_root_dir"] = str(Path(default_root_dir))
    if "devices" not in trainer_args and resolved_accelerator != "cpu":
        trainer_args["devices"] = trainer_args.get("devices", "auto")
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)
    if return_metrics:
        metrics: Dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        return model, metrics
    return model


def _prepare_stage_arguments(stage: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = dict(overrides or {})
    config_path = overrides.pop("config_path", None)
    stage_overrides = overrides.pop("config_overrides", None)
    stage_config = load_stage_config(stage, path=config_path)
    stage_config = merge_dicts(stage_config, stage_overrides)
    stage_config = merge_dicts(stage_config, overrides)
    stage_config.pop("optuna", None)
    return stage_config


def train_stage_A(model, train_loader, val_loader, **kwargs):
    params = _prepare_stage_arguments("A", kwargs)
    return train_stage_custom(model, train_loader, val_loader, **params)


def train_stage_B1(model, train_loader, val_loader, **kwargs):
    params = _prepare_stage_arguments("B1", kwargs)
    return train_stage_custom(model, train_loader, val_loader, **params)


def train_stage_B2(model, train_loader, val_loader, **kwargs):
    params = _prepare_stage_arguments("B2", kwargs)
    return train_stage_custom(model, train_loader, val_loader, **params)
