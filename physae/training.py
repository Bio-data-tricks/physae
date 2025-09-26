"""High-level training utilities for staged PhysAE optimisation."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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
):
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
    _apply_stage_freeze(
        model,
        train_base=train_base,
        train_heads=train_heads,
        train_film=train_film,
        train_refiner=train_refiner,
        heads_subset=heads_subset,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=1,
        callbacks=callbacks or [],
    )
    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)
    return model


def train_stage_A(model, train_loader, val_loader, **kwargs):
    defaults = dict(
        stage_name="A",
        epochs=20,
        base_lr=2e-4,
        refiner_lr=1e-6,
        train_base=True,
        train_heads=True,
        train_film=False,
        train_refiner=False,
        refine_steps=0,
        delta_scale=0.1,
        use_film=False,
        film_subset=None,
        heads_subset=None,
        baseline_fix_enable=False,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)


def train_stage_B1(model, train_loader, val_loader, **kwargs):
    defaults = dict(
        stage_name="B1",
        epochs=12,
        base_lr=1e-6,
        refiner_lr=1e-5,
        train_base=False,
        train_heads=False,
        train_film=False,
        train_refiner=True,
        refine_steps=2,
        delta_scale=0.12,
        use_film=True,
        film_subset=["T"],
        heads_subset=None,
        baseline_fix_enable=False,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)


def train_stage_B2(model, train_loader, val_loader, **kwargs):
    defaults = dict(
        stage_name="B2",
        epochs=15,
        base_lr=3e-5,
        refiner_lr=3e-6,
        train_base=True,
        train_heads=True,
        train_film=True,
        train_refiner=True,
        refine_steps=2,
        delta_scale=0.08,
        use_film=True,
        film_subset=["P", "T"],
        heads_subset=None,
        baseline_fix_enable=False,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)
