"""Training stage utilities for :class:`PhysicallyInformedAE`."""
from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from models.autoencoder import PhysicallyInformedAE
from losses.peak_weighted import PeakWeightedMSELoss


def _freeze_all(model: PhysicallyInformedAE) -> None:
    """Disable gradients for every parameter of ``model``."""
    for param in model.parameters():
        param.requires_grad_(False)


def _set_trainable_heads(
    model: PhysicallyInformedAE, names: Optional[List[str]]
) -> None:
    """Enable gradients for the requested prediction heads."""
    if model.head_mode == "single":
        for param in model.out_head.parameters():
            param.requires_grad_(True)
        return

    wanted = set(names) if names is not None else set(model.predict_params)
    for name, head in model.out_heads.items():
        trainable = name in wanted
        for param in head.parameters():
            param.requires_grad_(trainable)


def _apply_stage_freeze(
    model: PhysicallyInformedAE,
    train_base: bool,
    train_heads: bool,
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

    if train_refiner:
        for refiner in model.refiners:
            for param in refiner.parameters():
                param.requires_grad_(True)


def _load_weights_if_any(model: PhysicallyInformedAE, ckpt_in: Optional[str]) -> None:
    if not ckpt_in:
        return

    state = torch.load(ckpt_in, map_location="cpu")
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ weights chargés depuis: {ckpt_in}")


def _save_checkpoint(trainer: pl.Trainer, ckpt_out: Optional[str]) -> None:
    if not ckpt_out:
        return

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
    train_refiner: bool,
    refine_steps: int,
    delta_scale: float,
    heads_subset: Optional[List[str]] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    **trainer_kwargs,
) -> PhysicallyInformedAE:
    """Generic helper handling the shared boilerplate for B stages."""

    print(f"\n===== Stage {stage_name} =====")

    ckpt_in = trainer_kwargs.pop("ckpt_in", None)
    ckpt_out = trainer_kwargs.pop("ckpt_out", None)
    _load_weights_if_any(model, ckpt_in)

    if callbacks is not None and enable_progress_bar is False:
        try:
            from pytorch_lightning.callbacks.progress import TQDMProgressBar

            callbacks = [
                cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)
            ]
        except Exception:
            pass

    model.base_lr = float(base_lr)
    model.refiner_lr = float(refiner_lr)
    model.set_stage_mode(
        stage_name,
        refine_steps=int(refine_steps),
        delta_scale=float(delta_scale),
    )

    _apply_stage_freeze(
        model,
        train_base=train_base,
        train_heads=train_heads,
        train_refiner=train_refiner,
        heads_subset=heads_subset,
    )

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    from pytorch_lightning.strategies import Strategy

    strategy = trainer_kwargs.get("strategy")
    if isinstance(strategy, Strategy):
        trainer_kwargs.pop("accelerator", None)

    for env_key in ("PL_TRAINER_ACCELERATOR", "PL_TRAINER_MAX_EPOCHS"):
        import os

        os.environ.pop(env_key, None)

    trainer = pl.Trainer(
        max_epochs=int(epochs),
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)
    return model


def train_stage_DENOISER(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 30,
    denoiser_lr: float = 1e-4,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    ckpt_in: Optional[str] = None,
    ckpt_out: Optional[str] = None,
    **trainer_kwargs,
) -> PhysicallyInformedAE:
    if ckpt_in:
        state = torch.load(ckpt_in, map_location="cpu")
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ weights chargés depuis: {ckpt_in}")

    model.use_denoiser = True
    model.denoiser_lr = float(denoiser_lr)
    model.set_stage_mode("DEN", refine_steps=0, delta_scale=None)

    if callbacks is not None and enable_progress_bar is False:
        try:
            from pytorch_lightning.callbacks.progress import TQDMProgressBar

            callbacks = [
                cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)
            ]
        except Exception:
            pass

    from pytorch_lightning.strategies import Strategy

    if isinstance(trainer_kwargs.get("strategy"), Strategy):
        trainer_kwargs.pop("accelerator", None)

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    trainer = pl.Trainer(
        max_epochs=int(epochs),
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )
    trainer.fit(model, train_loader, val_loader)

    if ckpt_out:
        trainer.save_checkpoint(ckpt_out)
        print(f"✓ checkpoint sauvegardé: {ckpt_out}")

    return model


def train_stage_A(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 70,
    base_lr: float = 1e-4,
    heads_subset: Optional[List[str]] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    ckpt_in: Optional[str] = None,
    ckpt_out: Optional[str] = None,
    use_exp_params: Optional[List[str]] = None,
    **trainer_kwargs,
) -> PhysicallyInformedAE:
    _load_weights_if_any(model, ckpt_in)

    model.base_lr = float(base_lr)
    model.refiner_lr = 0.0
    model.set_stage_mode("A", refine_steps=0, delta_scale=None)

    model._use_exp_params_in_A = list(use_exp_params) if use_exp_params else []

    n_active = len(
        [p for p in model.predict_params if p not in model._use_exp_params_in_A]
    )
    if model._use_exp_params_in_A:
        print("✅ Mode A avec paramètres expérimentaux:")
        print(f"   → Physique utilise {model._use_exp_params_in_A} GT")
        print(
            f"   → Têtes {model._use_exp_params_in_A}: PAS de loss (pas entraînées)"
        )
        print(f"   → Focus sur {n_active} autres paramètres")
        print(
            "   → ReLoBRaLo sera réinitialisé automatiquement au premier batch"
        )
    else:
        print("✅ Mode A standard:")
        print(
            f"   → Tous les {len(model.predict_params)} paramètres prédits et entraînés"
        )

    def freeze_module(module, trainable: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(trainable)

    freeze_module(model.backbone, True)
    freeze_module(model.shared_head, True)

    if not heads_subset:
        if hasattr(model, "out_head"):
            freeze_module(model.out_head, True)
        if hasattr(model, "out_heads"):
            for head in model.out_heads.values():
                freeze_module(head, True)
    else:
        if hasattr(model, "out_head"):
            freeze_module(model.out_head, False)
        if hasattr(model, "out_heads"):
            for name, head in model.out_heads.items():
                freeze_module(head, name in heads_subset)

    for refiner in model.refiners:
        freeze_module(refiner, False)
    if getattr(model, "denoiser", None) is not None:
        freeze_module(model.denoiser, False)

    if callbacks is not None and enable_progress_bar is False:
        try:
            from pytorch_lightning.callbacks.progress import TQDMProgressBar

            callbacks = [
                cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)
            ]
        except Exception:
            pass

    from pytorch_lightning.strategies import Strategy

    if isinstance(trainer_kwargs.get("strategy"), Strategy):
        trainer_kwargs.pop("accelerator", None)

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    trainer = pl.Trainer(
        max_epochs=int(epochs),
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )
    trainer.fit(model, train_loader, val_loader)

    _save_checkpoint(trainer, ckpt_out)

    if hasattr(model, "_use_exp_params_in_A"):
        delattr(model, "_use_exp_params_in_A")

    return model


def train_stage_B1(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    **kwargs,
) -> PhysicallyInformedAE:
    defaults = dict(
        stage_name="B1",
        epochs=12,
        base_lr=1e-6,
        refiner_lr=1e-5,
        train_base=False,
        train_heads=False,
        train_refiner=True,
        refine_steps=2,
        delta_scale=0.12,
        heads_subset=None,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)


def train_stage_B2(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    **kwargs,
) -> PhysicallyInformedAE:
    defaults = dict(
        stage_name="B2",
        epochs=15,
        base_lr=3e-5,
        refiner_lr=3e-6,
        train_base=True,
        train_heads=True,
        train_refiner=True,
        refine_steps=2,
        delta_scale=0.08,
        heads_subset=None,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)


def train_refiner_idx(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    k: int,
    *,
    epochs: int = 40,
    refiner_lr: float = 1e-4,
    delta_scale: float = 0.10,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    ckpt_in: Optional[str] = None,
    ckpt_out: Optional[str] = None,
    use_denoiser_during_B: bool = False,
    use_exp_params_for_resid: Optional[List[str]] = None,
    w_pw_raw: Optional[float] = None,
    w_spectral_angle: Optional[float] = None,
    w_peak: Optional[float] = None,
    w_params: Optional[float] = None,
    use_relobralo_top: Optional[bool] = None,
    pw_cfg: Optional[dict] = None,
    **trainer_kwargs,
) -> PhysicallyInformedAE:
    _load_weights_if_any(model, ckpt_in)

    if w_pw_raw is not None:
        model.w_pw_raw = float(w_pw_raw)
    if w_spectral_angle is not None:
        model.w_spectral_angle = float(w_spectral_angle)
    if w_peak is not None:
        model.w_peak = float(w_peak)
    if w_params is not None:
        model.w_params = float(w_params)
    if use_relobralo_top is not None:
        model.use_relobralo_top = bool(use_relobralo_top)

    if pw_cfg is not None and hasattr(model, "peak_weighted_loss"):
        defaults = dict(
            peak_weight=4.0,
            baseline_weight=1.0,
            pre_smooth_sigma=1.3,
            salience_smooth_sigma=1.8,
            peak_kind="min",
            curv_scale_k=2.2,
            border_policy="taper",
            border_extra_margin=2,
            weight_normalize="mean",
            renorm_after_smoothing=True,
            salience_gamma=0.9,
            spread_kind="gaussian",
            spread_sigma=2.5,
            spread_kernel=11,
        )
        defaults.update(pw_cfg)
        model.peak_weighted_loss = PeakWeightedMSELoss(**defaults)

    model.use_denoiser = bool(use_denoiser_during_B)
    model.base_lr = 1e-8
    model.refiner_lr = float(refiner_lr)
    model.set_stage_mode("B1", refine_steps=int(k + 1), delta_scale=float(delta_scale))

    model._use_exp_params_for_resid = (
        list(use_exp_params_for_resid) if use_exp_params_for_resid else []
    )

    if model._use_exp_params_for_resid:
        print("✅ Mode raffineur avec paramètres expérimentaux:")
        print(
            f"   → Résidus calculés avec {model._use_exp_params_for_resid} GT"
        )
        print(
            f"   → Raffineur ne corrige PAS {model._use_exp_params_for_resid}"
        )

        mask = torch.ones(len(model.predict_params), dtype=torch.float32)
        for idx, name in enumerate(model.predict_params):
            if name in model._use_exp_params_for_resid:
                mask[idx] = 0.0
        model.refine_mask_base = mask.to(model.device)

        correctable = [
            name
            for name in model.predict_params
            if name not in model._use_exp_params_for_resid
        ]
        print(f"   → Paramètres corrigeables: {correctable}")
    else:
        print("✅ Mode raffineur standard:")
        print("   → Résidus avec tous les paramètres prédits")
        print("   → Raffineur corrige tous les paramètres")

    def freeze_module(module, trainable: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(trainable)

    def freeze_all_refiners_except(target: int) -> None:
        for idx, refiner in enumerate(model.refiners):
            freeze_module(refiner, idx == target)

    freeze_module(model.backbone, False)
    freeze_module(model.shared_head, False)
    if hasattr(model, "out_head"):
        freeze_module(model.out_head, False)
    if hasattr(model, "out_heads"):
        for head in model.out_heads.values():
            freeze_module(head, False)

    freeze_all_refiners_except(int(k))

    if callbacks is not None and enable_progress_bar is False:
        try:
            from pytorch_lightning.callbacks.progress import TQDMProgressBar

            callbacks = [
                cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)
            ]
        except Exception:
            pass

    from pytorch_lightning.strategies import Strategy

    if isinstance(trainer_kwargs.get("strategy"), Strategy):
        trainer_kwargs.pop("accelerator", None)

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    trainer = pl.Trainer(
        max_epochs=int(epochs),
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)

    if hasattr(model, "_use_exp_params_for_resid"):
        delattr(model, "_use_exp_params_for_resid")

    return model
