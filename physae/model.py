"""Lightning module implementing the physically informed auto-encoder."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .config import NORM_PARAMS
from .losses import ReLoBRaLoLoss
from .models import (
    EfficientNetEncoder,
    EfficientNetRefiner,
    build_encoder,
    build_refiner,
)
from .normalization import norm_param_torch, unnorm_param_torch
from .physics import batch_physics_forward_multimol_vgrid
from .optimizers import Lion
from .baseline import scale_first_window


class PhysicallyInformedAE(pl.LightningModule):
    def __init__(
        self,
        n_points: int,
        param_names: List[str],
        poly_freq_CH4,
        transitions_dict,
        lr: float = 1e-4,
        alpha_param: float = 1.0,
        alpha_phys: float = 1.0,
        head_mode: str = "multi",
        predict_params: Optional[List[str]] = None,
        film_params: Optional[List[str]] = None,
        refine_steps: int = 1,
        refine_delta_scale: float = 0.1,
        refine_target: str = "noisy",
        refine_warmup_epochs: int = 30,
        freeze_base_epochs: int = 20,
        base_lr: float | None = None,
        refiner_lr: float | None = None,
        stage3_lr_shrink: float = 0.33,
        stage3_refine_steps: Optional[int] = 2,
        stage3_delta_scale: Optional[float] = 0.08,
        stage3_alpha_phys: Optional[float] = 0.7,
        stage3_alpha_param: Optional[float] = 0.3,
        baseline_fix_enable: bool = False,
        baseline_fix_sideband: int = 50,
        baseline_fix_degree: int = 2,
        baseline_fix_weight: float = 1.0,
        baseline_fix_in_warmup: bool = False,
        recon_max1: bool = False,
        corr_mode: str = "savgol",
        corr_savgol_win: int = 11,
        corr_savgol_poly: int = 3,
        weight_mf: float = 1.0,
        encoder_width_mult: float = 1.0,
        encoder_depth_mult: float = 1.0,
        encoder_expand_ratio_scale: float = 1.0,
        encoder_se_ratio: float = 0.25,
        encoder_norm_groups: int = 8,
        encoder_name: str = "efficientnet",
        encoder_config: Dict[str, Any] | None = None,
        shared_head_hidden_scale: float = 0.5,
        refiner_encoder_width_mult: float = 1.0,
        refiner_encoder_depth_mult: float = 1.0,
        refiner_encoder_expand_ratio_scale: float = 1.0,
        refiner_encoder_se_ratio: float = 0.25,
        refiner_encoder_norm_groups: int = 8,
        refiner_hidden_scale: float = 0.5,
        refiner_name: str = "efficientnet",
        refiner_config: Dict[str, Any] | None = None,
        optimizer_name: str = "adamw",
        optimizer_betas: Tuple[float, float] | List[float] = (0.9, 0.999),
        optimizer_weight_decay: float = 1e-4,
        scheduler_eta_min: float = 1e-9,
        scheduler_T_max: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["transitions_dict", "poly_freq_CH4"])
        self.param_names = list(param_names)
        self.n_params = len(self.param_names)
        self.n_points = n_points
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict

        self.lr = float(lr)
        self.alpha_param = float(alpha_param)
        self.alpha_phys = float(alpha_phys)
        self.refine_steps = int(refine_steps)
        self.refine_target = refine_target.lower()
        assert self.refine_target in {"noisy", "clean"}

        self.refine_warmup_epochs = int(refine_warmup_epochs)
        self.freeze_base_epochs = int(freeze_base_epochs)
        self.base_lr = float(base_lr) if base_lr is not None else self.lr
        self.refiner_lr = float(refiner_lr) if refiner_lr is not None else self.lr
        self._froze_base = False

        self.stage3_lr_shrink = float(stage3_lr_shrink)
        self.stage3_refine_steps = stage3_refine_steps
        self.stage3_delta_scale = stage3_delta_scale
        self.stage3_alpha_phys = stage3_alpha_phys
        self.stage3_alpha_param = stage3_alpha_param

        self.baseline_fix_enable = bool(baseline_fix_enable)
        self.baseline_fix_sideband = int(baseline_fix_sideband)
        self.baseline_fix_degree = int(baseline_fix_degree)
        self.baseline_fix_weight = float(baseline_fix_weight)
        self.baseline_fix_in_warmup = bool(baseline_fix_in_warmup)

        self.weight_mf = float(weight_mf)

        self.corr_mode = str(corr_mode).lower()
        self.corr_savgol_win = int(corr_savgol_win)
        self.corr_savgol_poly = int(corr_savgol_poly)

        if predict_params is None:
            predict_params = self.param_names
        self.predict_params = list(predict_params)
        self.provided_params = [p for p in self.param_names if p not in self.predict_params]
        unknown = set(self.predict_params) - set(self.param_names)
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")
        if len(self.predict_params) == 0:
            raise ValueError("At least one parameter must be predicted.")

        if film_params is None:
            self.film_params = list(self.provided_params)
        else:
            self.film_params = list(film_params)
        bad = set(self.film_params) - set(self.param_names)
        if bad:
            raise ValueError(f"Unknown film_params: {bad}")
        not_provided = set(self.film_params) - set(self.provided_params)
        if not_provided:
            raise ValueError("film_params must be a subset of provided_params")

        self.name_to_idx = {name: i for i, name in enumerate(self.param_names)}
        self.predict_idx = [self.name_to_idx[name] for name in self.predict_params]
        self.provided_idx = [self.name_to_idx[name] for name in self.provided_params]

        encoder_kwargs = dict(encoder_config or {})
        encoder_kwargs.setdefault("in_channels", 1)
        encoder_kwargs.setdefault("width_mult", float(encoder_width_mult))
        encoder_kwargs.setdefault("depth_mult", float(encoder_depth_mult))
        encoder_kwargs.setdefault("expand_ratio_scale", float(encoder_expand_ratio_scale))
        encoder_kwargs.setdefault("se_ratio", float(encoder_se_ratio))
        encoder_kwargs.setdefault("norm_groups", int(encoder_norm_groups))

        self.encoder_name = str(encoder_name)
        backbone = build_encoder(self.encoder_name, **encoder_kwargs)
        if not isinstance(backbone, nn.Module):
            raise TypeError(
                f"Encoder '{self.encoder_name}' must return an nn.Module, got {type(backbone)!r}."
            )
        if not hasattr(backbone, "feat_dim"):
            raise AttributeError(
                f"Encoder '{self.encoder_name}' must expose a 'feat_dim' attribute."
            )
        self.backbone: nn.Module = backbone
        self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        feat_dim = self.backbone.feat_dim
        hidden = max(32, int(round(feat_dim * float(shared_head_hidden_scale))))

        self.shared_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.cond_dim = len(self.film_params)
        if self.cond_dim > 0:
            self.film = nn.Sequential(nn.Linear(self.cond_dim, hidden), nn.Tanh(), nn.Linear(hidden, 2 * hidden))
        else:
            self.film = None

        self.use_film = True
        if self.cond_dim > 0:
            self.register_buffer("film_mask", torch.ones(self.cond_dim))
        else:
            self.film_mask = None

        self.head_mode = str(head_mode).lower()
        if self.head_mode not in {"single", "multi"}:
            raise ValueError("head_mode must be 'single' or 'multi'")
        if self.head_mode == "single":
            self.out_head = nn.Linear(hidden, len(self.predict_params))
        else:
            self.out_heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in self.predict_params})

        refiner_kwargs = dict(refiner_config or {})
        refiner_kwargs.setdefault("m_params", len(self.predict_params))
        refiner_kwargs.setdefault("cond_dim", self.cond_dim)
        refiner_kwargs.setdefault("backbone_feat_dim", feat_dim)
        refiner_kwargs.setdefault("delta_scale", float(refine_delta_scale))
        refiner_kwargs.setdefault("encoder_width_mult", float(refiner_encoder_width_mult))
        refiner_kwargs.setdefault("encoder_depth_mult", float(refiner_encoder_depth_mult))
        refiner_kwargs.setdefault(
            "encoder_expand_ratio_scale", float(refiner_encoder_expand_ratio_scale)
        )
        refiner_kwargs.setdefault("encoder_se_ratio", float(refiner_encoder_se_ratio))
        refiner_kwargs.setdefault("encoder_norm_groups", int(refiner_encoder_norm_groups))
        refiner_kwargs.setdefault("hidden_scale", float(refiner_hidden_scale))

        self.refiner_name = str(refiner_name)
        legacy_key_map = {
            "width_mult": ("encoder_width_mult", float),
            "depth_mult": ("encoder_depth_mult", float),
            "expand_ratio_scale": ("encoder_expand_ratio_scale", float),
            "se_ratio": ("encoder_se_ratio", float),
            "norm_groups": ("encoder_norm_groups", int),
        }
        for legacy_key, (target_key, caster) in legacy_key_map.items():
            if legacy_key in refiner_kwargs:
                value = refiner_kwargs.pop(legacy_key)
                refiner_kwargs.setdefault(target_key, caster(value))
        refiner = build_refiner(self.refiner_name, **refiner_kwargs)
        if not isinstance(refiner, nn.Module):
            raise TypeError(
                f"Refiner '{self.refiner_name}' must return an nn.Module, got {type(refiner)!r}."
            )
        self.refiner = refiner
        self.loss_names_params = [f"param_{name}" for name in self.predict_params]
        self.relo_params = ReLoBRaLoLoss(self.loss_names_params, alpha=0.9, tau=1.0, history_len=10)
        self.loss_names_top = ["phys_mse", "phys_corr", "param_group"]
        self.relo_top = ReLoBRaLoLoss(self.loss_names_top, alpha=0.9, tau=1.0, history_len=10)

        self._override_stage: Optional[str] = None
        self._override_refine_steps: Optional[int] = None
        self._override_delta_scale: Optional[float] = None
        self.recon_max1 = bool(recon_max1)

        self.optimizer_name = str(optimizer_name).lower()
        betas_tuple = tuple(float(b) for b in optimizer_betas)
        if len(betas_tuple) != 2:
            raise ValueError("optimizer_betas must contain two values (beta1, beta2).")
        self.optimizer_betas = betas_tuple
        self.optimizer_weight_decay = float(optimizer_weight_decay)
        self.scheduler_eta_min = float(scheduler_eta_min)
        self.scheduler_T_max = int(scheduler_T_max) if scheduler_T_max is not None else None

    def set_film_usage(self, use: bool = True) -> None:
        self.use_film = bool(use)
        if hasattr(self, "refiner") and hasattr(self.refiner, "use_film"):
            self.refiner.use_film = self.use_film

    def set_film_subset(self, names=None) -> None:
        if self.cond_dim == 0 or self.film_mask is None:
            return
        if names is None or names == "all":
            mask = torch.ones(self.cond_dim, device=self.film_mask.device, dtype=self.film_mask.dtype)
        else:
            allowed = set(names)
            mask = torch.zeros(self.cond_dim, device=self.film_mask.device, dtype=self.film_mask.dtype)
            for i, name in enumerate(self.film_params):
                if name in allowed:
                    mask[i] = 1.0
        self.film_mask.copy_(mask)

    def set_stage_mode(
        self,
        mode: Optional[str],
        refine_steps: Optional[int] = None,
        delta_scale: Optional[float] = None,
    ) -> None:
        if mode is not None:
            mode = mode.upper()
            assert mode in {"A", "B1", "B2"}
        self._override_stage = mode
        self._override_refine_steps = refine_steps
        self._override_delta_scale = delta_scale
        if delta_scale is not None:
            self.refiner.delta_scale = float(delta_scale)
        if refine_steps is not None:
            self.refine_steps = int(refine_steps)

    def _predict_params_from_features(
        self, feat: torch.Tensor, cond_norm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.shared_head(feat)
        if self.film is not None and cond_norm is not None and self.use_film:
            if self.film_mask is not None and self.film_mask.numel() == cond_norm.shape[1]:
                cond_in = cond_norm * self.film_mask.unsqueeze(0)
            else:
                cond_in = cond_norm
            gamma_beta = self.film(cond_in)
            width = h.shape[1]
            gamma, beta = gamma_beta[:, :width], gamma_beta[:, width:]
            h = h * (1 + 0.1 * gamma) + 0.1 * beta
        if self.head_mode == "single":
            y = self.out_head(h)
        else:
            y = torch.cat([self.out_heads[name](h) for name in self.predict_params], dim=1)
        return torch.sigmoid(y).clamp(1e-4, 1 - 1e-4)

    def encode(self, spectra: torch.Tensor, pooled: bool = True, detach: bool = False):
        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat = self.feature_head(latent) if pooled else latent
        return feat.detach() if detach else feat

    def _denorm_params_subset(self, y_norm_subset: torch.Tensor, names: List[str]) -> torch.Tensor:
        cols = [unnorm_param_torch(name, y_norm_subset[:, i]) for i, name in enumerate(names)]
        return torch.stack(cols, dim=1)

    def _compose_full_phys(self, pred_phys: torch.Tensor, provided_phys_tensor: torch.Tensor) -> torch.Tensor:
        batch = pred_phys.shape[0]
        full = pred_phys.new_empty((batch, self.n_params))
        for j, idx in enumerate(self.predict_idx):
            full[:, idx] = pred_phys[:, j]
        for j, idx in enumerate(self.provided_idx):
            full[:, idx] = provided_phys_tensor[:, j]
        return full

    def _physics_reconstruction(self, y_phys_full: torch.Tensor, device, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        params = {name: y_phys_full[:, i] for i, name in enumerate(self.param_names)}
        v_grid_idx = torch.arange(self.n_points, dtype=torch.float64, device=device)
        baseline_idx = self.name_to_idx["baseline0"]
        baseline_coeffs = y_phys_full[:, baseline_idx : baseline_idx + 3]
        spectra, _ = batch_physics_forward_multimol_vgrid(
            params["sig0"],
            params["dsig"],
            self.poly_freq_CH4,
            v_grid_idx,
            baseline_coeffs,
            self.transitions_dict,
            params["P"],
            params["T"],
            {"CH4": params["mf_CH4"]},
            device=device,
        )
        spectra = spectra.to(torch.float32)
        if self.recon_max1:
            maxv = spectra.amax(dim=1, keepdim=True).clamp_min(1e-9)
            spectra = spectra / maxv
        return spectra

    def _make_condition_from_norm(self, params_true_norm: torch.Tensor) -> Optional[torch.Tensor]:
        if self.cond_dim == 0:
            return None
        cols = [params_true_norm[:, self.name_to_idx[name]].unsqueeze(1) for name in self.film_params]
        return torch.cat(cols, dim=1)

    def _make_condition_from_phys(self, provided_phys: dict, device, dtype=torch.float32) -> Optional[torch.Tensor]:
        if self.cond_dim == 0:
            return None
        missing = [name for name in self.film_params if name not in provided_phys]
        if missing:
            raise ValueError(f"Missing FiLM parameters: {missing}")
        cols = []
        for name in self.film_params:
            value = provided_phys[name].to(device)
            if value.ndim > 1:
                value = value.view(-1)
            cols.append(norm_param_torch(name, value).unsqueeze(1))
        return torch.cat(cols, dim=1).to(dtype)

    def _fit_residual_baseline_poly(self, resid: torch.Tensor, degree: int, sideband: int) -> torch.Tensor:
        batch, length = resid.shape
        device = resid.device
        dtype = resid.dtype
        degree = int(max(0, min(2, degree)))
        sideband = int(max(1, min(length // 2, sideband)))
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        mask[:sideband] = True
        mask[length - sideband :] = True
        x = torch.arange(length, device=device, dtype=dtype)
        cols = [torch.ones_like(x)]
        if degree >= 1:
            cols.append(x)
        if degree >= 2:
            cols.append(x * x)
        X = torch.stack(cols, dim=1)
        Xm = X[mask]
        Y = resid[:, mask]
        XtX = Xm.T @ Xm
        XtX_inv = torch.linalg.pinv(XtX)
        XtY = Xm.T @ Y.T
        coeffs = (XtX_inv @ XtY).T
        return coeffs

    def _apply_baseline_coeff_delta_in_norm(
        self, params_pred_norm: torch.Tensor, coeffs_phys: torch.Tensor, weight: float = 1.0
    ) -> torch.Tensor:
        if coeffs_phys is None:
            return params_pred_norm
        name_to_predcol = {name: j for j, name in enumerate(self.predict_params)}
        updated = params_pred_norm
        for k in range(coeffs_phys.shape[1]):
            pname = f"baseline{k}"
            if pname not in name_to_predcol:
                continue
            j = name_to_predcol[pname]
            vmin, vmax = NORM_PARAMS[pname]
            delta_norm = (-coeffs_phys[:, k]) / (vmax - vmin)
            updated[:, j] = torch.clamp(updated[:, j] + weight * delta_norm.to(updated.dtype), 1e-4, 1 - 1e-4)
        return updated

    def _maybe_baseline_fix(
        self,
        params_pred_norm: torch.Tensor,
        noisy: torch.Tensor,
        target_for_resid: torch.Tensor,
        provided_phys_tensor: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.baseline_fix_enable:
            return params_pred_norm
        pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
        y_phys_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
        recon = self._physics_reconstruction(y_phys_full, noisy.device, scale=scale)
        resid = recon - target_for_resid
        coeffs = self._fit_residual_baseline_poly(
            resid, degree=self.baseline_fix_degree, sideband=self.baseline_fix_sideband
        )
        params_new = self._apply_baseline_coeff_delta_in_norm(
            params_pred_norm, coeffs, weight=self.baseline_fix_weight
        )
        return params_new

    def _savgol_coeffs(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 1,
        delta: float = 1.0,
        device=None,
        dtype=torch.float64,
    ) -> torch.Tensor:
        assert deriv >= 0
        W = int(window_length)
        P = int(polyorder)
        if W % 2 == 0:
            W += 1
        if W < 3:
            W = 3
        if P >= W:
            P = W - 1
        m = (W - 1) // 2
        dev = device or self.device
        x = torch.arange(-m, m + 1, device=dev, dtype=dtype)
        A = torch.stack([x ** j for j in range(P + 1)], dim=1)
        pinv = torch.linalg.pinv(A)
        coeff = math.factorial(deriv) * pinv[deriv, :] / (delta ** deriv)
        return coeff.to(dtype=torch.float32)

    def _savgol_deriv(self, y: torch.Tensor, window_length: int, polyorder: int, deriv: int = 1) -> torch.Tensor:
        batch, length = y.shape
        W = int(window_length)
        if W % 2 == 0:
            W += 1
        if W > length:
            W = length if (length % 2 == 1) else (length - 1)
        W = max(W, 3)
        P = min(int(polyorder), W - 1)
        coeff = self._savgol_coeffs(W, P, deriv=deriv, device=y.device, dtype=torch.float64)
        coeff = coeff.view(1, 1, -1)
        pad = (W - 1) // 2
        y1 = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
        out = F.conv1d(y1, coeff).squeeze(1)
        return out

    def _dx(self, y: torch.Tensor) -> torch.Tensor:
        d = 0.5 * (y[:, 2:] - y[:, :-2])
        left = d[:, :1]
        right = d[:, -1:]
        return torch.cat([left, d, right], dim=1)

    def _pearson_corr_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-8,
        derivative: str | bool = False,
        savgol_win: int = 11,
        savgol_poly: int = 3,
    ) -> torch.Tensor:
        mode = derivative
        if isinstance(derivative, bool):
            mode = "central" if derivative else "none"
        mode = (mode or "none").lower()
        if mode == "central":
            y_hat = self._dx(y_hat)
            y = self._dx(y)
        elif mode == "savgol":
            y_hat = self._savgol_deriv(y_hat, savgol_win, savgol_poly, deriv=1)
            y = self._savgol_deriv(y, savgol_win, savgol_poly, deriv=1)
        y_hat = y_hat - y_hat.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        cov = (y_hat * y).sum(dim=1)
        denom = (torch.sqrt((y_hat ** 2).sum(dim=1) + eps) * torch.sqrt((y ** 2).sum(dim=1) + eps))
        corr = cov / denom
        return 1 - corr.mean()

    def _params_loss(self, params_pred_norm: torch.Tensor, params_true_norm: torch.Tensor) -> torch.Tensor:
        losses = []
        for j, name in enumerate(self.predict_params):
            loss = F.l1_loss(params_pred_norm[:, j], params_true_norm[:, self.name_to_idx[name]])
            if name == "mf_CH4":
                loss = loss * self.weight_mf
            losses.append(loss)
        return torch.stack(losses)

    def _physics_loss(self, recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_mse = F.mse_loss(recon, target)
        loss_huber = F.smooth_l1_loss(recon, target)
        loss_corr = self._pearson_corr_loss(recon, target, derivative=self.corr_mode,
                                            savgol_win=self.corr_savgol_win, savgol_poly=self.corr_savgol_poly)
        return loss_mse, loss_huber, loss_corr

    def _common_step(self, batch, step_name: str):
        noisy = batch["noisy_spectra"].to(self.device)
        clean = batch["clean_spectra"].to(self.device)
        params_true = batch["params"].to(self.device)
        scale = batch.get("scale")
        if scale is not None:
            scale = scale.to(self.device)
        cond_norm = self._make_condition_from_norm(params_true)
        latent, _ = self.backbone(noisy.unsqueeze(1))
        feat_shared = self.feature_head(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared, cond_norm=cond_norm)
        if len(self.provided_params) > 0:
            provided_phys_tensor = torch.stack(
                [unnorm_param_torch(name, params_true[:, self.name_to_idx[name]]) for name in self.provided_params],
                dim=1,
            )
        else:
            provided_phys_tensor = params_true.new_zeros((params_true.size(0), 0))
        target_for_resid = clean if self.refine_target == "clean" else noisy
        if self.current_epoch >= self.refine_warmup_epochs or self.baseline_fix_in_warmup:
            params_pred_norm = self._maybe_baseline_fix(
                params_pred_norm, noisy, target_for_resid, provided_phys_tensor, scale=scale
            )
        if self.refine_steps > 0 and self.current_epoch >= self.refine_warmup_epochs:
            for _ in range(self.refine_steps):
                pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
                y_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
                recon = self._physics_reconstruction(y_full, noisy.device, scale=scale)
                resid = recon - target_for_resid
                delta = self.refiner(
                    noisy=noisy,
                    resid=resid,
                    params_pred_norm=params_pred_norm,
                    cond_norm=cond_norm,
                    feat_shared=feat_shared,
                )
                params_pred_norm = params_pred_norm.add(delta).clamp(1e-4, 1 - 1e-4)
        pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
        y_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
        recon = self._physics_reconstruction(y_full, noisy.device, scale=scale)
        loss_phys_mse, loss_phys_huber, loss_phys_corr = self._physics_loss(recon, clean)
        per_param_losses = self._params_loss(params_pred_norm, params_true)
        loss_param_group = per_param_losses.mean() if len(per_param_losses) > 0 else torch.tensor(0.0, device=self.device)
        top_vec = torch.stack([loss_phys_mse, loss_phys_corr, loss_param_group])
        w_params = self.relo_params.compute_weights(per_param_losses)
        loss_params = torch.sum(w_params * per_param_losses)
        top_vec = torch.stack([loss_phys_mse, loss_phys_corr, loss_params])
        w_top = self.relo_top.compute_weights(top_vec)
        priors_top = torch.tensor(
            [self.alpha_phys, self.alpha_phys, self.alpha_param], device=top_vec.device, dtype=top_vec.dtype
        )
        w_top = w_top * priors_top
        w_top = 3.0 * w_top / (w_top.sum() + 1e-12)
        loss = torch.sum(w_top * top_vec)
        self.log(f"{step_name}_loss", loss, on_epoch=True)
        self.log(f"{step_name}_loss_phys", loss_phys_mse, on_epoch=True)
        self.log(f"{step_name}_loss_phys_huber", loss_phys_huber, on_epoch=True)
        self.log(f"{step_name}_loss_phys_corr", loss_phys_corr, on_epoch=True)
        self.log(f"{step_name}_loss_param_group", loss_param_group, on_epoch=True)
        if len(per_param_losses) > 0:
            self.log(f"{step_name}_loss_param", per_param_losses.mean(), on_epoch=True)
        self.log(f"{step_name}_w_top_phys", w_top[0], on_epoch=True)
        self.log(f"{step_name}_w_top_phys_corr", w_top[1], on_epoch=True)
        self.log(f"{step_name}_w_top_param_group", w_top[2], on_epoch=True)
        for j, name in enumerate(self.predict_params):
            self.log(f"{step_name}_loss_param_{name}", per_param_losses[j], on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        self._common_step(batch, "val")

    def on_train_epoch_start(self):
        if self._override_stage is not None:
            stage = self._override_stage
            if stage == "A":
                self._set_requires_grad(self.refiner, False)
                self._set_requires_grad(
                    [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film],
                    True,
                )
            elif stage == "B1":
                self._set_requires_grad(
                    [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film],
                    False,
                )
                self._set_requires_grad(self.refiner, True)
            elif stage == "B2":
                self._set_requires_grad(
                    [
                        self.backbone,
                        self.shared_head,
                        getattr(self, "out_head", None),
                        getattr(self, "out_heads", None),
                        self.film,
                        self.refiner,
                    ],
                    True,
                )
            return

        epoch = self.current_epoch
        stage3_start = self.refine_warmup_epochs + self.freeze_base_epochs
        if epoch < self.refine_warmup_epochs:
            self._set_requires_grad(self.refiner, False)
            self._set_requires_grad(
                [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film],
                True,
            )
            self._froze_base = False
        elif epoch < stage3_start:
            if not self._froze_base:
                self._set_requires_grad(
                    [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film],
                    False,
                )
                self._set_requires_grad(self.refiner, True)
                self._froze_base = True
        else:
            self._set_requires_grad(
                [
                    self.backbone,
                    self.shared_head,
                    getattr(self, "out_head", None),
                    getattr(self, "out_heads", None),
                    self.film,
                    self.refiner,
                ],
                True,
            )
            if epoch == stage3_start:
                if hasattr(self.trainer, "optimizers") and len(self.trainer.optimizers) > 0:
                    opt = self.trainer.optimizers[0]
                    for pg in opt.param_groups:
                        pg["lr"] *= self.stage3_lr_shrink
                if self.stage3_refine_steps is not None:
                    self.refine_steps = int(self.stage3_refine_steps)
                if self.stage3_delta_scale is not None:
                    self.refiner.delta_scale = float(self.stage3_delta_scale)
                if self.stage3_alpha_phys is not None:
                    self.alpha_phys = float(self.stage3_alpha_phys)
                if self.stage3_alpha_param is not None:
                    self.alpha_param = float(self.stage3_alpha_param)

    def _set_requires_grad(self, modules, flag: bool) -> None:
        if modules is None:
            return
        if not isinstance(modules, (list, tuple)):
            modules = [modules]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(flag)

    def configure_optimizers(self):
        base_params = list(self.backbone.parameters()) + list(self.shared_head.parameters())
        if hasattr(self, "out_head"):
            base_params += list(self.out_head.parameters())
        if hasattr(self, "out_heads"):
            base_params += list(self.out_heads.parameters())
        if self.film is not None:
            base_params += list(self.film.parameters())
        refiner_params = list(self.refiner.parameters())
        param_groups = [
            {"params": base_params, "lr": self.base_lr},
            {"params": refiner_params, "lr": self.refiner_lr},
        ]
        opt_name = getattr(self, "optimizer_name", "adamw").lower()
        opt_kwargs = {
            "lr": self.base_lr,
            "betas": getattr(self, "optimizer_betas", (0.9, 0.999)),
            "weight_decay": getattr(self, "optimizer_weight_decay", 1e-4),
        }
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
        elif opt_name == "lion":
            optimizer = Lion(param_groups, **opt_kwargs)
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'.")
        t_max = self.scheduler_T_max
        if t_max is None:
            t_max = self.trainer.max_epochs if self.trainer is not None else 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_max),
            eta_min=getattr(self, "scheduler_eta_min", 1e-9),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.no_grad()
    def infer(
        self,
        spectra: torch.Tensor,
        provided_phys: dict,
        *,
        refine: bool = True,
        resid_target: str = "input",
        scale: Optional[torch.Tensor] = None,
    ):
        self.eval()
        device = spectra.device
        batch = spectra.shape[0]
        missing = [name for name in self.provided_params if name not in provided_phys]
        if missing:
            raise ValueError(f"Missing provided physical parameters: {missing}")
        cond_norm = self._make_condition_from_phys(provided_phys, device, dtype=torch.float32)
        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat_shared = self.feature_head(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared, cond_norm=cond_norm)
        if len(self.provided_params) > 0:
            provided_list = [provided_phys[name].to(device) for name in self.provided_params]
            provided_phys_tensor = torch.stack(provided_list, dim=1)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((batch, 0))
        spectra_target = spectra if resid_target in ("input", "noisy") else None
        scale_est = scale_first_window(spectra).to(spectra.dtype)
        if refine and self.refine_steps > 0:
            for _ in range(self.refine_steps):
                pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
                y_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
                recon = self._physics_reconstruction(y_full, device, scale=None)
                if spectra_target is None:
                    break
                resid = recon - spectra_target
                params_pred_norm = self._maybe_baseline_fix(
                    params_pred_norm, spectra, spectra_target, provided_phys_tensor, scale=None
                )
                delta = self.refiner(
                    noisy=spectra,
                    resid=resid,
                    params_pred_norm=params_pred_norm,
                    cond_norm=cond_norm,
                    feat_shared=feat_shared,
                )
                params_pred_norm = params_pred_norm.add(delta).clamp(1e-4, 1 - 1e-4)
        pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
        y_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
        recon = self._physics_reconstruction(y_full, device, scale=None)
        return {
            "params_pred_norm": params_pred_norm,
            "y_phys_full": y_full,
            "spectra_recon": recon,
            "norm_scale": scale_est,
        }
