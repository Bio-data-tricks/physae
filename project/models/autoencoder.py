"""
Physically Informed Autoencoder (Lightning Module).

Complete implementation extracted from the original physae.py file.
"""
from typing import List, Optional, Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Import from new modular structure
from physics.tips import Tips2021QTpy
from physics.forward import batch_physics_forward_multimol_vgrid
from utils.lowess import lowess_value
from data.normalization import unnorm_param_torch
from .backbone import EfficientNetEncoder
from .refiner import EfficientNetRefiner
from .denoiser import Denoiser1D
from losses.spectral import SpectralAngleLoss
from losses.peak_weighted import PeakWeightedMSELoss
from losses.relobralo import ReLoBRaLoLoss
from training.scheduler import CosineAnnealingWarmRestartsWithDecayAndLinearWarmup


class GatedSharedHead(nn.Module):
    """Gated shared head for feature refinement."""

    def __init__(self, feat_dim: int, hidden: int, p_drop: float):
        super().__init__()
        self.short_head = nn.Linear(feat_dim, hidden)
        self.deep_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p_drop),
        )
        self.gate_head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_head(x)
        return g * self.deep_head(x) + (1 - g) * self.short_head(x)


class PhysicallyInformedAE(pl.LightningModule):
    """
    Physically Informed Autoencoder with PyTorch Lightning.

    This model combines:
    - EfficientNet encoder for feature extraction from spectra
    - Physical parameter prediction
    - Physics-based spectral reconstruction
    - Iterative refinement with cascade refiners
    - Multiple loss functions with optional ReLoBRaLo balancing
    - Multi-stage training support (A, B1, B2, DEN)
    """

    def __init__(
        self,
        n_points: int,
        param_names: List[str],
        poly_freq_CH4: Sequence[float] | torch.Tensor | None,
        transitions_dict,
        mlp_dropout: float = 0.10,
        refiner_mlp_dropout: float = 0.10,

        # --- optims & pondérations globales ---
        lr: float = 1e-4,
        # --- têtes de sortie ---
        head_mode: str = "multi",
        predict_params: Optional[List[str]] = None,
        # --- raffinement ---
        refine_steps: int = 1,
        refine_delta_scale: float = 0.1,
        refine_target: str = "noisy",
        refine_warmup_epochs: int = 30,
        freeze_base_epochs: int = 20,
        base_lr: float = None,
        refiner_lr: float = None,
        stage3_lr_shrink: float = 0.33,
        stage3_refine_steps: Optional[int] = 2,
        stage3_delta_scale: Optional[float] = 0.08,
        # --- reconstruction / pertes (existantes) ---
        recon_max1: bool = False,
        corr_mode: str = "savgol",
        corr_savgol_win: int = 11,
        corr_savgol_poly: int = 3,
        huber_beta: float = 0.002,
        weight_mf: float = 2.0,
        # --- EfficientNet backbone  ---
        backbone_variant: str = "s",
        refiner_variant: str = "s",
        backbone_width_mult: float = 1.0,
        backbone_depth_mult: float = 1.0,
        refiner_width_mult: float = 1.0,
        refiner_depth_mult: float = 1.0,
        backbone_stem_channels: int | None = None,
        refiner_stem_channels: int | None = None,
        backbone_drop_path: float = 0.1,
        refiner_drop_path: float = 0.1,
        backbone_se_ratio: float = 0.25,
        refiner_se_ratio: float = 0.25,
        refiner_feature_pool: str = "avg",
        refiner_shared_hidden_scale: float = 0.5,
        # --- PHYSIQUE / TIPS ---
        tipspy: Tips2021QTpy | None = None,
        use_relobralo_top: bool = True,     # activer ReLoBRaLo pour losses principales

        # --- débruitage ---
        use_denoiser: bool = False,
        denoiser_lr: float = 1e-4,
        denoiser_width: int = 64,

        # --- POIDS DE PERTES (CONFIGURATION COMPLÈTE) ---
        w_pw_raw: float = 1.0,
        w_spectral_angle: float = 0.5,
        w_peak: float = 0.5,
        w_params: float = 1.0,
        pw_cfg: dict | None = None,

        scheduler_T_0: int = 10,
        scheduler_T_mult: float = 1.2,
        scheduler_eta_min: float = 1e-9,
        scheduler_decay_factor: float = 0.75,
        scheduler_warmup_epochs: int = 5,
    ):
        super().__init__()

        self.param_names = list(param_names)
        self.n_params = len(self.param_names)
        self.n_points = n_points
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict
        self.tipspy = tipspy
        self.save_hyperparameters(ignore=["transitions_dict", "poly_freq_CH4", "tipspy"])
        self.mlp_dropout = float(mlp_dropout)
        self.refiner_mlp_dropout = float(refiner_mlp_dropout)

        # --- optims / pondérations globales ---
        self.lr = float(lr)
        self.huber_beta = float(huber_beta)

        # --- raffinement ---
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

        # --- pertes / métriques ---
        self.weight_mf = float(weight_mf)
        self.corr_mode = str(corr_mode).lower()
        self.corr_savgol_win = int(corr_savgol_win)
        self.corr_savgol_poly = int(corr_savgol_poly)
        self.recon_max1 = bool(recon_max1)

        self.w_pw_raw = float(w_pw_raw)
        self.w_spectral_angle = float(w_spectral_angle)
        self.w_peak = float(w_peak)
        self.w_params = float(w_params)
        self.use_relobralo_top = bool(use_relobralo_top)

        # --- instanciation des modules de loss ---
        self.spectral_angle_loss = SpectralAngleLoss(eps=1e-8)

        # PeakWeightedMSELoss (torch-only), config par défaut + override via pw_cfg
        _pw_default = dict(
            peak_weight=4.0, baseline_weight=1.0,
            pre_smooth_sigma=1.3, salience_smooth_sigma=1.8,
            peak_kind="min", curv_scale_k=2.2,
            border_policy="taper", border_extra_margin=2,
            weight_normalize="mean", renorm_after_smoothing=True,
            salience_gamma=0.9,
            spread_kind="gaussian", spread_sigma=2.5, spread_kernel=11
        )
        if pw_cfg is not None:
            _pw_default.update(pw_cfg)
        self.peak_weighted_loss = PeakWeightedMSELoss(**_pw_default)

        # --- débruiteur (optionnel) ---
        self.use_denoiser = bool(use_denoiser)
        self.denoiser_lr = float(denoiser_lr)
        self.denoiser = Denoiser1D(in_ch=1, base_ch=int(denoiser_width), depth=6)

        # --- paramètres à prédire / fournis ---
        if predict_params is None:
            predict_params = self.param_names
        self.predict_params = list(predict_params)
        self.provided_params = [p for p in self.param_names if p not in self.predict_params]
        unknown = set(self.predict_params) - set(self.param_names)
        assert not unknown, f"Paramètres inconnus: {unknown}"
        assert len(self.predict_params) >= 1

        self.name_to_idx = {n: i for i, n in enumerate(self.param_names)}
        self.predict_idx = [self.name_to_idx[p] for p in self.predict_params]
        self.provided_idx = [self.name_to_idx[p] for p in self.provided_params]

        self.scheduler_T_0 = int(scheduler_T_0)
        self.scheduler_T_mult = float(scheduler_T_mult)
        self.scheduler_eta_min = float(scheduler_eta_min)
        self.scheduler_decay_factor = float(scheduler_decay_factor)
        self.scheduler_warmup_epochs = int(scheduler_warmup_epochs)

        # ===== Backbone EfficientNet 1D =====
        self.backbone = EfficientNetEncoder(
            in_channels=1,
            variant=backbone_variant,
            width_mult=backbone_width_mult,
            depth_mult=backbone_depth_mult,
            se_ratio=backbone_se_ratio,
            drop_path_rate=backbone_drop_path,
            stem_channels=backbone_stem_channels,
        )

        self.head_mode = str(head_mode).lower()
        assert self.head_mode in {"single", "multi"}, f"head_mode invalide: {self.head_mode}"

        # AvgMax pooling
        self.feature_head = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
        feat_dim = 2 * self.backbone.feat_dim
        hidden = feat_dim // 2

        def _pool_features(z: torch.Tensor) -> torch.Tensor:
            a = self.feature_head[0](z).flatten(1)
            m = self.feature_head[1](z).flatten(1)
            return torch.cat([a, m], dim=1)
        self._pool_features = _pool_features

        # --- shared head: skip + gating encapsulé dans un nn.Module ---
        self.shared_head = GatedSharedHead(feat_dim, hidden, p_drop=self.mlp_dropout)

        # --- têtes de sortie (assure l'existence des deux attributs) ---
        if self.head_mode == "single":
            self.out_head = nn.Linear(hidden, len(self.predict_params))
            self.out_heads = None
        else:
            self.out_heads = nn.ModuleDict({pname: nn.Linear(hidden, 1) for pname in self.predict_params})
            self.out_head = None

        # ===== Raffineurs B/C/D =====
        base_refiner = EfficientNetRefiner(
            m_params=len(self.predict_params),
            delta_scale=refine_delta_scale,
            encoder_variant=refiner_variant,
            encoder_width_mult=refiner_width_mult,
            encoder_depth_mult=refiner_depth_mult,
            encoder_stem_channels=refiner_stem_channels,
            encoder_drop_path=refiner_drop_path,
            encoder_se_ratio=refiner_se_ratio,
            feature_pool=refiner_feature_pool,
            hidden_dim=max(64, int(self.backbone.feat_dim * refiner_shared_hidden_scale)),
            mlp_dropout=self.refiner_mlp_dropout
        )
        self.refiner = base_refiner
        self.cascade_stages = 3

        extra_refiners = [EfficientNetRefiner(
            m_params=len(self.predict_params),
            delta_scale=refine_delta_scale,
            encoder_variant=refiner_variant,
            encoder_width_mult=refiner_width_mult,
            encoder_depth_mult=refiner_depth_mult,
            encoder_stem_channels=refiner_stem_channels,
            encoder_drop_path=refiner_drop_path,
            encoder_se_ratio=refiner_se_ratio,
            feature_pool=refiner_feature_pool,
            hidden_dim=max(64, int(self.backbone.feat_dim * refiner_shared_hidden_scale)),
            mlp_dropout=self.refiner_mlp_dropout
        )
            for _ in range(self.cascade_stages - 1)
        ]
        self.refiners = nn.ModuleList([base_refiner] + extra_refiners)

        # ===== ReLoBRaLo =====
        self.loss_names_params = [f"param_{p}" for p in self.predict_params]
        self.relo_params = ReLoBRaLoLoss(num_losses=len(self.predict_params), alpha=0.9, tau=1.0, history_len=10)
        if self.use_relobralo_top:
            self.loss_names_top = ["phys_pointwise", "phys_spectral", "phys_peak", "param_group"]
            self.relo_top = ReLoBRaLoLoss(
                loss_names=self.loss_names_top,
                alpha=0.9,      # Momentum pour EMA
                tau=1.0,        # Température pour softmax
                history_len=10  # Historique pour stabilité
            )
        else:
            self.relo_top = None

        # ----- Stage override -----
        self._override_stage: Optional[str] = None
        self._override_refine_steps: Optional[int] = None
        self._override_delta_scale: Optional[float] = None

        # ----- Masques de raffinement -----
        def _target_mask(names: List[str]) -> torch.Tensor:
            m = torch.zeros(len(self.predict_params), dtype=torch.float32)
            idxs = [self.predict_params.index(n) for n in names if n in self.predict_params]
            if idxs:
                m[torch.tensor(idxs, dtype=torch.long)] = 1.0
            return m

        base_targets = ["sig0", "dsig", "mf_CH4", "mf_H2O"]
        pt_targets = base_targets + ["P", "T"]

        self.register_buffer("refine_mask_base", _target_mask(base_targets))
        self.register_buffer("refine_mask_with_PT", _target_mask(pt_targets))

    # ==== Helpers baseline poly3 sur bords ====
    @staticmethod
    def _design_poly3(n: int, device, dtype):
        x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        xc = x - x.mean()
        X = torch.stack([torch.ones_like(xc), xc, xc**2, xc**3], dim=1)  # [N,4]
        return X

    @classmethod
    def _baseline_poly3_from_edges(cls, resid: torch.Tensor, left_frac: float = 0.20, right_start: float = 0.75):
        B, N = resid.shape
        device, dtype = resid.device, resid.dtype
        Xfull = cls._design_poly3(N, device, dtype)                # [N,4]
        iL = torch.arange(0, max(1, int(N*left_frac)), device=device)
        iR = torch.arange(int(N*right_start), N, device=device)
        idx = torch.cat([iL, iR], dim=0)                           # [M]
        X = Xfull[idx]                                             # [M,4]
        Xt = X.t()                                                 # [4,M]
        XtX = Xt @ X                                               # [4,4]
        XtX = XtX + 1e-6 * torch.eye(4, device=device, dtype=dtype)
        P = torch.linalg.solve(XtX, Xt)                            # [4,M]
        y_edges = resid[:, idx]                                    # [B,M]
        coeff = (P @ y_edges.T).T                                  # [B,4]
        baseline = (Xfull @ coeff.transpose(0,1)).transpose(0,1)   # [B,N]
        resid_corr = resid - baseline
        return resid_corr, baseline

    def set_stage_mode(self, mode: Optional[str], refine_steps: Optional[int]=None, delta_scale: Optional[float]=None):
        if mode is not None:
            mode = mode.upper()
            assert mode in {'A','B1','B2','DEN'}
        self._override_stage = mode
        self._override_refine_steps = refine_steps
        self._override_delta_scale = delta_scale
        if delta_scale is not None:
            for r in self.refiners:
                r.delta_scale = float(delta_scale)
        if refine_steps is not None:
            self.refine_steps = int(refine_steps)

    # ---- utils tête ----
    def _predict_params_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.shared_head(feat)
        if self.head_mode == "multi" and self.out_heads is not None:
            outs = []
            for name in self.predict_params:
                y = self.out_heads[name](h)
                outs.append(torch.sigmoid(y))
            return torch.cat(outs, dim=1)
        elif self.head_mode == "single" and self.out_head is not None:
            y = self.out_head(h)
            return torch.sigmoid(y)
        else:
            raise RuntimeError("Configuration des têtes incohérente avec head_mode.")

    def encode(self, spectra: torch.Tensor, pooled: bool=True, detach: bool=False):
        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat = self._pool_features(latent) if pooled else latent
        return feat.detach() if detach else feat

    # ---- (dé)normalisation & physique ----
    def _denorm_params_subset(self, y_norm_subset: torch.Tensor, names: List[str]) -> torch.Tensor:
        cols = [unnorm_param_torch(n, y_norm_subset[:, i]) for i, n in enumerate(names)]
        return torch.stack(cols, dim=1)

    def _compose_full_phys(self, pred_phys: torch.Tensor, provided_phys_tensor: torch.Tensor) -> torch.Tensor:
        B = pred_phys.shape[0]
        full = pred_phys.new_empty((B, self.n_params))
        for j, idx in enumerate(self.predict_idx):
            full[:, idx] = pred_phys[:, j]
        for j, idx in enumerate(self.provided_idx):
            full[:, idx] = provided_phys_tensor[:, j]
        return full

    def _physics_reconstruction(self, y_phys_full: torch.Tensor, device, scale: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Reconstruit le spectre depuis les paramètres physiques dénormalisés.
        VERSION CORRIGÉE : Gère baseline0 via LOWESS quand il n'est pas prédit.
        """
        B = y_phys_full.shape[0]
        p = {k: y_phys_full[:, i] for i, k in enumerate(self.param_names)}

        v_grid_idx = torch.arange(self.n_points, dtype=torch.float64, device=device)

        # Préparer les fractions molaires
        mf_dict = {}
        for mol in self.transitions_dict.keys():
            key = f"mf_{mol}"
            mf_dict[mol] = p[key] if key in p else torch.zeros_like(p['P'])

        # Gestion de baseline0
        if 'baseline0' in self.predict_params:
            # CAS 1 : baseline0 est PRÉDIT - on l'utilise directement
            b0_idx = self.name_to_idx['baseline0']
            baseline_coeffs = y_phys_full[:, b0_idx:b0_idx+3]

        else:
            # CAS 2 : baseline0 est CALCULÉ via LOWESS (c'est le cas actuel)
            b1_idx = self.name_to_idx['baseline1']
            b2_idx = self.name_to_idx['baseline2']

            # Utiliser baseline0 = 1.0 (sera normalisé après)
            baseline_coeffs = torch.stack([
                torch.ones(B, device=device, dtype=torch.float64),  # b0 = 1.0
                y_phys_full[:, b1_idx].to(torch.float64),           # b1 prédit
                y_phys_full[:, b2_idx].to(torch.float64)            # b2 prédit
            ], dim=1)

        # Génération du spectre
        spectra, _ = batch_physics_forward_multimol_vgrid(
            p['sig0'], p['dsig'], self.poly_freq_CH4, v_grid_idx,
            baseline_coeffs, self.transitions_dict, p['P'], p['T'], mf_dict,
            tipspy=self.tipspy, device=device
        )
        spectra = spectra.to(torch.float32)

        # Normalisation finale par LOWESS
        scale_recon = lowess_value(spectra, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)
        spectra = spectra / scale_recon

        return spectra

    # ---------- Savitzky–Golay derivative (torch) ----------
    def _savgol_coeffs(self, window_length: int, polyorder: int, deriv: int = 1, delta: float = 1.0, device=None, dtype=torch.float64) -> torch.Tensor:
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
        A = torch.stack([x**j for j in range(P + 1)], dim=1)
        pinv = torch.linalg.pinv(A)
        coeff = math.factorial(deriv) * pinv[deriv, :] / (delta ** deriv)
        return coeff.to(dtype=torch.float32)

    def _savgol_deriv(self, y: torch.Tensor, window_length: int, polyorder: int, deriv: int = 1) -> torch.Tensor:
        B, N = y.shape
        W = int(window_length)
        if W % 2 == 0:
            W += 1
        if W > N:
            W = N if (N % 2 == 1) else (N - 1)
        W = max(W, 3)
        P = min(int(polyorder), W - 1)
        coeff = self._savgol_coeffs(W, P, deriv=deriv, device=y.device, dtype=torch.float64)  # [W]
        coeff = coeff.view(1, 1, -1)
        pad = (W - 1) // 2
        y1 = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
        out = F.conv1d(y1, coeff).squeeze(1)
        return out

    def _pearson_corr_basic(self, y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        y_hat = y_hat.float()
        y = y.float()
        y_hat_c = y_hat - y_hat.mean(dim=1, keepdim=True)
        y_c = y - y.mean(dim=1, keepdim=True)
        num = (y_hat_c * y_c).sum(dim=1)
        den = torch.sqrt((y_hat_c.pow(2).sum(dim=1) + eps) * (y_c.pow(2).sum(dim=1) + eps))
        corr = num / den
        return (1.0 - corr).mean()

    @staticmethod
    def _to_pdf(y: torch.Tensor, smooth_win: int = 0, eps: float = 1e-12) -> torch.Tensor:
        if smooth_win and smooth_win > 1:
            pad = (smooth_win - 1) // 2
            k = torch.ones(1, 1, smooth_win, device=y.device, dtype=y.dtype) / float(smooth_win)
            yy = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
            y = F.conv1d(yy, k).squeeze(1)
        y = y.clamp_min(0)
        return y / (y.sum(dim=1, keepdim=True) + eps)

    @staticmethod
    def _kl(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        a = a.clamp_min(eps)
        b = b.clamp_min(eps)
        return (a * (a.log() - b.log())).sum(dim=1)

    def _js(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        m = 0.5 * (p + q)
        return 0.5 * self._kl(p, m, eps) + 0.5 * self._kl(q, m, eps)

    def _common_step(self, batch, step_name: str):
        """Step commun pour train/val/test."""
        noisy, clean, params_true_norm = batch['noisy_spectra'], batch['clean_spectra'], batch['params']
        scale = batch.get('scale', None)
        if scale is not None:
            scale = scale.to(clean.device)

        # ========== FORWARD PASS BASE MODEL A ==========
        latent, _ = self.backbone(noisy.unsqueeze(1))
        feat_shared = self._pool_features(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared)

        # Paramètres fournis
        if len(self.provided_params) > 0:
            provided_cols = [params_true_norm[:, self.name_to_idx[n]] for n in self.provided_params]
            provided_norm_tensor = torch.stack(provided_cols, dim=1)
            provided_phys_tensor = self._denorm_params_subset(provided_norm_tensor, self.provided_params)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((noisy.shape[0], 0))

        # Récupérer les paramètres expérimentaux
        use_exp_params_in_A = getattr(self, '_use_exp_params_in_A', [])

        # ========== STAGE A: Reconstruction ==========
        spectra_recon = None

        if self._override_stage == 'A' and len(use_exp_params_in_A) > 0:
            params_for_recon_norm = params_pred_norm.clone()
            for param_name in use_exp_params_in_A:
                if param_name in self.predict_params:
                    idx = self.predict_params.index(param_name)
                    params_for_recon_norm[:, idx] = params_true_norm[:, self.name_to_idx[param_name]]

            pred_phys_recon = self._denorm_params_subset(params_for_recon_norm, self.predict_params)
            y_phys_full_recon = self._compose_full_phys(pred_phys_recon, provided_phys_tensor)
            spectra_recon = self._physics_reconstruction(y_phys_full_recon, clean.device, scale=None)

        # ========== RAFFINEMENT ==========
        e = self.current_epoch
        effective_refine_steps = 0 if e < self.refine_warmup_epochs else self.refine_steps

        if self._override_stage is not None:
            if self._override_stage == 'A':
                effective_refine_steps = 0
            elif self._override_stage in ('B1', 'B2', 'C', 'D'):
                effective_refine_steps = self.refine_steps

        target_for_resid = noisy if self.refine_target == "noisy" else clean
        n_stages = min(effective_refine_steps, len(self.refiners))
        mask_now = self.refine_mask_base.to(clean.device, dtype=params_pred_norm.dtype)

        use_exp_params_for_resid = getattr(self, '_use_exp_params_for_resid', [])

        for k in range(n_stages):
            if len(use_exp_params_for_resid) > 0:
                params_for_resid_norm = params_pred_norm.clone()
                for param_name in use_exp_params_for_resid:
                    if param_name in self.predict_params:
                        idx = self.predict_params.index(param_name)
                        params_for_resid_norm[:, idx] = params_true_norm[:, self.name_to_idx[param_name]]

                pred_phys_for_resid = self._denorm_params_subset(params_for_resid_norm, self.predict_params)
                y_phys_full_for_resid = self._compose_full_phys(pred_phys_for_resid, provided_phys_tensor)
                spectra_recon_for_resid = self._physics_reconstruction(y_phys_full_for_resid, clean.device, scale=None)
                resid = spectra_recon_for_resid - target_for_resid
            else:
                pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
                y_phys_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
                spectra_recon = self._physics_reconstruction(y_phys_full, clean.device, scale=None)
                resid = spectra_recon - target_for_resid

            resid_corr, _ = self._baseline_poly3_from_edges(resid, left_frac=0.20, right_start=0.75)

            resid_for_refiner = resid_corr
            if self.use_denoiser and self._override_stage in (None, 'B1', 'B2'):
                with torch.no_grad():
                    resid_for_refiner = self.denoiser(resid_corr)

            delta = self.refiners[k](resid_for_refiner)
            params_pred_norm = (params_pred_norm + delta * mask_now).clamp(1e-4, 1-1e-4)

        # ========== RECONSTRUCTION FINALE ==========
        if spectra_recon is None:
            pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
            y_phys_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
            spectra_recon = self._physics_reconstruction(y_phys_full, clean.device, scale=None)

        # ========== CALCUL DES LOSSES BRUTES ==========
        yh = spectra_recon
        yt = clean

        # Losses spectrales (brutes, sans poids)
        loss_pointwise_raw = F.mse_loss(yh, yt)
        loss_spectral_raw = self.spectral_angle_loss(yh, yt)

        loss_peak_raw = torch.tensor(0.0, device=clean.device)
        if self.w_peak > 0:
            loss_peak_raw = self.peak_weighted_loss(yh, yt)

        # Loss params (brute)
        per_param_losses = []
        per_param_names = []

        for j, name in enumerate(self.predict_params):
            if len(use_exp_params_in_A) > 0 and name in use_exp_params_in_A:
                continue

            true_j = params_true_norm[:, self.name_to_idx[name]]
            mult = self.weight_mf if name in ("mf_CH4", "mf_H2O") else 1.0
            lp = mult * F.mse_loss(params_pred_norm[:, j], true_j)
            per_param_losses.append(lp)
            per_param_names.append(name)

        loss_params_raw = torch.mean(torch.stack(per_param_losses)) if per_param_losses else torch.tensor(0.0, device=clean.device)

        # ========== APPLICATION DES POIDS (avec ou sans ReLoBRaLo) ==========
        if self.use_relobralo_top and step_name == "train" and self.relo_top is not None:
            # ReLoBRaLo équilibre automatiquement
            losses_tensor = torch.stack([
                loss_pointwise_raw,
                loss_spectral_raw,
                loss_peak_raw,
                loss_params_raw,
            ])  # Shape: [4]

            # Appeler ReLoBRaLo (retourne loss totale ET poids)
            loss_main = self.relo_top(losses_tensor)

            # Récupérer les poids pour logging
            if hasattr(self.relo_top, 'last_weights'):
                weights = self.relo_top.last_weights
            else:
                weights = torch.ones_like(losses_tensor) / len(losses_tensor)

            # Pour logging (losses pondérées individuelles)
            loss_pointwise = weights[0].item() * loss_pointwise_raw
            loss_spectral = weights[1].item() * loss_spectral_raw
            loss_peak = weights[2].item() * loss_peak_raw
            loss_params = weights[3].item() * loss_params_raw

            # Logger les poids adaptatifs
            self.log("relo_weight_phys_pointwise", weights[0].item(), on_epoch=True, sync_dist=True)
            self.log("relo_weight_phys_spectral", weights[1].item(), on_epoch=True, sync_dist=True)
            self.log("relo_weight_phys_peak", weights[2].item(), on_epoch=True, sync_dist=True)
            self.log("relo_weight_param_group", weights[3].item(), on_epoch=True, sync_dist=True)

        else:
            # Poids fixes (pour validation ou si ReLoBRaLo désactivé)
            loss_pointwise = self.w_pw_raw * loss_pointwise_raw
            loss_spectral = self.w_spectral_angle * loss_spectral_raw
            loss_peak = self.w_peak * loss_peak_raw
            loss_params = self.w_params * loss_params_raw

            loss_main = loss_pointwise + loss_spectral + loss_peak + loss_params

        # ========== DENOISER (si applicable) ==========
        if self._override_stage == 'DEN':
            resid_den = spectra_recon - noisy
            resid_den_corr, _ = self._baseline_poly3_from_edges(resid_den, left_frac=0.20, right_start=0.75)
            resid_clean = spectra_recon - clean
            resid_clean_corr, _ = self._baseline_poly3_from_edges(resid_clean, left_frac=0.20, right_start=0.75)
            resid_hat = self.denoiser(resid_den_corr)
            denoiser_loss = F.mse_loss(resid_hat, resid_clean_corr)
            loss = denoiser_loss
            self.log(f"{step_name}_loss_denoiser", denoiser_loss, on_epoch=True, sync_dist=True)
        else:
            loss = loss_main

        # ========== LOGGING ==========
        self.log(f"{step_name}_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)

        # Losses pondérées
        self.log(f"{step_name}_loss_pointwise", loss_pointwise, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_spectral", loss_spectral, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_peak", loss_peak, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_params", loss_params, on_epoch=True, sync_dist=True)

        # Losses BRUTES (pour voir les magnitudes réelles)
        self.log(f"{step_name}_loss_pointwise_raw", loss_pointwise_raw, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_spectral_raw", loss_spectral_raw, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_peak_raw", loss_peak_raw, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_params_raw", loss_params_raw, on_epoch=True, sync_dist=True)

        if len(use_exp_params_in_A) > 0:
            self.log(f"{step_name}_n_exp_params_A", float(len(use_exp_params_in_A)), on_epoch=True, sync_dist=True)

        if len(use_exp_params_for_resid) > 0:
            self.log(f"{step_name}_n_exp_params_resid", float(len(use_exp_params_for_resid)), on_epoch=True, sync_dist=True)

        # Logger les losses individuelles par paramètre
        if len(per_param_losses) > 0:
            for name, lp in zip(per_param_names, per_param_losses):
                self.log(f"{step_name}_param_{name}", lp, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def on_train_epoch_start(self):
        def set_trainable(mod, train: bool):
            if mod is None:
                return
            mods = list(mod) if isinstance(mod, nn.ModuleList) else [mod]
            for m in mods:
                if m is None:
                    continue
                for p in m.parameters():
                    p.requires_grad_(train)

        # --------- Stages explicites ----------
        if self._override_stage is not None:
            st = self._override_stage

            if st == 'A':
                set_trainable(self.backbone, True)
                set_trainable(self.shared_head, True)
                set_trainable(getattr(self, "out_head", None), True)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items():
                        set_trainable(head, True)
                set_trainable(self.refiners, False)
                set_trainable(self.denoiser, False)
                return

            if st == 'DEN':
                set_trainable(self.backbone, False)
                set_trainable(self.shared_head, False)
                set_trainable(getattr(self, "out_head", None), False)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items():
                        set_trainable(head, False)
                set_trainable(self.refiners, False)
                set_trainable(self.denoiser, True)
                return

            if st == 'B1':
                set_trainable(self.backbone, False)
                set_trainable(self.shared_head, False)
                set_trainable(getattr(self, "out_head", None), False)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items():
                        set_trainable(head, False)
                set_trainable(self.refiners, True)
                set_trainable(self.denoiser, False)
                return

            if st == 'B2':
                set_trainable(self.backbone, True)
                set_trainable(self.shared_head, True)
                set_trainable(getattr(self, "out_head", None), True)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items():
                        set_trainable(head, True)
                set_trainable(self.refiners, True)
                set_trainable(self.denoiser, False)
                return

        # --------- Fallback 3 phases (A warmup → B → B2) ----------
        e = self.current_epoch
        stage3_start = self.refine_warmup_epochs + self.freeze_base_epochs

        if e < self.refine_warmup_epochs:
            self._set_requires_grad(self.refiners, False)
            self._set_requires_grad(
                [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None)],
                True
            )
            self._froze_base = False
        elif e < stage3_start:
            if not self._froze_base:
                self._set_requires_grad(
                    [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None)],
                    False
                )
                self._set_requires_grad(self.refiners, True)
                self._froze_base = True
        else:
            self._set_requires_grad(
                [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.refiners],
                True
            )
            if e == stage3_start:
                if hasattr(self.trainer, "optimizers") and len(self.trainer.optimizers) > 0:
                    opt = self.trainer.optimizers[0]
                    for pg in opt.param_groups:
                        pg["lr"] *= self.stage3_lr_shrink
                if self.stage3_refine_steps is not None:
                    self.refine_steps = int(self.stage3_refine_steps)
                if self.stage3_delta_scale is not None:
                    for r in self.refiners:
                        r.delta_scale = float(self.stage3_delta_scale)

    def _set_requires_grad(self, modules, flag: bool):
        if modules is None:
            return
        if isinstance(modules, nn.ModuleList):
            modules = list(modules)
        if not isinstance(modules, (list, tuple)):
            modules = [modules]
        for m in modules:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad_(flag)

    def configure_optimizers(self):
        """Configure optimizer et scheduler personnalisé"""

        # ========== Paramètres à optimiser ==========
        # === BASE PARAMS (backbone + shared head + heads) ===
        base_params = list(self.backbone.parameters()) + list(self.shared_head.parameters())

        if getattr(self, "out_head", None) is not None:
            base_params += list(self.out_head.parameters())

        if getattr(self, "out_heads", None) is not None:
            base_params += list(self.out_heads.parameters())

        refiner_params = list(self.refiners.parameters())

        param_groups = [
            {"params": base_params, "lr": float(getattr(self, "base_lr", self.lr))},
            {"params": refiner_params, "lr": float(getattr(self, "refiner_lr", self.lr))},
        ]

        if getattr(self, "use_denoiser", False):
            param_groups.append({
                "params": self.denoiser.parameters(),
                "lr": float(self.denoiser_lr)
            })

        # ========== Optimizer ==========
        opt_name = getattr(self.hparams, "optimizer", "adamw").lower()
        weight_decay = getattr(self, "weight_decay", 1e-4)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        elif opt_name == "lion":
            try:
                from lion_pytorch import Lion
                betas = getattr(self.hparams, "betas", (0.9, 0.99))
                optimizer = Lion(param_groups, betas=betas, weight_decay=weight_decay)
            except ImportError:
                print("Warning: lion_pytorch not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # ========== Scheduler personnalisé ==========
        scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
            optimizer,
            T_0=self.scheduler_T_0,
            T_mult=self.scheduler_T_mult,
            eta_min=self.scheduler_eta_min,
            decay_factor=self.scheduler_decay_factor,
            warmup_epochs=self.scheduler_warmup_epochs,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    @torch.no_grad()
    def infer(
        self,
        spectra: torch.Tensor,
        provided_phys: dict,
        *,
        refine: bool = True,
        resid_target: str = "input",
        scale: Optional[torch.Tensor] = None,
        recon_PT: str = "pred",             # "pred" | "exp"
        Pexp: Optional[torch.Tensor] = None,
        Texp: Optional[torch.Tensor] = None,
        cascade_stages_override: Optional[int] = None,
    ):
        """
        Inference method with optional refinement and PT override.

        Args:
            spectra: Input spectra [B, N].
            provided_phys: Dictionary of provided physical parameters.
            refine: Apply refinement cascade (default: True).
            resid_target: Target for residual ("input" or "noisy").
            scale: Optional scale tensor.
            recon_PT: Use predicted or experimental P,T ("pred" or "exp").
            Pexp: Experimental pressure (required if recon_PT="exp").
            Texp: Experimental temperature (required if recon_PT="exp").
            cascade_stages_override: Override number of refinement stages.

        Returns:
            Dictionary with predicted parameters, reconstructed spectra, etc.
        """
        self.eval()
        device = spectra.device
        B = spectra.shape[0]

        if recon_PT not in ("pred", "exp"):
            raise ValueError("recon_PT doit être 'pred' ou 'exp'.")
        if recon_PT == "exp":
            if Pexp is None or Texp is None:
                raise ValueError("En mode recon_PT='exp', fournir Pexp et Texp.")
            Pexp = torch.as_tensor(Pexp, device=device, dtype=torch.float32).view(B)
            Texp = torch.as_tensor(Texp, device=device, dtype=torch.float32).view(B)

        missing = [n for n in self.provided_params if n not in provided_phys]
        assert not missing, f"Manque des paramètres fournis: {missing}"

        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat_shared = self._pool_features(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared)

        if len(self.provided_params) > 0:
            provided_list = [provided_phys[n].to(device) for n in self.provided_params]
            provided_phys_tensor = torch.stack(provided_list, dim=1)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((B, 0))

        spectra_target = spectra if resid_target in ("input", "noisy") else None
        scale_est = lowess_value(spectra, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)

        def _compose_full_with_PT_override(pred_norm_subset: torch.Tensor) -> torch.Tensor:
            pred_phys_subset = self._denorm_params_subset(pred_norm_subset, self.predict_params)
            y_full = self._compose_full_phys(pred_phys_subset, provided_phys_tensor)
            if recon_PT == "exp":
                idxP = self.name_to_idx.get("P", None)
                idxT = self.name_to_idx.get("T", None)
                if idxP is not None:
                    y_full[:, idxP] = Pexp
                if idxT is not None:
                    y_full[:, idxT] = Texp
            return y_full

        # masque de raffinement
        if recon_PT == "exp":
            mask_now = self.refine_mask_with_PT.to(device, dtype=params_pred_norm.dtype)
        else:
            mask_now = self.refine_mask_base.to(device, dtype=params_pred_norm.dtype)

        # cascade
        n_stages = cascade_stages_override if cascade_stages_override is not None else self.cascade_stages
        n_stages = max(0, min(int(n_stages), len(self.refiners)))

        if refine and n_stages > 0:
            for k in range(n_stages):
                y_full_k = _compose_full_with_PT_override(params_pred_norm)
                recon_k = self._physics_reconstruction(y_full_k, device, scale=None)
                if spectra_target is None:
                    break
                resid = recon_k - spectra_target
                resid_corr, _ = self._baseline_poly3_from_edges(resid, left_frac=0.20, right_start=0.75)

                resid_for_refiner = resid_corr
                if self.use_denoiser:
                    resid_for_refiner = self.denoiser(resid_corr)

                delta_k = self.refiners[k](resid_for_refiner)

                params_pred_norm = params_pred_norm.add(delta_k * mask_now)

        y_full_final = _compose_full_with_PT_override(params_pred_norm)
        recon_final = self._physics_reconstruction(y_full_final, device, scale=None)

        return {
            "params_pred_norm": params_pred_norm,
            "y_phys_full": y_full_final,
            "spectra_recon": recon_final,
            "norm_scale": scale_est,
        }
