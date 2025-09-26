"""Refinement head used for iterative prediction updates."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import EfficientNetEncoder


class EfficientNetRefiner(nn.Module):
    def __init__(
        self,
        m_params: int,
        cond_dim: int,
        backbone_feat_dim: int,
        delta_scale: float = 0.1,
        *,
        encoder_width_mult: float = 1.0,
        encoder_depth_mult: float = 1.0,
        encoder_expand_ratio_scale: float = 1.0,
        encoder_se_ratio: float = 0.25,
        encoder_norm_groups: int = 8,
        hidden_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.m_params = int(m_params)
        self.cond_dim = int(cond_dim)
        self.use_film = True

        self.encoder = EfficientNetEncoder(
            in_channels=2,
            width_mult=encoder_width_mult,
            depth_mult=encoder_depth_mult,
            expand_ratio_scale=encoder_expand_ratio_scale,
            se_ratio=encoder_se_ratio,
            norm_groups=encoder_norm_groups,
        )
        self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        dim = self.encoder.feat_dim
        hidden = max(64, int(round(dim * float(hidden_scale))))

        self.shared_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.film = (
            nn.Sequential(nn.Linear(self.cond_dim, hidden), nn.Tanh(), nn.Linear(hidden, 2 * hidden))
            if self.cond_dim > 0
            else None
        )

        self.scale_gate = nn.Linear(hidden, m_params)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden + backbone_feat_dim + m_params, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, m_params),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        resid: torch.Tensor,
        params_pred_norm: torch.Tensor,
        cond_norm: torch.Tensor | None,
        feat_shared: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([noisy, resid], dim=1)
        latent, _ = self.encoder(x)
        features = self.feature_head(latent)
        hidden = self.shared_head(features)

        if self.film is not None and cond_norm is not None and self.use_film:
            gamma_beta = self.film(cond_norm)
            width = hidden.shape[1]
            gamma, beta = gamma_beta[:, :width], gamma_beta[:, width:]
            hidden = hidden * (1 + 0.1 * gamma) + 0.1 * beta

        gate = torch.sigmoid(self.scale_gate(hidden))
        scale = self.delta_scale * gate

        context = torch.cat([hidden, feat_shared, params_pred_norm], dim=1)
        raw = self.delta_head(context)
        delta = torch.tanh(raw) * scale
        return delta
