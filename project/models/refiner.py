"""
Parameter refinement model using EfficientNet encoder + MLP head.
"""
import torch
import torch.nn as nn
from .backbone import EfficientNetEncoder


class EfficientNetRefiner(nn.Module):
    """
    Simplified refiner: takes ONLY the residual as input.
    Directly predicts delta corrections from the residual.
    """

    def __init__(
        self,
        m_params: int,
        *,
        delta_scale: float = 0.1,
        encoder_variant: str = "s",
        encoder_width_mult: float = 1.0,
        encoder_depth_mult: float = 1.0,
        encoder_stem_channels: int | None = None,
        encoder_drop_path: float = 0.1,
        encoder_se_ratio: float = 0.25,
        feature_pool: str = "avg",
        hidden_dim: int = 128,
        mlp_dropout: float = 0.10,
    ):
        """
        Args:
            m_params: Number of parameters to refine.
            delta_scale: Scale factor for delta predictions.
            encoder_variant: EfficientNet variant.
            encoder_width_mult: Width multiplier for encoder.
            encoder_depth_mult: Depth multiplier for encoder.
            encoder_stem_channels: Stem channels for encoder.
            encoder_drop_path: Drop path rate for encoder.
            encoder_se_ratio: SE ratio for encoder.
            feature_pool: Feature pooling mode ('avg', 'max', 'avgmax').
            hidden_dim: Hidden dimension for MLP.
            mlp_dropout: Dropout rate for MLP.
        """
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.m_params = int(m_params)
        self.mlp_dropout = float(mlp_dropout)

        # Encoder: 1 channel (residual)
        self.encoder = EfficientNetEncoder(
            in_channels=1,
            variant=encoder_variant,
            width_mult=encoder_width_mult,
            depth_mult=encoder_depth_mult,
            se_ratio=encoder_se_ratio,
            drop_path_rate=encoder_drop_path,
            stem_channels=encoder_stem_channels,
        )

        D = self.encoder.feat_dim
        pool = feature_pool.lower()
        self._feature_pool_mode = pool

        if pool == "avg":
            self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        elif pool == "max":
            self.feature_head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        elif pool == "avgmax":
            self.feature_head = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
        else:
            raise ValueError(f"Unknown feature_pool: {feature_pool}")

        in_dim = D if pool != "avgmax" else 2 * D
        H = hidden_dim

        # MLP: features → hidden
        self.mlp_hidden = nn.Sequential(
            nn.Linear(in_dim, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),

            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
        )

        # Adaptive gate per parameter (AFTER MLP)
        self.scale_gate = nn.Linear(H, m_params)

        # Delta prediction head
        self.delta_head = nn.Linear(H, m_params)

    def _pool_features(self, latent: torch.Tensor) -> torch.Tensor:
        """Pool features from encoder output."""
        if self._feature_pool_mode == "avg":
            return self.feature_head(latent)
        if self._feature_pool_mode == "max":
            return self.feature_head(latent)
        avgp, maxp = self.feature_head
        a = avgp(latent).flatten(1)
        m = maxp(latent).flatten(1)
        return torch.cat([a, m], dim=1)

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict parameter corrections from residual.

        Args:
            resid: [B, N] residual (recon - target).

        Returns:
            delta: [B, M] parameter corrections.
        """
        # Encode residual
        x = resid.unsqueeze(1)  # [B, 1, N]
        latent, _ = self.encoder(x)  # [B, D, N']
        feat = self._pool_features(latent)  # [B, in_dim]

        # Pass through MLP
        h = self.mlp_hidden(feat)  # [B, H]

        # Adaptive gate (controls amplitude per parameter)
        gate = torch.sigmoid(self.scale_gate(h))  # [B, M]
        scale = self.delta_scale * gate

        # Predict raw corrections
        raw_delta = self.delta_head(h)  # [B, M]

        # Bounded corrections
        delta = torch.tanh(raw_delta) * scale  # [B, M]
        return delta
