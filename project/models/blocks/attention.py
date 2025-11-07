"""
Attention mechanisms for 1D signals.
"""
import torch
import torch.nn as nn
from .convolutions import SiLU


class SE1d(nn.Module):
    """Squeeze-and-Excitation attention module (1D)."""

    def __init__(self, c, se_ratio=0.25):
        """
        Args:
            c: Number of channels.
            se_ratio: Squeeze ratio for hidden dimension.
        """
        super().__init__()
        hidden = max(1, int(c * se_ratio))
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(c, hidden, 1)
        self.act = SiLU()
        self.fc2 = nn.Conv1d(hidden, c, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.avg(x)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)


class ChannelAttention1d(nn.Module):
    """Enhanced Squeeze-and-Excitation with global context (1D)."""

    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: Number of channels.
            reduction: Reduction ratio for hidden dimension.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention1d(nn.Module):
    """Spatial attention on temporal/spectral dimension (1D)."""

    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: Convolutional kernel size.
        """
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class GatedSharedHead(nn.Module):
    """Gated shared head for feature refinement."""

    def __init__(self, feat_dim: int, hidden: int, p_drop: float):
        """
        Args:
            feat_dim: Input feature dimension.
            hidden: Hidden dimension.
            p_drop: Dropout probability.
        """
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
