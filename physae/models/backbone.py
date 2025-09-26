"""1D EfficientNet-like encoder blocks used by PhysAE."""

from __future__ import annotations

import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * torch.sigmoid(x)


class SqueezeExcitation1D(nn.Module):
    def __init__(self, in_channels: int, reduced_dim: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_dim, 1),
            SiLU(),
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * self.se(x)


class MBConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, expand_ratio: int,
                 se_ratio: float = 0.25) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        reduced_dim = max(1, int(in_channels * se_ratio))
        num_groups = 8

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                    nn.GroupNorm(num_groups, hidden_dim),
                    SiLU(),
                )
            )

        layers.extend(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel,
                        stride,
                        padding=kernel // 2,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(num_groups, hidden_dim),
                    SiLU(),
                ),
                SqueezeExcitation1D(hidden_dim, reduced_dim),
                nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
                nn.GroupNorm(max(1, out_channels // num_groups), out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.conv(x) if self.use_residual else self.conv(x)


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, 2, 1, bias=False),
            nn.GroupNorm(8, 32),
            SiLU(),
        )
        self.block1 = MBConvBlock1D(32, 24, 3, 2, 1)
        self.block2 = MBConvBlock1D(24, 40, 5, 2, 6)
        self.block3 = MBConvBlock1D(40, 80, 3, 2, 6)
        self.block4 = MBConvBlock1D(80, 112, 5, 1, 6)
        self.block5 = MBConvBlock1D(112, 192, 5, 2, 6)
        self.feat_dim = 192

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x, None
