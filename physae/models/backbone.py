"""1D EfficientNet-like encoder blocks used by PhysAE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

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


def _resolve_groups(channels: int, desired: int) -> int:
    desired = max(1, int(desired))
    channels = max(1, int(channels))
    groups = math.gcd(channels, desired)
    return groups if groups > 0 else 1


class MBConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        expand_ratio: int,
        *,
        se_ratio: float = 0.25,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        reduced_dim = max(1, int(in_channels * se_ratio))
        norm_groups = max(1, int(norm_groups))

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                    nn.GroupNorm(_resolve_groups(hidden_dim, norm_groups), hidden_dim),
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
                    nn.GroupNorm(_resolve_groups(hidden_dim, norm_groups), hidden_dim),
                    SiLU(),
                ),
                SqueezeExcitation1D(hidden_dim, reduced_dim),
                nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
                nn.GroupNorm(_resolve_groups(out_channels, norm_groups), out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.conv(x) if self.use_residual else self.conv(x)


@dataclass(frozen=True)
class MBConvConfig:
    out_channels: int
    kernel: int
    stride: int
    expand_ratio: int
    repeats: int = 1


def _round_channels(value: int, multiplier: float) -> int:
    scaled = int(round(value * multiplier))
    return max(4, scaled)


def _round_repeats(value: int, multiplier: float) -> int:
    scaled = int(round(value * multiplier))
    return max(1, scaled)


class EfficientNetEncoder(nn.Module):
    DEFAULT_BLOCKS: Sequence[MBConvConfig] = (
        MBConvConfig(out_channels=24, kernel=3, stride=2, expand_ratio=1),
        MBConvConfig(out_channels=40, kernel=5, stride=2, expand_ratio=6),
        MBConvConfig(out_channels=80, kernel=3, stride=2, expand_ratio=6),
        MBConvConfig(out_channels=112, kernel=5, stride=1, expand_ratio=6),
        MBConvConfig(out_channels=192, kernel=5, stride=2, expand_ratio=6),
    )

    def __init__(
        self,
        in_channels: int = 1,
        *,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        se_ratio: float = 0.25,
        expand_ratio_scale: float = 1.0,
        norm_groups: int = 8,
        block_settings: Sequence[MBConvConfig] | None = None,
    ) -> None:
        super().__init__()
        block_settings = tuple(block_settings or self.DEFAULT_BLOCKS)
        width_mult = float(width_mult)
        depth_mult = float(depth_mult)
        expand_ratio_scale = float(expand_ratio_scale)
        se_ratio = float(se_ratio)
        norm_groups = int(norm_groups)

        stem_out = _round_channels(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_out, 3, 2, 1, bias=False),
            nn.GroupNorm(_resolve_groups(stem_out, norm_groups), stem_out),
            SiLU(),
        )

        blocks = []
        in_channels_current = stem_out
        for config in block_settings:
            out_channels = _round_channels(config.out_channels, width_mult)
            expand_ratio = max(1, int(round(config.expand_ratio * expand_ratio_scale)))
            repeats = _round_repeats(config.repeats, depth_mult)
            for repeat_idx in range(repeats):
                stride = config.stride if repeat_idx == 0 else 1
                block_in = in_channels_current if repeat_idx == 0 else out_channels
                blocks.append(
                    MBConvBlock1D(
                        block_in,
                        out_channels,
                        config.kernel,
                        stride,
                        expand_ratio,
                        se_ratio=se_ratio,
                        norm_groups=norm_groups,
                    )
                )
                in_channels_current = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.feat_dim = in_channels_current

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        x = self.stem(x)
        x = self.blocks(x)
        return x, None
