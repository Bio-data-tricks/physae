"""1D EfficientNet-like encoder blocks used by PhysAE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Type

import torch
import torch.nn as nn

from .registry import register_encoder


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


class FusedMBConvBlock1D(nn.Module):
    """1D adaptation of the fused MBConv block used by EfficientNet-V2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        expand_ratio: int,
        *,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        norm_groups = max(1, int(norm_groups))

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        hidden_dim,
                        kernel,
                        stride,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(_resolve_groups(hidden_dim, norm_groups), hidden_dim),
                    SiLU(),
                    nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
                    nn.GroupNorm(_resolve_groups(out_channels, norm_groups), out_channels),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(_resolve_groups(out_channels, norm_groups), out_channels),
                    SiLU(),
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
    block_type: str = "mbconv"


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
            block_type = getattr(config, "block_type", "mbconv").lower()
            if block_type not in {"mbconv", "fused"}:
                raise ValueError(f"Unsupported block_type '{block_type}' in EfficientNet configuration.")
            block_cls: Type[nn.Module]
            block_kwargs: dict[str, Any]
            if block_type == "fused":
                block_cls = FusedMBConvBlock1D
                block_kwargs = {"norm_groups": norm_groups}
            else:
                block_cls = MBConvBlock1D
                block_kwargs = {"se_ratio": se_ratio, "norm_groups": norm_groups}

            for repeat_idx in range(repeats):
                stride = config.stride if repeat_idx == 0 else 1
                block_in = in_channels_current if repeat_idx == 0 else out_channels
                blocks.append(
                    block_cls(
                        block_in,
                        out_channels,
                        config.kernel,
                        stride,
                        expand_ratio,
                        **block_kwargs,
                    )
                )
                in_channels_current = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.feat_dim = in_channels_current

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        x = self.stem(x)
        x = self.blocks(x)
        return x, None


@register_encoder("efficientnet")
def build_efficientnet_encoder(**kwargs: Any) -> EfficientNetEncoder:
    """Factory compatible with :func:`register_encoder` to build the default encoder."""

    in_channels = int(kwargs.pop("in_channels", 1))
    return EfficientNetEncoder(in_channels=in_channels, **kwargs)


class EfficientNetLargeEncoder(EfficientNetEncoder):
    """More expressive EfficientNet variant tuned for higher-capacity experiments."""

    DEFAULT_BLOCKS: Sequence[MBConvConfig] = (
        MBConvConfig(out_channels=32, kernel=3, stride=2, expand_ratio=1, repeats=2),
        MBConvConfig(out_channels=56, kernel=5, stride=2, expand_ratio=6, repeats=4),
        MBConvConfig(out_channels=112, kernel=3, stride=2, expand_ratio=6, repeats=4),
        MBConvConfig(out_channels=160, kernel=5, stride=1, expand_ratio=6, repeats=6),
        MBConvConfig(out_channels=272, kernel=5, stride=2, expand_ratio=6, repeats=2),
    )

    def __init__(self, in_channels: int = 1, **kwargs: Any) -> None:
        width_mult = float(kwargs.pop("width_mult", 1.4))
        depth_mult = float(kwargs.pop("depth_mult", 1.8))
        expand_ratio_scale = float(kwargs.pop("expand_ratio_scale", 1.2))
        block_settings = tuple(kwargs.pop("block_settings", self.DEFAULT_BLOCKS))
        super().__init__(
            in_channels=in_channels,
            width_mult=width_mult,
            depth_mult=depth_mult,
            expand_ratio_scale=expand_ratio_scale,
            block_settings=block_settings,
            **kwargs,
        )


@register_encoder("efficientnet_large")
def build_efficientnet_large_encoder(**kwargs: Any) -> EfficientNetLargeEncoder:
    """Build the higher-capacity EfficientNet variant registered as ``efficientnet_large``."""

    in_channels = int(kwargs.pop("in_channels", 1))
    return EfficientNetLargeEncoder(in_channels=in_channels, **kwargs)


class EfficientNetV2Encoder(EfficientNetEncoder):
    """Hybrid MBConv/fused-MBConv encoder inspired by EfficientNet-V2."""

    DEFAULT_BLOCKS: Sequence[MBConvConfig] = (
        MBConvConfig(out_channels=24, kernel=3, stride=1, expand_ratio=1, repeats=2, block_type="fused"),
        MBConvConfig(out_channels=48, kernel=3, stride=2, expand_ratio=4, repeats=4, block_type="fused"),
        MBConvConfig(out_channels=64, kernel=3, stride=2, expand_ratio=4, repeats=4, block_type="fused"),
        MBConvConfig(out_channels=128, kernel=3, stride=2, expand_ratio=4, repeats=6, block_type="mbconv"),
        MBConvConfig(out_channels=160, kernel=5, stride=1, expand_ratio=6, repeats=9, block_type="mbconv"),
        MBConvConfig(out_channels=256, kernel=5, stride=2, expand_ratio=6, repeats=15, block_type="mbconv"),
    )

    def __init__(self, in_channels: int = 1, **kwargs: Any) -> None:
        width_mult = float(kwargs.pop("width_mult", 1.0))
        depth_mult = float(kwargs.pop("depth_mult", 1.0))
        expand_ratio_scale = float(kwargs.pop("expand_ratio_scale", 1.0))
        block_settings = tuple(kwargs.pop("block_settings", self.DEFAULT_BLOCKS))
        super().__init__(
            in_channels=in_channels,
            width_mult=width_mult,
            depth_mult=depth_mult,
            expand_ratio_scale=expand_ratio_scale,
            block_settings=block_settings,
            **kwargs,
        )


@register_encoder("efficientnet_v2")
def build_efficientnet_v2_encoder(**kwargs: Any) -> EfficientNetV2Encoder:
    """Register EfficientNet-V2 style encoder under ``efficientnet_v2``."""

    in_channels = int(kwargs.pop("in_channels", 1))
    return EfficientNetV2Encoder(in_channels=in_channels, **kwargs)


class LayerNorm1d(nn.Module):
    """LayerNorm operating on the channel dimension of 1D tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError("LayerNorm1d expects input of shape (B, C, L)")
        y = x.permute(0, 2, 1)
        mean = y.mean(dim=-1, keepdim=True)
        var = y.var(dim=-1, keepdim=True, unbiased=False)
        y = (y - mean) / torch.sqrt(var + self.eps)
        y = y * self.weight + self.bias
        return y.permute(0, 2, 1)


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt-style block with depthwise convolution in 1D."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, 1)
        self.layer_scale_init_value = float(layer_scale_init_value)
        if self.layer_scale_init_value > 0:
            self.gamma = nn.Parameter(self.layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1) * x
        return residual + x


class ConvNeXtEncoder(nn.Module):
    """High-capacity encoder leveraging ConvNeXt-style blocks."""

    def __init__(
        self,
        in_channels: int = 1,
        *,
        dims: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (3, 3, 9, 3),
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(dims) != len(depths):
            raise ValueError("dims and depths must have the same length")

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=4, stride=4, padding=1, bias=True)
        )
        self.downsample_layers.append(stem)

        for idx in range(1, len(dims)):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm1d(dims[idx - 1]),
                    nn.Conv1d(dims[idx - 1], dims[idx], kernel_size=2, stride=2, bias=True),
                )
            )

        self.stages = nn.ModuleList()
        for stage_idx, depth in enumerate(depths):
            blocks = [
                ConvNeXtBlock1D(
                    dims[stage_idx],
                    kernel_size=kernel_size,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(int(depth))
            ]
            self.stages.append(nn.Sequential(*blocks))

        self.out_norm = LayerNorm1d(dims[-1])
        self.feat_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        for stage_idx in range(1, len(self.stages)):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
        x = self.out_norm(x)
        return x, None


@register_encoder("convnext")
def build_convnext_encoder(**kwargs: Any) -> ConvNeXtEncoder:
    """Register a ConvNeXt-inspired high-capacity encoder."""

    in_channels = int(kwargs.pop("in_channels", 1))
    # Parameters inherited from EfficientNet config are ignored if provided.
    kwargs.pop("width_mult", None)
    kwargs.pop("depth_mult", None)
    kwargs.pop("expand_ratio_scale", None)
    kwargs.pop("se_ratio", None)
    kwargs.pop("norm_groups", None)
    return ConvNeXtEncoder(in_channels=in_channels, **kwargs)
