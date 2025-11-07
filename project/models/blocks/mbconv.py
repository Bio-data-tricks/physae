"""
Mobile Convolution blocks (MBConv and FusedMBConv) for efficient 1D networks.
"""
import torch
import torch.nn as nn
from .convolutions import ConvBNAct1d, DropPath
from .attention import SE1d


class FusedMBConv1d(nn.Module):
    """Fused Mobile Convolution block (1D) - efficient inverted residual with fused operations."""

    def __init__(self, in_c, out_c, k, s, expand_ratio, drop_path=0.0):
        """
        Args:
            in_c: Input channels.
            out_c: Output channels.
            k: Kernel size.
            s: Stride.
            expand_ratio: Expansion ratio for hidden dimension.
            drop_path: Drop path probability.
        """
        super().__init__()
        mid = int(in_c * expand_ratio)
        self.use_res = (s == 1 and in_c == out_c)

        if expand_ratio != 1:
            self.fused = nn.Sequential(
                ConvBNAct1d(in_c, mid, k=k, s=s, p=None, g=1, act=True),
                ConvBNAct1d(mid, out_c, k=1, s=1, act=False),
            )
        else:
            self.fused = nn.Sequential(
                ConvBNAct1d(in_c, out_c, k=k, s=s, p=None, g=1, act=True),
            )
        self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # LayerScale (gamma) — starts close to identity
        self.gamma = nn.Parameter(torch.ones(1, out_c, 1) * 1e-3)

    def forward(self, x):
        y = self.fused(x)
        if self.use_res:
            y = x + self.drop(self.gamma * y)
        return y


class MBConv1d(nn.Module):
    """Standard Mobile Convolution block (1D) - depthwise separable convolution."""

    def __init__(self, in_c, out_c, k, s, expand_ratio, se_ratio=0.25, drop_path=0.0, dw_dilation: int = 1):
        """
        Args:
            in_c: Input channels.
            out_c: Output channels.
            k: Kernel size.
            s: Stride.
            expand_ratio: Expansion ratio for hidden dimension.
            se_ratio: Squeeze-Excitation ratio.
            drop_path: Drop path probability.
            dw_dilation: Dilation for depthwise convolution.
        """
        super().__init__()
        mid = int(in_c * expand_ratio)
        self.use_res = (s == 1 and in_c == out_c)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct1d(in_c, mid, k=1, s=1, act=True))

        # Depthwise (moderate configurable dilation)
        layers.append(ConvBNAct1d(mid, mid, k=k, s=s, g=mid, act=True, d=dw_dilation))

        if se_ratio is not None and se_ratio > 0:
            layers.append(SE1d(mid, se_ratio=se_ratio))

        layers.append(ConvBNAct1d(mid, out_c, k=1, s=1, act=False))
        self.block = nn.Sequential(*layers)

        self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(1, out_c, 1) * 1e-3)

    def forward(self, x):
        y = self.block(x)
        if self.use_res:
            y = x + self.drop(self.gamma * y)
        return y
