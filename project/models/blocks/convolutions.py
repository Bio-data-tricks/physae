"""
Basic convolutional building blocks for 1D signals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """Sigmoid Linear Unit activation function."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class DropPath(nn.Module):
    """Stochastic Depth (per sample) for regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def Norm1d(c):
    """Create 1D normalization layer (GroupNorm wrapper)."""
    groups = 8 if c >= 8 else 1
    return nn.GroupNorm(groups, c)


class ConvBNAct1d(nn.Sequential):
    """Sequential combination of Conv1d + BatchNorm + Activation."""

    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, bias=False, act=True, d=1):
        """
        Args:
            in_c: Input channels.
            out_c: Output channels.
            k: Kernel size.
            s: Stride.
            p: Padding (auto-calculated if None).
            g: Groups.
            bias: Use bias.
            act: Use activation.
            d: Dilation.
        """
        if p is None:
            p = ((k - 1) // 2) * d
        mods = [nn.Conv1d(in_c, out_c, k, s, p, dilation=d, groups=g, bias=bias), Norm1d(out_c)]
        if act:
            mods.append(SiLU())
        super().__init__(*mods)


class BlurPool1D(nn.Module):
    """1D blur pooling for anti-aliasing downsampling."""

    def __init__(self, c, kernel=(1, 4, 6, 4, 1)):
        """
        Args:
            c: Number of channels.
            kernel: Blur kernel coefficients.
        """
        super().__init__()
        k = torch.tensor(kernel, dtype=torch.float32)
        k = (k / k.sum()).view(1, 1, -1).repeat(c, 1, 1)
        self.register_buffer('k', k)

    def forward(self, x):
        """x: [B, C, N]"""
        return F.conv1d(x, self.k, stride=2, padding=self.k.shape[-1] // 2, groups=x.size(1))


def _make_divisible(v, divisor=8, min_value=None):
    """Make number divisible by a divisor (used in EfficientNet)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # prevent going down by >10%
        new_v += divisor
    return new_v
