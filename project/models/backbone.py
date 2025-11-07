"""
EfficientNet-based encoder for 1D spectral signals.
"""
import math
import torch.nn as nn
from .blocks.convolutions import ConvBNAct1d, Norm1d, SiLU, BlurPool1D, _make_divisible
from .blocks.mbconv import FusedMBConv1d, MBConv1d


# EfficientNetV2 configurations
_EFFV2_CFGS = {
    "s": [
        ("fused", 2, 24, 3, 1, 1.0),
        ("fused", 4, 48, 3, 2, 4.0),
        ("fused", 4, 64, 3, 2, 4.0),
        ("mb", 4, 128, 3, 2, 4.0),
        ("mb", 6, 160, 3, 1, 6.0),
        ("mb", 8, 256, 3, 2, 6.0),
    ],
    "m": [
        ("fused", 3, 24, 3, 1, 1.0),
        ("fused", 5, 48, 3, 2, 4.0),
        ("fused", 5, 80, 3, 2, 4.0),
        ("mb", 5, 160, 3, 2, 6.0),
        ("mb", 7, 176, 3, 1, 6.0),
        ("mb", 10, 304, 3, 2, 6.0),
    ],
    "l": [
        ("fused", 4, 32, 3, 1, 1.0),
        ("fused", 7, 64, 3, 2, 4.0),
        ("fused", 7, 96, 3, 2, 4.0),
        ("mb", 7, 192, 3, 2, 6.0),
        ("mb", 10, 224, 3, 1, 6.0),
        ("mb", 14, 384, 3, 2, 6.0),
    ],
}


class EfficientNetEncoder(nn.Module):
    """EfficientNet-based encoder for 1D spectral signals (configurable variants)."""

    def __init__(
        self,
        in_channels=1,
        variant: str = "s",
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.1,
        stem_channels: int | None = None,
        use_blurpool: bool = True,
        mb_dilate_from_stage: int = 3,  # from this "stage" (0-index) for 'mb'
        mb_dilation_value: int = 2,
    ):
        """
        Args:
            in_channels: Number of input channels.
            variant: EfficientNet variant ('s', 'm', 'l').
            width_mult: Width multiplier for channels.
            depth_mult: Depth multiplier for layers.
            se_ratio: Squeeze-Excitation ratio.
            drop_path_rate: Drop path rate.
            stem_channels: Stem output channels (default: 32).
            use_blurpool: Use blur pooling for anti-aliasing.
            mb_dilate_from_stage: Stage index to start using dilation in MBConv.
            mb_dilation_value: Dilation value for MBConv.
        """
        super().__init__()
        cfg = _EFFV2_CFGS[variant]
        stem_c = stem_channels or 32
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_c, kernel_size=3, stride=2, padding=1, bias=False),
            Norm1d(stem_c),
            SiLU()
        )
        in_c = stem_c

        blocks = []
        total_blocks = sum(int(math.ceil(r * depth_mult)) for (_, r, *_) in cfg)
        b_idx = 0
        stage_idx = -1
        for (kind, repeats, out_c, k, s, exp) in cfg:
            stage_idx += 1
            out_c = _make_divisible(out_c * width_mult, 8)
            repeats = int(math.ceil(repeats * depth_mult))

            for i in range(repeats):
                stride = s if i == 0 else 1
                dp = drop_path_rate * b_idx / max(1, total_blocks - 1)

                # Anti-alias downsample: BlurPool before block, and block at stride=1
                pre = []
                if use_blurpool and stride == 2:
                    pre.append(BlurPool1D(in_c))
                    stride = 1  # block remains stride=1 thanks to BlurPool

                # Dilation on certain deep MBConv
                dw_dil = 1
                if kind == "mb" and stage_idx >= mb_dilate_from_stage:
                    dw_dil = mb_dilation_value  # e.g., 2

                if kind == "fused":
                    block = FusedMBConv1d(in_c, out_c, k=k, s=stride, expand_ratio=exp, drop_path=dp)
                else:
                    block = MBConv1d(in_c, out_c, k=k, s=stride, expand_ratio=exp,
                                     se_ratio=se_ratio, drop_path=dp, dw_dilation=dw_dil)

                blocks += pre + [block]
                in_c = out_c
                b_idx += 1

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Identity()
        self.feat_dim = in_c

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, N].

        Returns:
            Tuple of (features, None).
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x, None
