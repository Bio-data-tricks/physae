"""
1D denoiser for spectral residual corrections (length-preserving).
"""
import torch
import torch.nn as nn
from .blocks.convolutions import Norm1d, SiLU


class ResidualBlock1D(nn.Module):
    """Simple 1D residual block with dilated convolutions."""

    def __init__(self, c, k=7, d=1, p=None):
        """
        Args:
            c: Number of channels.
            k: Kernel size.
            d: Dilation.
            p: Padding (auto-calculated if None).
        """
        super().__init__()
        if p is None:
            p = (k - 1) // 2 * d
        self.block = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=k, padding=p, dilation=d, bias=False),
            Norm1d(c),
            SiLU(),
            nn.Conv1d(c, c, kernel_size=1, bias=False),
            Norm1d(c),
        )
        self.act = SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class Denoiser1D(nn.Module):
    """
    Simple denoiser, *length-preserving* (stride=1), output = residual correction.
    """

    def __init__(self, in_ch=1, base_ch=64, depth=6):
        """
        Args:
            in_ch: Input channels.
            base_ch: Base number of channels.
            depth: Number of residual blocks.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3, bias=False),
            Norm1d(base_ch),
            SiLU(),
        )
        blocks = []
        # Dilation scale: 1,2,4,8... moderate
        for i in range(depth):
            d = 2 ** (i % 4)  # cycle 1,2,4,8
            blocks.append(ResidualBlock1D(base_ch, k=7, d=d))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(base_ch, 1, kernel_size=1, bias=True)

    def forward(self, resid):
        """
        Forward pass.

        Args:
            resid: [B, N] residual.

        Returns:
            correction: [B, N] residual correction.
        """
        y = self.stem(resid.unsqueeze(1))
        y = self.blocks(y)
        y = self.head(y).squeeze(1)
        return y


def _design_poly3(n: int, device, dtype):
    """Design cubic polynomial basis for fitting."""
    # columns: [1, x, x^2, x^3] with centered-scaled x for stability
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    xc = x - x.mean()
    X = torch.stack([torch.ones_like(xc), xc, xc ** 2, xc ** 3], dim=1)  # [N,4]
    return X


def baseline_poly3_from_edges(resid: torch.Tensor, left_frac: float = 0.20, right_start: float = 0.75):
    """
    Fit and subtract a degree-3 polynomial baseline from residual edges.

    Args:
        resid: [B, N] residual (recon - target).
        left_frac: Fraction of left edge to use for fitting (default: 0.20).
        right_start: Start fraction of right edge to use for fitting (default: 0.75).

    Returns:
        Tuple of (resid_corr, baseline):
            - resid_corr: [B, N] corrected residual.
            - baseline: [B, N] fitted baseline.
    """
    B, N = resid.shape
    device, dtype = resid.device, resid.dtype
    Xfull = _design_poly3(N, device, resid.dtype)  # [N,4]
    iL = torch.arange(0, max(1, int(N * left_frac)), device=device)
    iR = torch.arange(int(N * right_start), N, device=device)
    idx = torch.cat([iL, iR], dim=0)  # [M]
    X = Xfull[idx]  # [M,4]
    # (X^T X)^{-1} X^T y, batched
    Xt = X.t()  # [4,M]
    XtX = Xt @ X  # [4,4]
    # light regularization for stability
    lam = 1e-6
    XtX = XtX + lam * torch.eye(4, device=device, dtype=dtype)
    XtX_inv = torch.linalg.inv(XtX)  # [4,4]
    P = XtX_inv @ Xt  # [4,M]
    y_edges = resid[:, idx]  # [B,M]
    coeff = (P @ y_edges.T).T  # [B,4]
    baseline = (Xfull @ coeff.transpose(0, 1)).transpose(0, 1)  # [B,N]
    resid_corr = resid - baseline
    return resid_corr, baseline
