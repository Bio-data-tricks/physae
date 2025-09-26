"""Baseline and smoothing utilities used throughout the project."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, device, dtype) -> tuple[torch.Tensor, int]:
    """Return a 1D Gaussian kernel and its radius."""

    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, -1), radius


def scale_first_window(y: torch.Tensor, window: int = 20, sigma: float = 4.0) -> torch.Tensor:
    """Estimate an amplitude scale factor from the first samples of ``y``."""

    if y.ndim == 1:
        y = y.unsqueeze(0)
    kernel, radius = gaussian_kernel1d(sigma, y.device, y.dtype)
    smoothed = F.conv1d(F.pad(y[:, None, :], (radius, radius), mode="reflect"), kernel).squeeze(1)
    m = min(int(window), smoothed.size(1))
    return smoothed[:, :m].amax(dim=1)


def _lowess_start_one(y1d: torch.Tensor, frac: float = 0.08, iters: int = 2) -> torch.Tensor:
    """LOWESS estimate at the beginning of a 1D signal."""

    assert y1d.ndim == 1
    count = y1d.numel()
    window = max(5, int(frac * count))
    if window <= 2:
        return y1d[0].to(torch.float64)

    device = y1d.device
    y = y1d[:window].to(torch.float64)
    x = torch.arange(window, device=device, dtype=torch.float64)
    x = x / x[-1]

    dist = x
    weights = (1 - dist.pow(3)).clamp(min=0).pow(3)
    X = torch.stack([torch.ones_like(x), x], dim=1)

    def solve_wls(Xmat, yvec, wvec):
        WX = Xmat * wvec.unsqueeze(1)
        XtWX = Xmat.T @ WX
        XtWy = WX.T @ yvec
        return torch.linalg.pinv(XtWX) @ XtWy

    beta = solve_wls(X, y, weights)
    for _ in range(max(0, iters - 1)):
        residual = y - (X @ beta)
        scale = torch.median(torch.abs(residual)) + 1e-12
        u = (residual / (6 * scale)).clamp(min=-1, max=1)
        w_robust = (1 - u.pow(2)).clamp(min=0).pow(2)
        beta = solve_wls(X, y, weights * w_robust + 1e-12)

    return beta[0]


def lowess_start_value(y: torch.Tensor, frac: float = 0.08, iters: int = 2) -> torch.Tensor:
    """Return the LOWESS value at the start of a 1D or 2D tensor."""

    if y.ndim == 1:
        return _lowess_start_one(y, frac=frac, iters=iters).to(y.dtype)
    if y.ndim == 2:
        values = [_lowess_start_one(y[i], frac=frac, iters=iters) for i in range(y.shape[0])]
        return torch.stack(values).to(y.dtype)
    raise ValueError("lowess_start_value expects a 1D or 2D tensor")


def _lowess_value_at(y1d: torch.Tensor, x0: float, frac: float = 0.08, iters: int = 2) -> torch.Tensor:
    """Return a LOWESS estimate at a specific abscissa ``x0``."""

    assert y1d.ndim == 1
    count = y1d.numel()
    k = max(5, int(frac * count))
    device = y1d.device
    dtype = torch.float64

    x = torch.arange(count, device=device, dtype=dtype)
    y = y1d.to(dtype)
    dist = (x - x0).abs()
    idx = torch.topk(dist, k, largest=False).indices
    xs, ys = x[idx], y[idx]
    dmax = dist[idx].max().clamp_min(1e-12)

    u = (dist[idx] / dmax).clamp(max=1)
    w = (1 - u.pow(3)).clamp(min=0).pow(3)

    X = torch.stack([torch.ones_like(xs), (xs - x0)], dim=1)

    def solve_wls(Xmat, yvec, wvec):
        WX = Xmat * wvec.unsqueeze(1)
        XtWX = Xmat.T @ WX
        XtWy = WX.T @ yvec
        return torch.linalg.pinv(XtWX) @ XtWy

    beta = solve_wls(X, ys, w)
    for _ in range(max(0, iters - 1)):
        residual = ys - (X @ beta)
        scale = torch.median(torch.abs(residual)) + 1e-12
        u = (residual / (6 * scale)).clamp(min=-1, max=1)
        w_robust = (1 - u.pow(2)).clamp(min=0).pow(2)
        beta = solve_wls(X, ys, w * w_robust + 1e-12)

    return beta[0].to(y1d.dtype)


def lowess_max_value(
    y: torch.Tensor, frac: float = 0.08, iters: int = 2, n_eval: int = 64
) -> torch.Tensor:
    """Estimate the maximum of ``y`` using multiple LOWESS evaluations."""

    def evaluate(y1d: torch.Tensor) -> torch.Tensor:
        count = y1d.numel()
        xs = torch.linspace(0, max(0, count - 1), n_eval, device=y1d.device, dtype=torch.float32)
        values = torch.stack([_lowess_value_at(y1d, float(x0), frac=frac, iters=iters) for x0 in xs])
        return values.max()

    if y.ndim == 1:
        return evaluate(y).clamp_min(torch.tensor(1e-6, dtype=y.dtype, device=y.device)).to(y.dtype)
    if y.ndim == 2:
        out = [evaluate(y[i]) for i in range(y.shape[0])]
        return torch.stack(out).to(y.dtype).clamp_min(torch.tensor(1e-6, dtype=y.dtype, device=y.device))
    raise ValueError("lowess_max_value expects a 1D or 2D tensor")


# Backwards compatibility ------------------------------------------------------
_gauss1d = gaussian_kernel1d
