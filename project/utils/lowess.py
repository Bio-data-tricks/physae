"""
LOWESS (Locally Weighted Scatterplot Smoothing) implementation in PyTorch.
"""
import torch


def _k_from_length(N: int, frac: float | None, win: int | None) -> int:
    """
    Calculate window size for LOWESS from data length.

    Args:
        N: Number of data points.
        frac: Fraction of points to use (if win is None).
        win: Fixed number of points (takes precedence if provided).

    Returns:
        Window size k (at least 5, at most N).
    """
    if win is not None:
        k = int(win)
    elif frac is not None:
        k = int(max(5, frac * N))
    else:
        k = max(5, int(0.08 * N))  # default = 8% of points
    return max(5, min(k, N))


def _lowess_at_1d(y1d: torch.Tensor, x0: float, k: int, iters: int = 2) -> torch.Tensor:
    """
    Estimate y(x0) via linear LOWESS with robust reweighting (bisquare).

    Args:
        y1d: 1D tensor of values [N].
        x0: Position to estimate (0..N-1).
        k: Number of neighbors (>=5).
        iters: Number of robust reweighting iterations (default: 2).

    Returns:
        Smoothed value at x0.
    """
    assert y1d.ndim == 1
    N = y1d.numel()
    dev = y1d.device
    dtype = torch.float64

    x = torch.arange(N, device=dev, dtype=dtype)  # [0..N-1]
    y = y1d.to(dtype)

    # k nearest neighbors around x0
    dist = (x - x0).abs()
    idx = torch.topk(dist, k, largest=False).indices
    xs = x[idx]
    ys = y[idx]
    dmax = dist[idx].max().clamp_min(1e-12)

    # tri-cube weights on normalized distance
    u = (dist[idx] / dmax).clamp(max=1)
    w = (1 - u.pow(3)).clamp(min=0).pow(3)

    # local linear regression in (xs - x0)
    X = torch.stack([torch.ones_like(xs), (xs - x0)], dim=1)

    def _solve_wls(X, y, w):
        WX = X * w.unsqueeze(1)
        XtWX = X.T @ WX
        XtWy = WX.T @ y
        return torch.linalg.pinv(XtWX) @ XtWy

    beta = _solve_wls(X, ys, w)

    # robust reweighting (bisquare)
    for _ in range(max(0, iters - 1)):
        r = ys - (X @ beta)
        s = torch.median(torch.abs(r)) + 1e-12
        uu = (r / (6 * s)).clamp(min=-1, max=1)
        w_rob = (1 - uu.pow(2)).clamp(min=0).pow(2)
        beta = _solve_wls(X, ys, w * w_rob + 1e-12)

    # smoothed value at x0 = intercept
    return beta[0].to(y1d.dtype)


def lowess_value(
    y: torch.Tensor,
    kind: str = "start",  # "start" | "at" | "max"
    *,
    frac: float | None = 0.08,  # fraction of points (if win=None)
    win: int | None = None,  # fixed number of points (takes precedence if provided)
    x0: float | None = None,  # required if kind=="at"
    n_eval: int = 64,  # number of evaluations for kind=="max"
    iters: int = 2,
    clamp_min_value: float = 1e-6  # useful floor for scales
) -> torch.Tensor:
    """
    Multi-scale LOWESS for signal smoothing.

    Args:
        y: Input tensor [N] or [B, N].
        kind: Type of smoothing:
            - "start": LOWESS value at the beginning (x0=0).
            - "at": LOWESS value at x0 (0..N-1).
            - "max": maximum of LOWESS evaluated on 'n_eval' positions.
        frac: Fraction of points for window (if win=None).
        win: Fixed number of points for window.
        x0: Position for kind="at".
        n_eval: Number of evaluation points for kind="max".
        iters: Number of robust reweighting iterations.
        clamp_min_value: Minimum value to clamp output.

    Returns:
        Smoothed value(s): scalar if y is 1D, [B] if y is 2D.
    """
    if y.ndim == 1:
        y = y.unsqueeze(0)
        squeeze = True
    elif y.ndim == 2:
        squeeze = False
    else:
        raise ValueError("lowess_value expects a 1D or 2D tensor.")

    B, N = y.shape
    k = _k_from_length(N, frac, win)

    def _one(y1d: torch.Tensor) -> torch.Tensor:
        if kind == "start":
            return _lowess_at_1d(y1d, x0=0.0, k=k, iters=iters)
        elif kind == "at":
            if x0 is None:
                raise ValueError("x0 must be provided for kind='at'.")
            return _lowess_at_1d(y1d, float(x0), k=k, iters=iters)
        elif kind == "max":
            xs = torch.linspace(0, max(0, N - 1), n_eval, device=y1d.device, dtype=torch.float32)
            vals = torch.stack([_lowess_at_1d(y1d, float(xx), k=k, iters=iters) for xx in xs])
            return vals.max()
        else:
            raise ValueError("kind must be 'start', 'at' or 'max'.")

    out = torch.stack([_one(y[i]) for i in range(B)]).to(y.dtype)
    out = out.clamp_min(torch.tensor(clamp_min_value, dtype=out.dtype, device=out.device))
    return out[0] if squeeze else out
