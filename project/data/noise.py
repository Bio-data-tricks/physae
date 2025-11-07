"""
Noise augmentation for synthetic spectra.
"""
import math
import torch
import torch.nn.functional as F


def add_noise_variety(spectra, *, generator=None, **cfg):
    """
    Add realistic noise to synthetic spectra with configurable profiles.

    Args:
        spectra: Input spectra tensor [B, N].
        generator: Optional random generator for reproducibility.
        **cfg: Configuration parameters:
            - std_add_range: Range for additive noise std (default: (0.001, 0.01))
            - std_mult_range: Range for multiplicative noise std (default: (0.002, 0.02))
            - p_drift: Probability of adding drift (default: 0.7)
            - drift_sigma_range: Range for drift kernel sigma (default: (8.0, 90.0))
            - drift_amp_range: Range for drift amplitude (default: (0.002, 0.03))
            - p_fringes: Probability of adding fringes (default: 0.6)
            - n_fringes_range: Range for number of fringes (default: (1, 3))
            - fringe_freq_range: Range for fringe frequency (default: (0.2, 12.0))
            - fringe_amp_range: Range for fringe amplitude (default: (0.001, 0.01))
            - p_spikes: Probability of adding spikes (default: 0.4)
            - spikes_count_range: Range for spike count (default: (1, 4))
            - spike_amp_range: Range for spike amplitude (default: (0.002, 0.03))
            - spike_width_range: Range for spike width (default: (1.0, 4.0))
            - clip: Clipping range (default: (0.0, 1.3))

    Returns:
        Noisy spectra tensor [B, N].
    """
    std_add_range = cfg.get("std_add_range", (0.001, 0.01))
    std_mult_range = cfg.get("std_mult_range", (0.002, 0.02))
    p_drift = cfg.get("p_drift", 0.7)
    drift_sigma_range = cfg.get("drift_sigma_range", (8.0, 90.0))
    drift_amp_range = cfg.get("drift_amp_range", (0.002, 0.03))
    p_fringes = cfg.get("p_fringes", 0.6)
    n_fringes_range = cfg.get("n_fringes_range", (1, 3))
    fringe_freq_range = cfg.get("fringe_freq_range", (0.2, 12.0))
    fringe_amp_range = cfg.get("fringe_amp_range", (0.001, 0.01))
    p_spikes = cfg.get("p_spikes", 0.4)
    spikes_count_range = cfg.get("spikes_count_range", (1, 4))
    spike_amp_range = cfg.get("spike_amp_range", (0.002, 0.03))
    spike_width_range = cfg.get("spike_width_range", (1.0, 4.0))
    clip = cfg.get("clip", (0.0, 1.3))

    y = spectra
    B, N = y.shape[-2], y.shape[-1]
    dev, dtype = y.device, y.dtype
    g = generator

    def r(a, b):
        return (torch.rand((), device=dev, generator=g) * (b - a) + a).item()

    def ri(a, b):
        return int(torch.randint(a, b + 1, (), device=dev, generator=g).item())

    def rbool(p):
        return bool(torch.rand((), device=dev, generator=g) < p)

    # Additive noise
    std_add = r(*std_add_range)
    std_mult = r(*std_mult_range)
    add = torch.randn(y.shape, device=dev, dtype=dtype, generator=g)
    add = add - add.mean(dim=-1, keepdim=True)
    add_std = add.std(dim=-1, keepdim=True)
    add_std = torch.where(add_std < 1e-8, torch.ones_like(add_std), add_std)
    add = add / add_std * std_add

    # Multiplicative noise
    mult_noise = torch.randn(y.shape, device=dev, dtype=dtype, generator=g)
    mult_noise = mult_noise - mult_noise.mean(dim=-1, keepdim=True)
    mult_std = mult_noise.std(dim=-1, keepdim=True)
    mult_std = torch.where(mult_std < 1e-8, torch.ones_like(mult_std), mult_std)
    mult_noise = mult_noise / mult_std * std_mult
    mult = 1.0 + mult_noise

    out = y * mult + add

    # Drift
    if rbool(p_drift):
        sigma = r(*drift_sigma_range)
        rad = max(1, int(3 * sigma))
        xk = torch.arange(-rad, rad + 1, device=dev, dtype=dtype)
        kernel = torch.exp(-0.5 * (xk / sigma) ** 2)
        kernel = kernel / kernel.sum()
        drift = torch.randn((B, N), device=dev, dtype=dtype, generator=g)
        drift = drift - drift.mean(dim=-1, keepdim=True)
        drift = F.pad(drift.unsqueeze(1), (rad, rad), mode="reflect")
        drift = F.conv1d(drift, kernel.view(1, 1, -1)).squeeze(1)
        drift = drift / (drift.std(dim=-1, keepdim=True) + 1e-8) * r(*drift_amp_range)
        out = out + drift

    # Fringes
    if rbool(p_fringes):
        t = torch.linspace(0, 1, N, device=dev, dtype=dtype)
        fringes = torch.zeros((B, N), device=dev, dtype=dtype)
        for _ in range(ri(*n_fringes_range)):
            f = r(*fringe_freq_range)
            phi = r(0.0, 2 * math.pi)
            amp = r(*fringe_amp_range)
            fringes = fringes + amp * torch.sin(2 * math.pi * f * t + phi)
        # Normalization: remove DC bias and scale
        fringes = fringes - fringes.mean(dim=-1, keepdim=True)
        std = fringes.std(dim=-1, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        fringes = fringes / std
        span = (y.max(dim=-1, keepdim=True).values - y.min(dim=-1, keepdim=True).values).clamp_min(1e-6)
        final_amp = r(*fringe_amp_range)
        fringes = fringes * (final_amp * span)
        out = out + fringes

    # Spikes
    if rbool(p_spikes):
        grid = torch.arange(N, device=dev, dtype=dtype)
        spikes = torch.zeros((B, N), device=dev, dtype=dtype)
        for _ in range(ri(*spikes_count_range)):
            mu = r(0.0, N - 1.0)
            width = r(*spike_width_range)
            amp = r(*spike_amp_range) * (1.0 if rbool(0.5) else -1.0)
            spikes = spikes + amp * torch.exp(-0.5 * ((grid - mu) / width) ** 2)
        out = out + spikes

    if clip is not None:
        out = torch.clamp(out, *clip)
    return out
