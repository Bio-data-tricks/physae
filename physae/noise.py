"""Procedural noise generation used to augment synthetic spectra."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def add_noise_variety(spectra: torch.Tensor, *, generator=None, **config) -> torch.Tensor:
    """Inject a rich set of artefacts into ``spectra``.

    The implementation follows the original version but exposes defaults in a
    dictionary, making it easier to tweak behaviour from the outside.
    """

    std_add_range = config.get("std_add_range", (0.001, 0.01))
    std_mult_range = config.get("std_mult_range", (0.002, 0.02))
    p_drift = config.get("p_drift", 0.7)
    drift_sigma_range = config.get("drift_sigma_range", (8.0, 90.0))
    drift_amp_range = config.get("drift_amp_range", (0.002, 0.03))
    p_fringes = config.get("p_fringes", 0.6)
    n_fringes_range = config.get("n_fringes_range", (1, 3))
    fringe_freq_range = config.get("fringe_freq_range", (0.2, 12.0))
    fringe_amp_range = config.get("fringe_amp_range", (0.001, 0.01))
    p_spikes = config.get("p_spikes", 0.4)
    spikes_count_range = config.get("spikes_count_range", (1, 4))
    spike_amp_range = config.get("spike_amp_range", (0.002, 0.03))
    spike_width_range = config.get("spike_width_range", (1.0, 4.0))
    clip = config.get("clip", (0.0, 1.3))

    y = spectra
    batch, length = y.shape[-2], y.shape[-1]
    device, dtype = y.device, y.dtype
    generator = generator

    def rand_float(a, b):
        return (torch.rand((), device=device, generator=generator) * (b - a) + a).item()

    def rand_int(a, b):
        low = math.ceil(a)
        high = math.floor(b)
        if low > high:
            raise ValueError(
                "Invalid integer range: lower bound is greater than upper bound"
            )
        return int(
            torch.randint(low, high + 1, (), device=device, generator=generator).item()
        )

    def rand_bool(p):
        return bool(torch.rand((), device=device, generator=generator) < p)

    std_add = rand_float(*std_add_range)
    std_mult = rand_float(*std_mult_range)
    additive = torch.randn(y.shape, device=device, dtype=dtype, generator=generator) * std_add
    additive = additive - additive.mean(dim=-1, keepdim=True)
    multiplicative = 1.0 + torch.randn(y.shape, device=device, dtype=dtype, generator=generator) * std_mult
    multiplicative = multiplicative / multiplicative.mean(dim=-1, keepdim=True)
    out = y * multiplicative + additive

    if rand_bool(p_drift):
        sigma = rand_float(*drift_sigma_range)
        radius = max(1, int(3 * sigma))
        grid = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
        kernel = kernel / kernel.sum()
        drift = torch.randn((batch, length), device=device, dtype=dtype, generator=generator)
        drift = drift - drift.mean(dim=-1, keepdim=True)
        drift = F.pad(drift.unsqueeze(1), (radius, radius), mode="reflect")
        drift = F.conv1d(drift, kernel.view(1, 1, -1)).squeeze(1)
        drift = drift / (drift.std(dim=-1, keepdim=True) + 1e-8) * rand_float(*drift_amp_range)
        out = out + drift

    if rand_bool(p_fringes):
        t = torch.linspace(0, 1, length, device=device, dtype=dtype)
        fringes = torch.zeros((batch, length), device=device, dtype=dtype)
        for _ in range(rand_int(*n_fringes_range)):
            freq = rand_float(*fringe_freq_range)
            phase = rand_float(0.0, 2 * math.pi)
            amp = rand_float(*fringe_amp_range)
            fringes = fringes + amp * torch.sin(2 * math.pi * freq * t + phase)
        out = out + fringes

    if rand_bool(p_spikes):
        grid = torch.arange(length, device=device, dtype=dtype)
        spikes = torch.zeros((batch, length), device=device, dtype=dtype)
        for _ in range(rand_int(*spikes_count_range)):
            mu = rand_float(0.0, length - 1.0)
            width = rand_float(*spike_width_range)
            amp = rand_float(*spike_amp_range) * (1.0 if rand_bool(0.5) else -1.0)
            spikes = spikes + amp * torch.exp(-0.5 * ((grid - mu) / width) ** 2)
        out = out + spikes

    if clip is not None:
        out = torch.clamp(out, *clip)
    return out
