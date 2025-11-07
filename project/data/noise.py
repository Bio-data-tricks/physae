"""Noise augmentation helpers for synthetic spectra."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _torch_uniform(a: float, b: float, *, device: torch.device, generator: torch.Generator | None) -> float:
    return (torch.rand((), device=device, generator=generator) * (b - a) + a).item()


def _torch_randint(a: int, b: int, *, device: torch.device, generator: torch.Generator | None) -> int:
    return int(torch.randint(a, b + 1, (), device=device, generator=generator).item())


def _torch_bernoulli(p: float, *, device: torch.device, generator: torch.Generator | None) -> bool:
    return bool(torch.rand((), device=device, generator=generator) < p)


def _line_ref_from_baseline(clean: np.ndarray, baseline_norm: np.ndarray) -> np.ndarray:
    """Estimate line depth reference from baseline-normalised spectra."""

    baseline = np.asarray(baseline_norm, dtype=np.float64).reshape(clean.shape[0], 1)
    depth = np.clip(baseline - clean, a_min=0.0, a_max=None)
    ref = np.percentile(depth, 98, axis=1, keepdims=True)
    return np.clip(ref, 1e-8, None)


def _scale_to_amplitude(noise: np.ndarray, amplitude: np.ndarray, *, center: bool) -> np.ndarray:
    arr = noise
    if center:
        arr = arr - arr.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(arr**2, axis=1, keepdims=True)) + 1e-12
    return noise / rms * amplitude


def _np_rng_from_generator(generator: torch.Generator | None) -> np.random.Generator:
    if generator is not None:
        seed = torch.randint(0, 2**63 - 1, (), generator=generator).item()
    else:
        seed = torch.randint(0, 2**63 - 1, ()).item()
    return np.random.default_rng(int(seed))


def _apply_complex_noise(
    clean: torch.Tensor,
    *,
    generator: torch.Generator | None,
    baseline_norm: torch.Tensor | None,
    noise_type: str,
    noise_level: float,
    max_rel_to_line: float,
    clip: Sequence[float] | None,
) -> torch.Tensor:
    if baseline_norm is None:
        raise ValueError("apply_noise: baseline_norm doit être fourni.")

    clean_cpu = clean.detach().cpu().numpy().astype(np.float64)
    if clean_cpu.ndim == 1:
        clean_cpu = clean_cpu[None, :]
    baseline = baseline_norm.detach().cpu().numpy().astype(np.float64)
    if baseline.ndim == 0:
        baseline = baseline.reshape(1)
    baseline = baseline.reshape(clean_cpu.shape[0], 1)

    rng = _np_rng_from_generator(generator)
    noise = np.zeros_like(clean_cpu, dtype=np.float64)
    N, L = clean_cpu.shape

    line_ref = _line_ref_from_baseline(clean_cpu, baseline)
    amplitude = float(noise_level) * float(max_rel_to_line) * line_ref

    noise_name = noise_type.lower()

    if noise_name == "gaussian":
        noise = rng.standard_normal((N, L))
        noise = _scale_to_amplitude(noise, amplitude, center=True)

    elif noise_name == "shot":
        T = np.clip(clean_cpu, 1e-12, None)
        noise = rng.standard_normal((N, L)) * np.sqrt(T)
        noise = _scale_to_amplitude(noise, amplitude, center=True)

    elif noise_name == "flicker":
        for i in range(N):
            white = rng.standard_normal(L)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(L)
            freqs[0] = 1.0
            pink = np.fft.irfft(fft * (1.0 / np.sqrt(freqs)), n=L)
            pink = pink / (np.std(pink) + 1e-12)
            noise[i] = pink
        noise = _scale_to_amplitude(noise, amplitude, center=True)

    elif noise_name == "etaloning":
        xg = np.linspace(0.0, 1.0, L)
        for i in range(N):
            n_modes = rng.integers(1, 4)
            tmp = np.zeros(L)
            for _ in range(int(n_modes)):
                freq = rng.uniform(3.0, 40.0)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                mode_amp = rng.uniform(0.1, 0.8)
                tmp += mode_amp * np.sin(2.0 * np.pi * freq * xg + phase)
            tmp = tmp / (np.std(tmp) + 1e-12)
            noise[i] = tmp
        noise = _scale_to_amplitude(noise, amplitude, center=True)

    elif noise_name == "glitches":
        x = np.arange(L)
        for i in range(N):
            n_glitches = max(1, int(noise_level * rng.integers(3, 9)))
            tmp = np.zeros(L)
            for _ in range(n_glitches):
                r = rng.random()
                if r < 0.25:
                    pos = rng.integers(int(0.1 * L), int(0.9 * L))
                    rise = rng.uniform(1.5, 5.0)
                    jump = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 8.0)
                    trans = jump / (1.0 + np.exp(-(x - pos) / rise))
                    if rng.random() < 0.6:
                        f = rng.uniform(0.5, 2.0)
                        d = rng.uniform(0.1, 0.3)
                        amp = jump * rng.uniform(0.1, 0.3)
                        mask = (x > pos).astype(float)
                        trans += amp * np.sin(2.0 * np.pi * f * (x - pos)) * np.exp(-d * (x - pos)) * mask
                    tmp += trans
                elif r < 0.45:
                    pos = rng.integers(0, L)
                    width = rng.uniform(2.0, 15.0)
                    amp = rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 6.0)
                    tmp += amp * np.exp(-0.5 * ((x - pos) / width) ** 2)
                elif r < 0.65:
                    pos = rng.integers(int(0.1 * L), int(0.8 * L))
                    tau = rng.uniform(3.0, 20.0)
                    amp = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 7.0)
                    mask = (x >= pos).astype(float)
                    exp_pulse = amp * np.exp(-(x - pos) / tau) * mask
                    rise_time = rng.uniform(0.5, 2.0)
                    rise_mask = (x >= pos) & (x < pos + rise_time)
                    if rise_mask.any():
                        rx = x[rise_mask] - pos
                        exp_pulse[rise_mask] *= rx / rise_time
                    tmp += exp_pulse
                elif r < 0.80:
                    pos = rng.integers(int(0.1 * L), int(0.8 * L))
                    f = rng.uniform(0.3, 1.5)
                    d = rng.uniform(0.05, 0.2)
                    amp = rng.uniform(1.0, 4.0)
                    mask = (x >= pos).astype(float)
                    tmp += amp * np.sin(2.0 * np.pi * f * (x - pos)) * np.exp(-d * (x - pos)) * mask
                elif r < 0.90:
                    n_steps = rng.integers(2, 5)
                    pos_start = rng.integers(int(0.2 * L), int(0.6 * L))
                    step_spacing = rng.integers(5, 20)
                    base_amp = rng.uniform(0.5, 2.0)
                    steps = np.zeros(L)
                    for k in range(int(n_steps)):
                        pos = pos_start + k * step_spacing
                        if pos >= L:
                            break
                        this_amp = rng.choice([-1.0, 1.0]) * base_amp * rng.uniform(0.7, 1.3)
                        rise = rng.uniform(0.5, 2.0)
                        steps += this_amp / (1.0 + np.exp(-(x - pos) / rise))
                    tmp += steps
                else:
                    pos = rng.integers(int(0.2 * L), int(0.7 * L))
                    jump = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 5.0)
                    rise = rng.uniform(1.0, 3.0)
                    base = jump / (1.0 + np.exp(-(x - pos) / rise))
                    if rng.random() < 0.7:
                        f = rng.uniform(1.0, 3.0)
                        amp = jump * rng.uniform(0.2, 0.4)
                        mask = (x > pos).astype(float)
                        base += amp * np.sin(2.0 * np.pi * f * (x - pos)) * np.exp(-0.15 * (x - pos)) * mask
                    if rng.random() < 0.5:
                        p2 = pos + rng.integers(5, 30)
                        if p2 < L:
                            w = rng.uniform(2.0, 8.0)
                            amp2 = jump * rng.uniform(0.3, 0.6)
                            base += amp2 * np.exp(-0.5 * ((x - p2) / w) ** 2)
                    tmp += base
            if rng.random() < noise_level * 0.3:
                fbg = rng.uniform(0.05, 0.2)
                abg = rng.uniform(0.3, 1.0)
                ph = rng.uniform(0.0, 2.0 * np.pi)
                tmp += abg * np.sin(2.0 * np.pi * fbg * x + ph)
            noise[i] = tmp
        noise = _scale_to_amplitude(noise, amplitude, center=False)

    else:
        raise ValueError(f"Unknown complex noise type: {noise_type}")

    noisy = clean_cpu + noise
    if clip is not None:
        lo, hi = clip
        noisy = np.clip(noisy, lo, hi)
    return torch.from_numpy(noisy).to(clean.dtype)


def _apply_legacy_noise(
    spectra: torch.Tensor,
    *,
    generator: torch.Generator | None,
    cfg: dict,
) -> torch.Tensor:
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

    std_add = _torch_uniform(*std_add_range, device=dev, generator=g)
    std_mult = _torch_uniform(*std_mult_range, device=dev, generator=g)
    add = torch.randn(y.shape, device=dev, dtype=dtype, generator=g)
    add = add - add.mean(dim=-1, keepdim=True)
    add_std = add.std(dim=-1, keepdim=True)
    add_std = torch.where(add_std < 1e-8, torch.ones_like(add_std), add_std)
    add = add / add_std * std_add

    mult_noise = torch.randn(y.shape, device=dev, dtype=dtype, generator=g)
    mult_noise = mult_noise - mult_noise.mean(dim=-1, keepdim=True)
    mult_std = mult_noise.std(dim=-1, keepdim=True)
    mult_std = torch.where(mult_std < 1e-8, torch.ones_like(mult_std), mult_std)
    mult_noise = mult_noise / mult_std * std_mult
    mult = 1.0 + mult_noise

    out = y * mult + add

    if _torch_bernoulli(p_drift, device=dev, generator=g):
        sigma = _torch_uniform(*drift_sigma_range, device=dev, generator=g)
        rad = max(1, int(3 * sigma))
        xk = torch.arange(-rad, rad + 1, device=dev, dtype=dtype)
        kernel = torch.exp(-0.5 * (xk / sigma) ** 2)
        kernel = kernel / kernel.sum()
        drift = torch.randn((B, N), device=dev, dtype=dtype, generator=g)
        drift = drift - drift.mean(dim=-1, keepdim=True)
        drift = F.pad(drift.unsqueeze(1), (rad, rad), mode="reflect")
        drift = F.conv1d(drift, kernel.view(1, 1, -1)).squeeze(1)
        drift = drift / (drift.std(dim=-1, keepdim=True) + 1e-8)
        drift = drift * _torch_uniform(*drift_amp_range, device=dev, generator=g)
        out = out + drift

    if _torch_bernoulli(p_fringes, device=dev, generator=g):
        t = torch.linspace(0, 1, N, device=dev, dtype=dtype)
        fringes = torch.zeros((B, N), device=dev, dtype=dtype)
        for _ in range(_torch_randint(*n_fringes_range, device=dev, generator=g)):
            f = _torch_uniform(*fringe_freq_range, device=dev, generator=g)
            phi = _torch_uniform(0.0, 2 * math.pi, device=dev, generator=g)
            amp = _torch_uniform(*fringe_amp_range, device=dev, generator=g)
            fringes = fringes + amp * torch.sin(2 * math.pi * f * t + phi)
        fringes = fringes - fringes.mean(dim=-1, keepdim=True)
        std = fringes.std(dim=-1, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        fringes = fringes / std
        span = (y.max(dim=-1, keepdim=True).values - y.min(dim=-1, keepdim=True).values).clamp_min(1e-6)
        final_amp = _torch_uniform(*fringe_amp_range, device=dev, generator=g)
        fringes = fringes * (final_amp * span)
        out = out + fringes

    if _torch_bernoulli(p_spikes, device=dev, generator=g):
        grid = torch.arange(N, device=dev, dtype=dtype)
        spikes = torch.zeros((B, N), device=dev, dtype=dtype)
        for _ in range(_torch_randint(*spikes_count_range, device=dev, generator=g)):
            mu = _torch_uniform(0.0, N - 1.0, device=dev, generator=g)
            width = _torch_uniform(*spike_width_range, device=dev, generator=g)
            amp = _torch_uniform(*spike_amp_range, device=dev, generator=g)
            amp = amp * (1.0 if _torch_bernoulli(0.5, device=dev, generator=g) else -1.0)
            spikes = spikes + amp * torch.exp(-0.5 * ((grid - mu) / width) ** 2)
        out = out + spikes

    if clip is not None:
        out = torch.clamp(out, *clip)

    return out


def add_noise_variety(
    spectra: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    baseline_norm: torch.Tensor | None = None,
    **cfg,
) -> torch.Tensor:
    """Add configurable noise (legacy and complex profiles) to spectra."""

    cfg_local = dict(cfg)
    complex_cfg = cfg_local.pop("complex", None)

    base_required = bool(cfg_local)
    mode = None
    prob = 1.0
    if complex_cfg is not None:
        mode = str(complex_cfg.get("mode", "replace")).lower()
        prob = float(complex_cfg.get("probability", 1.0))
        if mode not in {"replace", "add", "additive", "blend"}:
            raise ValueError("complex mode must be 'replace', 'add' or 'blend'")
        if not (0.0 <= prob <= 1.0):
            raise ValueError("complex probability must be between 0 and 1")
        if mode != "replace" or prob < 1.0:
            base_required = True

    out = None
    if base_required:
        out = _apply_legacy_noise(spectra, generator=generator, cfg=cfg_local)

    if complex_cfg is None:
        return out if out is not None else spectra.clone()

    dev = spectra.device
    apply_complex = prob >= 1.0 or torch.rand((), device=dev, generator=generator) < prob
    if not apply_complex:
        return out if out is not None else spectra.clone()

    types = complex_cfg.get("noise_types") or complex_cfg.get("types")
    weights = complex_cfg.get("noise_type_weights") or complex_cfg.get("weights")
    if complex_cfg.get("noise_type"):
        noise_type = str(complex_cfg["noise_type"])
    elif types:
        if not isinstance(types, Iterable):
            raise ValueError("complex noise_types must be iterable")
        types_list = [str(t) for t in types]
        if weights is None:
            probs = torch.ones(len(types_list), device=dev)
        else:
            weights_seq = [float(w) for w in weights]
            if len(weights_seq) != len(types_list):
                raise ValueError("noise_type_weights length must match noise_types")
            probs = torch.tensor(weights_seq, device=dev)
        probs = probs / probs.sum().clamp(min=torch.finfo(probs.dtype).eps)
        choice = torch.multinomial(probs, 1, generator=generator).item()
        noise_type = types_list[choice]
    else:
        noise_type = "gaussian"

    if "noise_level" in complex_cfg:
        noise_level = float(complex_cfg["noise_level"])
    else:
        lo, hi = complex_cfg.get("noise_level_range", (0.5, 1.5))
        noise_level = _torch_uniform(float(lo), float(hi), device=dev, generator=generator)

    max_rel = float(complex_cfg.get("max_rel_to_line", 0.10))
    clip = complex_cfg.get("clip")
    clip_seq: Sequence[float] | None = None
    if clip is not None:
        if not isinstance(clip, Sequence) or len(clip) != 2:
            raise ValueError("complex clip must be a sequence of length 2")
        clip_seq = [float(clip[0]), float(clip[1])]

    complex_out = _apply_complex_noise(
        spectra,
        generator=generator,
        baseline_norm=baseline_norm,
        noise_type=noise_type,
        noise_level=noise_level,
        max_rel_to_line=max_rel,
        clip=clip_seq,
    )

    if mode == "replace":
        return complex_out

    if out is None:
        out = _apply_legacy_noise(spectra, generator=generator, cfg={})

    if mode in {"add", "additive"}:
        delta = complex_out - spectra
        result = out + delta
    else:  # blend
        alpha = float(complex_cfg.get("blend_alpha", 0.5))
        alpha = min(max(alpha, 0.0), 1.0)
        result = out * (1.0 - alpha) + complex_out * alpha

    if clip_seq is not None:
        result = torch.clamp(result, *clip_seq)
    return result
