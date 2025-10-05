import math, random, sys, os
import shutil
from typing import Optional, List, Dict, Union, Iterable
import os
import csv
import socket
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import Callback
from datetime import datetime
import json, yaml  # pip install pyyaml si besoin
from lion_pytorch import Lion  


# --- Matplotlib: backend non interactif + polices ---
import matplotlib as mpl
mpl.use("Agg")  # important: définir le backend AVANT d'importer pyplot
try:
    mpl.rcParams['font.family'] = ['DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
except Exception:
    pass
import matplotlib.pyplot as plt  # import unique (plusieurs imports causent des soucis HPC)

# === Helpers rank0 + save figure (réutilisés partout) ===
def is_rank0():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def save_fig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 1) Configs globaux & normalisation
# ============================================================
PARAMS = ['sig0', 'dsig', 'mf_CH4', 'baseline0', 'baseline1', 'baseline2', 'P', 'T']
PARAM_TO_IDX = {n:i for i,n in enumerate(PARAMS)}
LOG_SCALE_PARAMS = {'mf_CH4'}
LOG_FLOOR = 1e-7
NORM_PARAMS: Dict[str, tuple] = {}

def norm_param_value(name: str, val: float):
    vmin, vmax = NORM_PARAMS[name]
    if name in LOG_SCALE_PARAMS:
        vmin = max(vmin, LOG_FLOOR); vmax = max(vmax, vmin*(1+1e-12)); val = max(val, LOG_FLOOR)
        lv, lmin, lmax = math.log10(val), math.log10(vmin), math.log10(vmax)
        return (lv - lmin) / (lmax - lmin)
    else:
        return (val - vmin) / (vmax - vmin)

def norm_param_torch(name: str, val_t: torch.Tensor) -> torch.Tensor:
    vmin, vmax = NORM_PARAMS[name]
    vmin_t = torch.as_tensor(vmin, dtype=val_t.dtype, device=val_t.device)
    vmax_t = torch.as_tensor(vmax, dtype=val_t.dtype, device=val_t.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        vmax_t = torch.maximum(vmax_t, vmin_t*(1+torch.finfo(val_t.dtype).eps))
        val_t  = torch.clamp(val_t,  min=LOG_FLOOR)
        lval   = torch.log10(val_t); lmin = torch.log10(vmin_t); lmax = torch.log10(vmax_t)
        return (lval - lmin) / (lmax - lmin)
    else:
        return (val_t - vmin_t) / (vmax_t - vmin_t)

def unnorm_param_torch(name: str, val_norm_t: torch.Tensor) -> torch.Tensor:
    vmin, vmax = NORM_PARAMS[name]
    vmin_t = torch.as_tensor(vmin, dtype=val_norm_t.dtype, device=val_norm_t.device)
    vmax_t = torch.as_tensor(vmax, dtype=val_norm_t.dtype, device=val_norm_t.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        vmax_t = torch.maximum(vmax_t, vmin_t*(1+torch.finfo(val_norm_t.dtype).eps))
        lmin   = torch.log10(vmin_t); lmax = torch.log10(vmax_t)
        lval   = val_norm_t*(lmax-lmin)+lmin
        return torch.pow(10.0, lval)
    else:
        return val_norm_t*(vmax_t-vmin_t)+vmin_t

import math, torch, torch.nn.functional as F

# ---------- LOWESS unifié ----------
import math, torch

def _k_from_length(N: int, frac: float | None, win: int | None) -> int:
    if win is not None:
        k = int(win)
    elif frac is not None:
        k = int(max(5, frac * N))
    else:
        k = max(5, int(0.08 * N))   # défaut = 8% des points
    return max(5, min(k, N))

def _lowess_at_1d(y1d: torch.Tensor, x0: float, k: int, iters: int = 2) -> torch.Tensor:
    """
    Estime y(x0) via LOWESS linéaire + repondération robuste (bisquare).
    - y1d: [N] (float tensor)
    - x0 : position réelle (0..N-1)
    - k  : nb de voisins (>=5)
    """
    assert y1d.ndim == 1
    N = y1d.numel()
    dev = y1d.device
    dtype = torch.float64

    x = torch.arange(N, device=dev, dtype=dtype)    # [0..N-1]
    y = y1d.to(dtype)

    # k plus proches voisins autour de x0
    dist = (x - x0).abs()
    idx  = torch.topk(dist, k, largest=False).indices
    xs   = x[idx]; ys = y[idx]
    dmax = dist[idx].max().clamp_min(1e-12)

    # poids tri-cube sur distance normalisée
    u = (dist[idx] / dmax).clamp(max=1)
    w = (1 - u.pow(3)).clamp(min=0).pow(3)

    # régression linéaire locale en (xs - x0)
    X = torch.stack([torch.ones_like(xs), (xs - x0)], dim=1)

    def _solve_wls(X, y, w):
        WX   = X * w.unsqueeze(1)
        XtWX = X.T @ WX
        XtWy = WX.T @ y
        return torch.linalg.pinv(XtWX) @ XtWy

    beta = _solve_wls(X, ys, w)

    # repondération robuste (bisquare)
    for _ in range(max(0, iters - 1)):
        r = ys - (X @ beta)
        s = torch.median(torch.abs(r)) + 1e-12
        uu = (r / (6*s)).clamp(min=-1, max=1)
        w_rob = (1 - uu.pow(2)).clamp(min=0).pow(2)
        beta = _solve_wls(X, ys, w * w_rob + 1e-12)

    # valeur lissée en x0 = intercept
    return beta[0].to(y1d.dtype)

def lowess_value(
    y: torch.Tensor,
    kind: str = "start",            # "start" | "at" | "max"
    *,
    frac: float | None = 0.08,      # fraction de points (si win=None)
    win: int | None = None,         # nb de points fixes (prend le dessus si fourni)
    x0: float | None = None,        # requis si kind=="at"
    n_eval: int = 64,               # nb d’évaluations pour kind=="max"
    iters: int = 2,
    clamp_min_value: float = 1e-6   # plancher utile pour des échelles
) -> torch.Tensor:
    """
    - kind="start" : valeur LOWESS en tout début (x0=0).
    - kind="at"    : valeur LOWESS en x0 (0..N-1).
    - kind="max"   : max de LOWESS évalué sur 'n_eval' positions.
    y : [N] ou [B,N]
    Retour:
      - start/at : scalaire (1D) si y est 1D, ou [B] si y est 2D
      - max      : idem
    """
    if y.ndim == 1:
        y = y.unsqueeze(0)
        squeeze = True
    elif y.ndim == 2:
        squeeze = False
    else:
        raise ValueError("lowess_value attend un tenseur 1D ou 2D.")

    B, N = y.shape
    k = _k_from_length(N, frac, win)

    def _one(y1d: torch.Tensor) -> torch.Tensor:
        if kind == "start":
            return _lowess_at_1d(y1d, x0=0.0, k=k, iters=iters)
        elif kind == "at":
            if x0 is None:
                raise ValueError("x0 doit être fourni pour kind='at'.")
            return _lowess_at_1d(y1d, float(x0), k=k, iters=iters)
        elif kind == "max":
            xs = torch.linspace(0, max(0, N-1), n_eval, device=y1d.device, dtype=torch.float32)
            vals = torch.stack([_lowess_at_1d(y1d, float(xx), k=k, iters=iters) for xx in xs])
            return vals.max()
        else:
            raise ValueError("kind doit être 'start', 'at' ou 'max'.")

    out = torch.stack([_one(y[i]) for i in range(B)]).to(y.dtype)
    out = out.clamp_min(torch.tensor(clamp_min_value, dtype=out.dtype, device=out.device))
    return out[0] if squeeze else out


# ============================================================
# 2) Moteur physique & parsing
# ============================================================
def parse_csv_transitions(csv_str):
    transitions = []
    for line in csv_str.strip().splitlines():
        if not line.strip() or line.strip().startswith("#"): continue
        toks = [t.strip() for t in line.split(";")]
        while len(toks) < 14: toks.append('0')
        transitions.append({
            'mid': int(toks[0]), 'lid': int(float(toks[1])), 'center': float(toks[2]),
            'amplitude': float(toks[3]), 'gamma_air': float(toks[4]), 'gamma_self': float(toks[5]),
            'e0': float(toks[6]), 'n_air': float(toks[7]), 'shift_air': float(toks[8]),
            'abundance': float(toks[9]), 'gDicke': float(toks[10]), 'nDicke': float(toks[11]),
            'lmf': float(toks[12]), 'nlmf': float(toks[13]),
        })
    return transitions

def transitions_to_tensors(transitions, device):
    keys = ['amplitude', 'center', 'gamma_air', 'gamma_self', 'n_air',
            'shift_air', 'gDicke', 'nDicke', 'lmf', 'nlmf']
    return [torch.tensor([t[k] for t in transitions], dtype=torch.float32, device=device) for k in keys]

MOLECULE_PARAMS = {'CH4': {'M': 16.04, 'PL': 15.12}, 'H2O': {'M': 18.02, 'PL': 15.12}}

_b = torch.tensor([-0.0173-0.0463j, -0.7399+0.8395j, 5.8406+0.9536j, -5.5834-11.2086j], dtype=torch.cdouble)
_b = torch.cat((_b, _b.conj()))
_c = torch.tensor([2.2377-1.626j, 1.4652-1.7896j, 0.8393-1.892j, 0.2739-1.9418j], dtype=torch.cdouble)
_c = torch.cat((_c, -_c.conj()))

def wofz_torch(z: torch.Tensor) -> torch.Tensor:
    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
    _b_local = _b.to(device=z.device, dtype=z.dtype)
    _c_local = _c.to(device=z.device, dtype=z.dtype)
    w = (_b_local / (z.unsqueeze(-1) - _c_local)).sum(dim=-1)
    w = w * (1j * inv_sqrt_pi)
    mask = (z.imag < 0)
    w_ref_all = torch.exp(-(z**2)) * 2.0 - w.conj()
    w = torch.where(mask, w_ref_all, w)
    return w

def pine_profile_torch_complex(x, sigma_hwhm, gamma, gDicke, *, device='cpu'):
    sigma = sigma_hwhm / math.sqrt(2*math.log(2.))
    xh = math.sqrt(math.log(2.))*x/sigma_hwhm
    yh = math.sqrt(math.log(2.))*gamma/sigma_hwhm
    zD = math.sqrt(math.log(2.))*gDicke/sigma_hwhm
    z = xh + 1j*(yh + zD)
    k = -wofz_torch(z)
    k_r, k_i = k.real, k.imag
    denom = (1 - zD*math.sqrt(math.pi)*k_r)**2 + (zD*math.sqrt(math.pi)*k_i)**2
    real = (k_r - zD*math.sqrt(math.pi)*(k_r**2 + k_i**2)) / denom
    imag = k_i / denom
    factor = math.sqrt(math.log(2.) / math.pi) / sigma
    return real * factor, imag * factor

def apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T, TREF=296.): 
    flm = lmf * ((T/TREF) ** nlmf)
    return real_prof + imag_prof * flm

def polyval_torch(coeffs, x):
    powers = torch.arange(coeffs.shape[1], device=coeffs.device, dtype=coeffs.dtype)
    return torch.sum(coeffs.unsqueeze(2) * x.unsqueeze(0).pow(powers.view(1, -1, 1)), dim=1)

def batch_physics_forward_multimol_vgrid(
    sig0, dsig, poly_freq, v_grid_idx, baseline_coeffs,
    transitions_dict, P, T, mf_dict, device='cpu'
):
    B, N = sig0.shape[0], v_grid_idx.shape[0]
    v_grid_idx = v_grid_idx.to(device=device, dtype=torch.float64)
    sig0, dsig, P, T = (
        sig0.to(dtype=torch.float64).unsqueeze(1),
        dsig.to(dtype=torch.float64).unsqueeze(1),
        P.to(dtype=torch.float64).unsqueeze(1),
        T.to(dtype=torch.float64).unsqueeze(1)
    )
    if baseline_coeffs.dim() == 1:
        baseline_coeffs = baseline_coeffs.unsqueeze(0)
    baseline_coeffs = baseline_coeffs.to(dtype=torch.float64)

    poly_freq_torch = torch.tensor(poly_freq, dtype=torch.float64, device=device).unsqueeze(0).expand(B, -1)
    coeffs = torch.cat([sig0, dsig, poly_freq_torch], dim=1)
    v_grid_batch = polyval_torch(coeffs, v_grid_idx)

    total_profile = torch.zeros((B, N), device=device, dtype=torch.float64)

    C  = torch.tensor(2.99792458e10, dtype=torch.float64, device=device)
    NA = torch.tensor(6.02214129e23, dtype=torch.float64, device=device)
    KB = torch.tensor(1.380649e-16, dtype=torch.float64, device=device)
    P0 = torch.tensor(1013.25, dtype=torch.float64, device=device)
    T0 = torch.tensor(273.15, dtype=torch.float64, device=device)
    L0 = torch.tensor(2.6867773e19, dtype=torch.float64, device=device)
    TREF = torch.tensor(296.0, dtype=torch.float64, device=device)

    for mol, trans in transitions_dict.items():
        tensors = transitions_to_tensors(trans, device)
        amp, center, ga, gs, na, sa, gd, nd, lmf, nlmf = [t.to(dtype=torch.float64).view(1, -1, 1) for t in tensors]
        mf   = mf_dict[mol].to(dtype=torch.float64).view(B, 1, 1)
        Mmol = torch.tensor(MOLECULE_PARAMS[mol]['M'], dtype=torch.float64, device=device)
        PL   = torch.tensor(MOLECULE_PARAMS[mol]['PL'], dtype=torch.float64, device=device)
        T_exp, P_exp, v_grid_exp = T.view(B, 1, 1), P.view(B, 1, 1), v_grid_batch.view(B, 1, N)
        x = v_grid_exp - center
        sigma_HWHM = (center / C) * torch.sqrt(2 * NA * KB * T_exp * math.log(2.) / Mmol)
        gamma = P_exp / P0 * (TREF / T_exp) ** na * (ga * (1 - mf) + gs * mf)
        real_prof, imag_prof = pine_profile_torch_complex(x, sigma_HWHM, gamma, gd, device=device)
        profile = apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T_exp)
        band = profile * amp * PL * 100 * mf * L0 * P_exp / P0 * T0 / T_exp
        total_profile += band.sum(dim=1)

    transmission = torch.exp(-total_profile)

    # Baseline poly (monômes 1, x, x^2 sur l'index — cohérent avec les ranges)
    x_bl = torch.arange(N, device=device, dtype=torch.float64)
    powers_bl = torch.arange(baseline_coeffs.shape[1], device=device, dtype=torch.float64)
    baseline = torch.sum(baseline_coeffs.unsqueeze(2) * x_bl.unsqueeze(0).pow(powers_bl.view(1, -1, 1)), dim=1)

    return transmission * baseline, v_grid_batch

# ============================================================
# 3) Dataset & bruits
# ============================================================
def add_noise_variety(spectra, *, generator=None, **cfg):
    std_add_range     = cfg.get("std_add_range", (0.001, 0.01))
    std_mult_range    = cfg.get("std_mult_range", (0.002, 0.02))
    p_drift           = cfg.get("p_drift", 0.7)
    drift_sigma_range = cfg.get("drift_sigma_range", (8.0, 90.0))
    drift_amp_range   = cfg.get("drift_amp_range", (0.002, 0.03))
    p_fringes         = cfg.get("p_fringes", 0.6)
    n_fringes_range   = cfg.get("n_fringes_range", (1, 3))
    fringe_freq_range = cfg.get("fringe_freq_range", (0.2, 12.0))
    fringe_amp_range  = cfg.get("fringe_amp_range", (0.001, 0.01))
    p_spikes          = cfg.get("p_spikes", 0.4)
    spikes_count_range= cfg.get("spikes_count_range", (1, 4))
    spike_amp_range   = cfg.get("spike_amp_range", (0.002, 0.03))
    spike_width_range = cfg.get("spike_width_range", (1.0, 4.0))
    clip              = cfg.get("clip", (0.0, 1.3))

    y = spectra
    B, N = y.shape[-2], y.shape[-1]
    dev, dtype = y.device, y.dtype
    g = generator

    def r(a, b):  return (torch.rand((), device=dev, generator=g) * (b - a) + a).item()
    def ri(a, b): return int(torch.randint(a, b + 1, (), device=dev, generator=g).item())
    def rbool(p): return bool(torch.rand((), device=dev, generator=g) < p)

    std_add  = r(*std_add_range)
    std_mult = r(*std_mult_range)
    add  = torch.randn(y.shape, device=dev, dtype=dtype, generator=g) * std_add
    add  = add - add.mean(dim=-1, keepdim=True)
    mult = 1.0 + torch.randn(y.shape, device=dev, dtype=dtype, generator=g) * std_mult
    mult = mult / mult.mean(dim=-1, keepdim=True)
    out = y * mult + add

    if rbool(p_drift):
        sigma = r(*drift_sigma_range)
        rad = max(1, int(3 * sigma))
        xk = torch.arange(-rad, rad + 1, device=dev, dtype=dtype)
        kernel = torch.exp(-0.5 * (xk / sigma) ** 2); kernel = kernel / kernel.sum()
        drift = torch.randn((B, N), device=dev, dtype=dtype, generator=g)
        drift = drift - drift.mean(dim=-1, keepdim=True)
        drift = F.pad(drift.unsqueeze(1), (rad, rad), mode="reflect")
        drift = F.conv1d(drift, kernel.view(1, 1, -1)).squeeze(1)
        drift = drift / (drift.std(dim=-1, keepdim=True) + 1e-8) * r(*drift_amp_range)
        out = out + drift

    if rbool(p_fringes):
        t = torch.linspace(0, 1, N, device=dev, dtype=dtype)
        fringes = torch.zeros((B, N), device=dev, dtype=dtype)
        for _ in range(ri(*n_fringes_range)):
            f, phi, amp = r(*fringe_freq_range), r(0.0, 2 * math.pi), r(*fringe_amp_range)
            fringes = fringes + amp * torch.sin(2 * math.pi * f * t + phi)
        out = out + fringes

    if rbool(p_spikes):
        grid = torch.arange(N, device=dev, dtype=dtype)
        spikes = torch.zeros((B, N), device=dev, dtype=dtype)
        for _ in range(ri(*spikes_count_range)):
            mu, width = r(0.0, N - 1.0), r(*spike_width_range)
            amp = r(*spike_amp_range) * (1.0 if rbool(0.5) else -1.0)
            spikes = spikes + amp * torch.exp(-0.5 * ((grid - mu) / width) ** 2)
        out = out + spikes

    if clip is not None:
        out = torch.clamp(out, *clip)
    return out

class SpectraDataset(Dataset):
    def __init__(self, n_samples, num_points, poly_freq_CH4, transitions_dict,
                 sample_ranges: Optional[dict] = None, strict_check: bool = True,
                 with_noise: bool = True, noise_profile: Optional[dict] = None,
                 freeze_noise: bool = False):
        self.n_samples, self.num_points = n_samples, num_points
        self.poly_freq_CH4, self.transitions_dict = poly_freq_CH4, transitions_dict
        self.sample_ranges = sample_ranges if sample_ranges is not None else NORM_PARAMS
        self.with_noise = bool(with_noise)
        self.noise_profile = dict(noise_profile or {})
        self.freeze_noise = bool(freeze_noise)
        self.epoch = 0
        if strict_check:
            for k in PARAMS:
                smin, smax = self.sample_ranges[k]
                nmin, nmax = NORM_PARAMS[k]
                if smin < nmin or smax > nmax:
                    raise ValueError(
                        f"sample_ranges['{k}']={self.sample_ranges[k]} sort de NORM_PARAMS[{k}]={NORM_PARAMS[k]}."
                    )

    def set_epoch(self, e: int):  self.epoch = int(e)

    def _make_generator(self, idx: int) -> torch.Generator:
        base = torch.initial_seed()
        g = torch.Generator(device='cpu')
        if self.freeze_noise:
            seed = (123456789 + 97 * idx) % (2**63 - 1)
        else:
            seed = (base + 1_000_003 * self.epoch + 97 * idx) % (2**63 - 1)
        g.manual_seed(seed)
        return g

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        device, dtype = 'cpu', torch.float32
        vals = [torch.empty(1, dtype=dtype).uniform_(*self.sample_ranges[k]) for k in PARAMS]
        sig0, dsig, mf_CH4, b0, b1, b2, P, T = vals
        baseline_coeffs = torch.cat([b0, b1, b2]).unsqueeze(0)
        mf_dict = {'CH4': mf_CH4}
        v_grid_idx = torch.arange(self.num_points, dtype=dtype, device=device)

        params_norm = torch.tensor([norm_param_value(k, v.item()) for k, v in zip(PARAMS, vals)],
                                   dtype=torch.float32)

        spectra_clean, _ = batch_physics_forward_multimol_vgrid(
            sig0, dsig, self.poly_freq_CH4, v_grid_idx,
            baseline_coeffs, self.transitions_dict, P, T, mf_dict, device=device
        )
        spectra_clean = spectra_clean.to(torch.float32)

        if self.with_noise:
            g = self._make_generator(idx)
            spectra_noisy = add_noise_variety(spectra_clean, generator=g, **self.noise_profile)
        else:
            spectra_noisy = spectra_clean

        # --- ÉCHELLE POUR L'ENTRÉE (noisy) ---
        #scale_noisy = scale_first_window(spectra_noisy).unsqueeze(1).clamp_min(1e-8)
        scale_noisy = lowess_value(spectra_noisy, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)

        noisy_spectra = spectra_noisy / scale_noisy   # normalisé pour l’encodeur

        # --- MAX SIMPLE SUR LE CLEAN ---
        max_clean = lowess_value(spectra_clean, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)
        clean_spectra = spectra_clean / max_clean     # max(clean)=1

        return {
            'noisy_spectra': noisy_spectra[0].detach(),
            'clean_spectra': clean_spectra[0].detach(),
            'params': params_norm,
            'scale': scale_noisy.squeeze(1).to(torch.float32)[0]
        }

# ============================================================
# EfficientNetV2-1D encoder (drop-in replacement)
# ============================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- utils ---
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # prevent going down by >10%
        new_v += divisor
    return new_v

class DropPath(nn.Module):
    """Stochastic Depth (per sample)"""
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

class SiLU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

def Norm1d(c): 
    groups = 8 if c >= 8 else 1
    return nn.GroupNorm(groups, c)

class ConvBNAct1d(nn.Sequential):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, bias=False, act=True):
        if p is None: p = k // 2
        mods = [nn.Conv1d(in_c, out_c, k, s, p, groups=g, bias=bias), Norm1d(out_c)]
        if act: mods.append(SiLU())
        super().__init__(*mods)

# --- Squeeze & Excitation (optionnel dans MBConv) ---
class SE1d(nn.Module):
    def __init__(self, c, se_ratio=0.25):
        super().__init__()
        hidden = max(1, int(c * se_ratio))
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(c, hidden, 1)
        self.act = SiLU()
        self.fc2 = nn.Conv1d(hidden, c, 1)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        s = self.avg(x)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)

# --- Fused-MBConv (EffNetV2) : conv(k,s) + pw expansion fusionnés ---
class FusedMBConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, s, expand_ratio, drop_path=0.0):
        super().__init__()
        mid = int(in_c * expand_ratio)
        self.use_res = (s == 1 and in_c == out_c)
        if expand_ratio != 1:
            self.fused = nn.Sequential(
                ConvBNAct1d(in_c, mid, k=k, s=s, p=k//2, g=1, act=True),
                ConvBNAct1d(mid, out_c, k=1, s=1, act=False),
            )
        else:
            self.fused = nn.Sequential(
                ConvBNAct1d(in_c, out_c, k=k, s=s, p=k//2, g=1, act=True),
            )
        self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        y = self.fused(x)
        if self.use_res:
            y = self.drop(y) + x
        return y

# --- MBConv (dw + se + pw) ---
class MBConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, s, expand_ratio, se_ratio=0.25, drop_path=0.0):
        super().__init__()
        mid = int(in_c * expand_ratio)
        self.use_res = (s == 1 and in_c == out_c)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct1d(in_c, mid, k=1, s=1, act=True))

        # depthwise
        layers.append(ConvBNAct1d(mid, mid, k=k, s=s, p=k//2, g=mid, act=True))
        if se_ratio is not None and se_ratio > 0:
            layers.append(SE1d(mid, se_ratio=se_ratio))

        # project
        layers.append(ConvBNAct1d(mid, out_c, k=1, s=1, act=False))
        self.block = nn.Sequential(*layers)
        self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        y = self.block(x)
        if self.use_res:
            y = self.drop(y) + x
        return y

# --- config EffNetV2 (S/M/L approximés pour 1D) ---
_EFFV2_CFGS = {
    # (block, repeats, out_c, kernel, stride, expand)
    # Fused stages then MBConv stages (comme V2 2D, adaptés au 1D)
    "s": [
        ("fused", 2,  24, 3, 1, 1.0),
        ("fused", 4,  48, 3, 2, 4.0),
        ("fused", 4,  64, 3, 2, 4.0),
        ("mb",    4, 128, 3, 2, 4.0),
        ("mb",    6, 160, 3, 1, 6.0),
        ("mb",    8, 256, 3, 2, 6.0),
    ],
    "m": [
        ("fused", 3,  24, 3, 1, 1.0),
        ("fused", 5,  48, 3, 2, 4.0),
        ("fused", 5,  80, 3, 2, 4.0),
        ("mb",    5, 160, 3, 2, 6.0),
        ("mb",    7, 176, 3, 1, 6.0),
        ("mb",   10, 304, 3, 2, 6.0),
    ],
    "l": [
        ("fused", 4,  32, 3, 1, 1.0),
        ("fused", 7,  64, 3, 2, 4.0),
        ("fused", 7,  96, 3, 2, 4.0),
        ("mb",    7, 192, 3, 2, 6.0),
        ("mb",   10, 224, 3, 1, 6.0),
        ("mb",   14, 384, 3, 2, 6.0),
    ],
}

class EfficientNetEncoder(nn.Module):
    """
    Drop-in encoder pour ton pipeline:
      - même interface que EfficientNetEncoder
      - retourne (features_1d, None)
      - attribut self.feat_dim pour les têtes
    """
    def __init__(
        self,
        in_channels=1,
        variant: str = "s",
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.1,
        stem_channels: int | None = None,
    ):
        super().__init__()
        cfg = _EFFV2_CFGS[variant]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels or 32, kernel_size=3, stride=2, padding=1, bias=False),
            Norm1d(stem_channels or 32),
            SiLU()
        )
        in_c = stem_channels or 32

        blocks = []
        total_blocks = sum(int(math.ceil(r * depth_mult)) for (_, r, *_ ) in cfg)
        b_idx = 0
        for (kind, repeats, out_c, k, s, exp) in cfg:
            out_c = _make_divisible(out_c * width_mult, 8)
            repeats = int(math.ceil(repeats * depth_mult))
            for i in range(repeats):
                stride = s if i == 0 else 1
                dp = drop_path_rate * b_idx / max(1, total_blocks - 1)
                if kind == "fused":
                    block = FusedMBConv1d(in_c, out_c, k=k, s=stride, expand_ratio=exp, drop_path=dp)
                else:
                    block = MBConv1d(in_c, out_c, k=k, s=stride, expand_ratio=exp,
                                     se_ratio=se_ratio, drop_path=dp)
                blocks.append(block)
                in_c = out_c
                b_idx += 1
        self.blocks = nn.Sequential(*blocks)

        # head (optionnelle) : légère proj pour offrir une dimension "propre"
        self.head = nn.Identity()  # garde la résolution temporelle courante
        self.feat_dim = in_c

    def forward(self, x):  # x: [B, C, N]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x, None


# ============================================================
# 5) ReLoBRaLo
# ============================================================
import torch

class ReLoBRaLoLoss:
    """
    ReLoBRaLo avec tirage Torch déterministe (DDP friendly).

    Args
    ----
    loss_names : list[str]
        Noms (ordre) des composantes de pertes.
    alpha : float
        Lissage EMA des poids (0→réactif, 1→inertiel).
    tau : float
        Température pour le softmax.
    history_len : int
        Longueur max d'historique par perte.
    seed : int | None
        Graine du générateur Torch. Si None, on laisse l'état global.
    """

    def __init__(self, loss_names, alpha=0.9, tau=1.0, history_len=10, seed=12345):
        self.loss_names = list(loss_names)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.history_len = int(history_len)

        # Historique sous forme de listes (simple & rapide)
        self.loss_history = {name: [] for name in self.loss_names}

        # Poids init = 1
        self.weights = torch.ones(len(self.loss_names), dtype=torch.float32)

        # Générateur Torch pour tirages stochastiques (DDP-safe)
        self._g = torch.Generator(device="cpu")
        if seed is not None:
            self._g.manual_seed(int(seed))

    def set_seed(self, seed: int):
        """Permet de (re)fixer la graine si besoin (ex: au début d'un run)."""
        self._g.manual_seed(int(seed))

    def to(self, device=None, dtype=None):
        """Optionnel : déplacer/convertir les tenseurs internes."""
        if device is not None or dtype is not None:
            self.weights = self.weights.to(device=device or self.weights.device,
                                           dtype=dtype or self.weights.dtype)
        return self

    def _append_history(self, current_losses):
        # current_losses: Tensor [K] (mêmes composantes que loss_names)
        for i, name in enumerate(self.loss_names):
            self.loss_history[name].append(float(current_losses[i].detach().cpu()))
            if len(self.loss_history[name]) > self.history_len:
                self.loss_history[name].pop(0)

    @torch.no_grad()
    def compute_weights(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Met à jour et retourne les poids (Tensor [K]) pour agréger les pertes.
        """
        device = current_losses.device
        dtype  = current_losses.dtype

        # 1) Alimente l'historique
        self._append_history(current_losses)

        # 2) Pas assez d'historique -> retourner les poids courants (move to device)
        if len(self.loss_history[self.loss_names[0]]) < 2:
            return self.weights.to(device=device, dtype=dtype)

        # 3) Ratios relatifs: L_t / L_{j} avec j tiré uniformément dans l'historique
        ratios = []
        for name in self.loss_names:
            hist = self.loss_history[name]  # liste python
            # j ∈ [0, len(hist)-2] pour éviter l'élément courant (le -1)
            j = int(torch.randint(low=0, high=len(hist)-1, size=(), generator=self._g).item())
            num = float(hist[-1])
            den = float(hist[j]) + 1e-8
            ratios.append(num / den)

        ratios_t = torch.tensor(ratios, device=device, dtype=dtype)

        # 4) Balancing (même logique que ta version)
        mean_rel = ratios_t.mean()
        balancing = mean_rel / (ratios_t + 1e-8)

        # Softmax tempéré puis mise à l’échelle par K (nombre de pertes)
        K = len(self.loss_names)
        new_w = K * torch.softmax(balancing / self.tau, dim=0)

        # 5) EMA des poids pour éviter les oscillations
        w_old = self.weights.to(device=device, dtype=dtype)
        w_new = self.alpha * w_old + (1.0 - self.alpha) * new_w

        # 6) Stocke et renvoie
        self.weights = w_new.detach().cpu()  # stockage neutre (CPU)
        return w_new

# ============================================================
# 6) Raffineur type EfficientNet (identique au backbone)
# ============================================================
# ============================================================
# 6) Raffineur type EfficientNet (paramétrable)
# ============================================================
class EfficientNetRefiner(nn.Module):
    def __init__(
        self,
        m_params: int,
        cond_dim: int,
        backbone_feat_dim: int,
        *,
        # — contrôles existants —
        delta_scale: float = 0.1,
        max_refine_steps: int = 3,
        encoder_variant: str = "s",
        # — NOUVEAUX hyperparamètres exposés —
        encoder_width_mult: float = 1.0,
        encoder_depth_mult: float = 1.0,
        encoder_stem_channels: int | None = None,
        encoder_drop_path: float = 0.1,
        encoder_se_ratio: float = 0.25,
        feature_pool: str = "avg",           # {"avg","max","avgmax"}
        shared_hidden_scale: float = 0.5,    # taille du MLP partagé: H = max(64, D * scale)
        time_embed_dim: int | None = None,   # si None → max(16, H//4)
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.m_params = int(m_params)
        self.cond_dim = int(cond_dim)
        self.use_film = True

        # ----- encodeur 1D (2 canaux: noisy + resid) -----
        self.encoder = EfficientNetEncoder(
            in_channels=2,
            variant=encoder_variant,
            width_mult=encoder_width_mult,
            depth_mult=encoder_depth_mult,
            se_ratio=encoder_se_ratio,
            drop_path_rate=encoder_drop_path,
            stem_channels=encoder_stem_channels,
        )

        # pooling des features 1D
        D = self.encoder.feat_dim
        pool = feature_pool.lower()
        if pool == "avg":
            self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        elif pool == "max":
            self.feature_head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        elif pool == "avgmax":
            self.feature_head = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
        else:
            raise ValueError(f"feature_pool inconnu: {feature_pool}")

        # dimension du MLP partagé
        H = max(64, int(round(D * float(shared_hidden_scale))))

        # Si avgmax, on double l'entrée du MLP partagé
        in_shared = D if pool != "avgmax" else 2 * D

        self.shared_head = nn.Sequential(
            nn.Linear(in_shared, H), nn.LayerNorm(H), nn.GELU(),
            nn.Linear(H, H), nn.LayerNorm(H), nn.GELU(),
        )

        # ----- time-step embedding (+ FiLM temps) -----
        self.time_embed_dim = int(time_embed_dim) if time_embed_dim is not None else max(16, H // 4)
        self.time_embed = nn.Embedding(max_refine_steps, self.time_embed_dim)

        film_in = self.cond_dim + self.time_embed_dim
        self.film_time = nn.Sequential(
            nn.Linear(film_in, H), nn.Tanh(),
            nn.Linear(H, 2 * H)  # -> gamma, beta
        )

        # portes d’amplitude par paramètre
        self.scale_gate = nn.Linear(H, m_params)

        # tête delta (utilise aussi les features du backbone principal + params normés)
        self.delta_head = nn.Sequential(
            nn.Linear(H + backbone_feat_dim + m_params, H),
            nn.LayerNorm(H), nn.GELU(),
            nn.Linear(H, m_params)
        )

        # mémorise pour forward
        self._feature_pool_mode = pool
        self._H = H

    def _pool_features(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, C, T]
        if self._feature_pool_mode == "avg":
            return self.feature_head(latent)
        if self._feature_pool_mode == "max":
            return self.feature_head(latent)
        # avgmax: concat(AvgPool, MaxPool)
        avgp, maxp = self.feature_head
        a = avgp(latent).flatten(1)
        m = maxp(latent).flatten(1)
        return torch.cat([a, m], dim=1)

    def forward(self, noisy, resid, params_pred_norm, cond_norm, feat_shared, t_step: int = 0):
        x = torch.stack([noisy, resid], dim=1)  # [B, 2, N]
        latent, _ = self.encoder(x)
        feat = self._pool_features(latent)
        h = self.shared_head(feat)

        # ----- Time-step embedding + (optionnel) condition -----
        B = h.size(0)
        t_ids = torch.full((B,), int(t_step), device=h.device, dtype=torch.long)
        tvec = self.time_embed(t_ids)  # [B, time_embed_dim]
        cond_in = tvec if (cond_norm is None) else torch.cat([cond_norm, tvec], dim=1)
        gamma_beta = self.film_time(cond_in)
        gamma, beta = gamma_beta[:, :h.shape[1]], gamma_beta[:, h.shape[1]:]
        h = h * (1 + 0.1 * gamma) + 0.1 * beta

        gate = torch.sigmoid(self.scale_gate(h))          # [B, m_params]
        scale = self.delta_scale * gate

        z = torch.cat([h, feat_shared, params_pred_norm], dim=1)
        raw = self.delta_head(z)
        delta = torch.tanh(raw) * scale
        return delta


# ============================================================
# 7) Modèle principal (+ Baseline Fix) + contrôles de STAGE & FiLM
# ============================================================
class PhysicallyInformedAE(pl.LightningModule):
    def __init__(
        self,
        n_points: int,
        param_names: List[str],
        poly_freq_CH4,
        transitions_dict,
        ranges_train: dict | None = None,
        ranges_val: dict | None = None,
        noise_train: dict | None = None,
        noise_val: dict | None = None,
        # --- optims & pondérations ---
        lr: float = 1e-4,
        alpha_param: float = 1.0,
        alpha_phys: float = 1.0,
        # --- têtes de sortie ---
        head_mode: str = "multi",
        predict_params: Optional[List[str]] = None,
        film_params: Optional[List[str]] = None,
        # --- raffinement ---
        refine_steps: int = 1,
        refine_delta_scale: float = 0.1,
        refine_target: str = "noisy",
        refine_warmup_epochs: int = 30,
        freeze_base_epochs: int = 20,
        base_lr: float = None,
        refiner_lr: float = None,
        stage3_lr_shrink: float = 0.33,
        stage3_refine_steps: Optional[int] = 2,
        stage3_delta_scale: Optional[float] = 0.08,
        stage3_alpha_phys: Optional[float] = 0.7,
        stage3_alpha_param: Optional[float] = 0.3,
        # --- reconstruction / pertes ---
        recon_max1: bool = False,
        corr_mode: str = "savgol",
        corr_savgol_win: int = 11,
        corr_savgol_poly: int = 3,
        huber_beta: float = 0.002,  
        weight_mf: float = 1.0,
        # --- EfficientNet backbone  ---
        backbone_variant: str = "s",
        refiner_variant: str  = "s",
        backbone_width_mult: float = 1.0,
        backbone_depth_mult: float = 1.0,
        refiner_width_mult: float = 1.0,
        refiner_depth_mult: float = 1.0,
        backbone_stem_channels: int | None = None,
        refiner_stem_channels: int | None = None,
        backbone_drop_path: float = 0.1,
        refiner_drop_path: float = 0.1,
        backbone_se_ratio: float = 0.25,
        refiner_se_ratio: float = 0.25,
        refiner_feature_pool: str = "avg",
        refiner_shared_hidden_scale: float = 0.5,
        refiner_time_embed_dim: int | None = None,
    ):
        super().__init__()

        self.param_names = list(param_names); self.n_params = len(self.param_names)
        self.n_points = n_points
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict
        self.save_hyperparameters(ignore=["transitions_dict", "poly_freq_CH4"])

        # --- optims / pondérations ---
        self.lr = float(lr)
        self.alpha_param = float(alpha_param); self.alpha_phys = float(alpha_phys)
        self.huber_beta = float(huber_beta)

        # --- raffinement ---
        self.refine_steps = int(refine_steps)
        self.refine_target = refine_target.lower(); assert self.refine_target in {"noisy", "clean"}
        self.refine_warmup_epochs = int(refine_warmup_epochs)
        self.freeze_base_epochs = int(freeze_base_epochs)
        self.base_lr = float(base_lr) if base_lr is not None else self.lr
        self.refiner_lr = float(refiner_lr) if refiner_lr is not None else self.lr
        self._froze_base = False

        self.stage3_lr_shrink = float(stage3_lr_shrink)
        self.stage3_refine_steps = stage3_refine_steps
        self.stage3_delta_scale = stage3_delta_scale
        self.stage3_alpha_phys = stage3_alpha_phys
        self.stage3_alpha_param = stage3_alpha_param

        # --- pertes / métriques ---
        self.weight_mf = float(weight_mf)
        self.corr_mode = str(corr_mode).lower()
        self.corr_savgol_win = int(corr_savgol_win)
        self.corr_savgol_poly = int(corr_savgol_poly)
        self.recon_max1 = bool(recon_max1)

        # --- paramètres à prédire / fournis ---
        if predict_params is None:
            predict_params = self.param_names
        self.predict_params = list(predict_params)
        self.provided_params = [p for p in self.param_names if p not in self.predict_params]
        unknown = set(self.predict_params) - set(self.param_names)
        assert not unknown, f"Paramètres inconnus: {unknown}"
        assert len(self.predict_params) >= 1

        if film_params is None:
            self.film_params = list(self.provided_params)
        else:
            self.film_params = list(film_params)
        bad = set(self.film_params) - set(self.param_names)
        assert not bad, f"film_params inconnus: {bad}"
        not_provided = set(self.film_params) - set(self.provided_params)
        assert not not_provided, f"film_params doit ⊆ provided_params; conflits: {not_provided}"

        self.name_to_idx = {n: i for i, n in enumerate(self.param_names)}
        self.predict_idx = [self.name_to_idx[p] for p in self.predict_params]
        self.provided_idx = [self.name_to_idx[p] for p in self.provided_params]

        # ===== Backbone EfficientNet 1D (expose width/depth/stem/drop/se) =====
        self.backbone = EfficientNetEncoder(
            in_channels=1,
            variant=backbone_variant,
            width_mult=backbone_width_mult,
            depth_mult=backbone_depth_mult,
            se_ratio=backbone_se_ratio,
            drop_path_rate=backbone_drop_path,
            stem_channels=backbone_stem_channels,
        )
        self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        feat_dim = self.backbone.feat_dim; hidden = feat_dim // 2

        self.shared_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
        )

        self.cond_dim = len(self.film_params)
        if self.cond_dim > 0:
            self.film = nn.Sequential(nn.Linear(self.cond_dim, hidden), nn.Tanh(), nn.Linear(hidden, 2 * hidden))
        else:
            self.film = None

        # ----- FiLM runtime switches -----
        self.use_film = True
        if self.cond_dim > 0:
            self.register_buffer("film_mask", torch.ones(self.cond_dim))
        else:
            self.film_mask = None

        self.head_mode = str(head_mode).lower(); assert self.head_mode in {"single", "multi"}
        if self.head_mode == "single":
            self.out_head = nn.Linear(hidden, len(self.predict_params))
        else:
            self.out_heads = nn.ModuleDict({p: nn.Linear(hidden, 1) for p in self.predict_params})

        # ===== Refiner EfficientNet 1D (expose width/depth/stem/drop/se & tête) =====
        self.refiner = EfficientNetRefiner(
            m_params=len(self.predict_params),
            cond_dim=self.cond_dim,
            backbone_feat_dim=self.backbone.feat_dim,
            delta_scale=refine_delta_scale,
            max_refine_steps=max(3, self.refine_steps if isinstance(self.refine_steps, int) else 3),
            encoder_variant=refiner_variant,
            encoder_width_mult=refiner_width_mult,
            encoder_depth_mult=refiner_depth_mult,
            encoder_stem_channels=refiner_stem_channels,
            encoder_drop_path=refiner_drop_path,
            encoder_se_ratio=refiner_se_ratio,
            feature_pool=refiner_feature_pool,
            shared_hidden_scale=refiner_shared_hidden_scale,
            time_embed_dim=refiner_time_embed_dim,
        )

        # ===== ReLoBRaLo =====
        self.loss_names_params = [f"param_{p}" for p in self.predict_params]
        self.relo_params = ReLoBRaLoLoss(self.loss_names_params, alpha=0.9, tau=1.0, history_len=10)
        self.loss_names_top = ["phys_mse", "phys_corr", "param_group"]
        self.relo_top = ReLoBRaLoLoss(self.loss_names_top, alpha=0.9, tau=1.0, history_len=10)

        # ----- Stage override (A/B1/B2) -----
        self._override_stage: Optional[str] = None
        self._override_refine_steps: Optional[int] = None
        self._override_delta_scale: Optional[float] = None


    # ==== FiLM runtime control ====
    def set_film_usage(self, use: bool = True):
        self.use_film = bool(use)
        if hasattr(self, "refiner") and hasattr(self.refiner, "use_film"):
            self.refiner.use_film = self.use_film

    def set_film_subset(self, names=None):
        if self.cond_dim == 0 or self.film_mask is None: return
        if names is None or names == "all":
            mask = torch.ones(self.cond_dim, device=self.film_mask.device, dtype=self.film_mask.dtype)
        else:
            allowed = set(names)
            mask = torch.zeros(self.cond_dim, device=self.film_mask.device, dtype=self.film_mask.dtype)
            for i, n in enumerate(self.film_params):
                if n in allowed: mask[i] = 1.0
        self.film_mask.copy_(mask)

    # ==== Stage override ====
    def set_stage_mode(self, mode: Optional[str], refine_steps: Optional[int]=None, delta_scale: Optional[float]=None):
        """
        mode: None / 'A' / 'B1' / 'B2'
        """
        if mode is not None:
            mode = mode.upper(); assert mode in {'A','B1','B2'}
        self._override_stage = mode
        self._override_refine_steps = refine_steps
        self._override_delta_scale = delta_scale
        if delta_scale is not None:
            self.refiner.delta_scale = float(delta_scale)
        if refine_steps is not None:
            self.refine_steps = int(refine_steps)

    # ---- utils tête ----
    def _predict_params_from_features(self, feat: torch.Tensor, cond_norm: Optional[torch.Tensor]=None) -> torch.Tensor:
        h = self.shared_head(feat)
        if self.film is not None and cond_norm is not None and self.use_film:
            if self.film_mask is not None and self.film_mask.numel() == cond_norm.shape[1]:
                cond_in = cond_norm * self.film_mask.unsqueeze(0)
            else:
                cond_in = cond_norm
            gb = self.film(cond_in); H = h.shape[1]
            gamma, beta = gb[:, :H], gb[:, H:]
            h = h * (1 + 0.1*gamma) + 0.1*beta
        if self.head_mode == "single":
            y = self.out_head(h)
        else:
            y = torch.cat([self.out_heads[p](h) for p in self.predict_params], dim=1)
        return torch.sigmoid(y).clamp(1e-4, 1-1e-4)

    def encode(self, spectra: torch.Tensor, pooled: bool=True, detach: bool=False):
        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat = self.feature_head(latent) if pooled else latent
        return feat.detach() if detach else feat

    # ---- (dé)normalisation & physique ----
    def _denorm_params_subset(self, y_norm_subset: torch.Tensor, names: List[str]) -> torch.Tensor:
        cols = [unnorm_param_torch(n, y_norm_subset[:, i]) for i, n in enumerate(names)]
        return torch.stack(cols, dim=1)

    def _compose_full_phys(self, pred_phys: torch.Tensor, provided_phys_tensor: torch.Tensor) -> torch.Tensor:
        B = pred_phys.shape[0]
        full = pred_phys.new_empty((B, self.n_params))
        for j, idx in enumerate(self.predict_idx):  full[:, idx] = pred_phys[:, j]
        for j, idx in enumerate(self.provided_idx): full[:, idx] = provided_phys_tensor[:, j]
        return full

    def _physics_reconstruction(self, y_phys_full: torch.Tensor, device, scale: Optional[torch.Tensor]=None) -> torch.Tensor:
        p = {k: y_phys_full[:, i] for i, k in enumerate(self.param_names)}
        v_grid_idx = torch.arange(self.n_points, dtype=torch.float64, device=device)
        b0_idx = self.name_to_idx['baseline0']
        baseline_coeffs = y_phys_full[:, b0_idx:b0_idx+3]
        spectra, _ = batch_physics_forward_multimol_vgrid(
            p['sig0'], p['dsig'], self.poly_freq_CH4, v_grid_idx,
            baseline_coeffs, self.transitions_dict, p['P'], p['T'], {'CH4': p['mf_CH4']}, device=device
        )
        spectra = spectra.to(torch.float32)

        scale_recon = lowess_value(spectra, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)
        spectra = spectra / scale_recon
        return spectra

    # ---- FiLM conditions ----
    def _make_condition_from_norm(self, params_true_norm: torch.Tensor) -> Optional[torch.Tensor]:
        if self.cond_dim == 0: return None
        cols = [params_true_norm[:, self.name_to_idx[n]].unsqueeze(1) for n in self.film_params]
        return torch.cat(cols, dim=1)

    def _make_condition_from_phys(self, provided_phys: dict, device, dtype=torch.float32) -> Optional[torch.Tensor]:
        if self.cond_dim == 0: return None
        missing = [n for n in self.film_params if n not in provided_phys]
        if missing: raise ValueError(f"FiLM: manquants dans provided_phys: {missing}")
        cols = []
        for n in self.film_params:
            v = provided_phys[n].to(device)
            if v.ndim > 1: v = v.view(-1)
            cols.append(norm_param_torch(n, v).unsqueeze(1))
        return torch.cat(cols, dim=1).to(dtype)

    # ---- Baseline-fix utils ----
    def _fit_residual_baseline_poly(self, resid: torch.Tensor, degree: int, sideband: int) -> torch.Tensor:
        B, N = resid.shape
        device = resid.device; dtype = resid.dtype
        degree = int(max(0, min(2, degree)))
        sideband = int(max(1, min(N//2, sideband)))
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[:sideband] = True; mask[N - sideband:] = True
        x = torch.arange(N, device=device, dtype=dtype)
        Xcols = [torch.ones_like(x)]
        if degree >= 1: Xcols.append(x)
        if degree >= 2: Xcols.append(x * x)
        X = torch.stack(Xcols, dim=1)
        Xm = X[mask]
        Y  = resid[:, mask]
        XtX = Xm.T @ Xm
        XtX_inv = torch.linalg.pinv(XtX)
        XtY = Xm.T @ Y.T
        coeffs = (XtX_inv @ XtY).T
        return coeffs

    # ---------- Savitzky–Golay derivative (torch, no SciPy) ----------
    def _savgol_coeffs(self, window_length: int, polyorder: int, deriv: int = 1, delta: float = 1.0,
                       device=None, dtype=torch.float64) -> torch.Tensor:
        """
        Returns 1D convolution coefficients (length W) to estimate the 'deriv' derivative
        at the window center, for uniform sampling step 'delta'.
        """
        assert deriv >= 0
        W = int(window_length)
        P = int(polyorder)
        if W % 2 == 0: W += 1                  # make odd
        if W < 3: W = 3
        if P >= W: P = W - 1
        m = (W - 1) // 2

        dev = device or self.device
        x = torch.arange(-m, m + 1, device=dev, dtype=dtype)  # [-m..m], shape [W]
        # Design matrix: A[i, j] = x_i^j, shape [W, P+1]
        A = torch.stack([x**j for j in range(P + 1)], dim=1)  # [W, P+1]
        pinv = torch.linalg.pinv(A)                            # [P+1, W]
        # derivative at 0: d! * coefficient of x^d
        coeff = math.factorial(deriv) * pinv[deriv, :] / (delta ** deriv)  # [W]
        return coeff.to(dtype=torch.float32)

    def _savgol_deriv(self, y: torch.Tensor, window_length: int, polyorder: int, deriv: int = 1) -> torch.Tensor:
        """
        Apply SG derivative filter with 'reflect' padding. y: [B, N] -> [B, N]
        """
        B, N = y.shape
        # make parameters safe vs N
        W = int(window_length)
        if W % 2 == 0: W += 1
        if W > N:
            W = N if (N % 2 == 1) else (N - 1)
        W = max(W, 3)
        P = min(int(polyorder), W - 1)

        coeff = self._savgol_coeffs(W, P, deriv=deriv, device=y.device, dtype=torch.float64)  # [W]
        coeff = coeff.view(1, 1, -1)  # [out_ch, in_ch, W]

        pad = (W - 1) // 2
        y1 = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")  # [B,1,N+2*pad]
        out = F.conv1d(y1, coeff).squeeze(1)                    # [B, N]
        return out

    def _dx(self, y: torch.Tensor) -> torch.Tensor:
        """Central difference derivative, same length via interior points + pad endpoints."""
        # interior: centered diff
        d = 0.5 * (y[:, 2:] - y[:, :-2])  # [B, N-2]
        # pad ends by replication to keep same length
        left  = d[:, :1]
        right = d[:, -1:]
        return torch.cat([left, d, right], dim=1)

    def _pearson_corr_loss(self, y_hat: torch.Tensor, y: torch.Tensor,
                           eps: float = 1e-8,
                           derivative: str | bool = False,
                           savgol_win: int = 11,
                           savgol_poly: int = 3) -> torch.Tensor:
        """
        Pearson correlation loss. If derivative:
          - "none"/False : on raw signal
          - True/"central": on central-diff derivative
          - "savgol": on Savitzky–Golay derivative (window/poly configurable)
        """
        mode = derivative
        if isinstance(derivative, bool):
            mode = "central" if derivative else "none"
        mode = (mode or "none").lower()

        if mode == "central":
            y_hat = self._dx(y_hat)
            y     = self._dx(y)

        elif mode in ("savgol", "sg", "savitzky_golay", "savitzky-golay"):
            y_hat = self._savgol_deriv(y_hat, savgol_win, savgol_poly, deriv=1)
            y     = self._savgol_deriv(y,     savgol_win, savgol_poly, deriv=1)
        # else "none": raw

        y_hat = y_hat.float(); y = y.float()
        y_hat_c = y_hat - y_hat.mean(dim=1, keepdim=True)
        y_c     = y - y.mean(dim=1, keepdim=True)
        num = (y_hat_c * y_c).sum(dim=1)
        den = torch.sqrt((y_hat_c.pow(2).sum(dim=1) + eps) * (y_c.pow(2).sum(dim=1) + eps))
        corr = num / den
        return (1.0 - corr).mean()


    # ---- training/validation step ----
    def _common_step(self, batch, step_name: str):
        noisy, clean, params_true_norm = batch['noisy_spectra'], batch['clean_spectra'], batch['params']
        scale = batch.get('scale', None)
        if scale is not None:
            scale = scale.to(clean.device)

        # FiLM condition (norm)
        cond_norm = self._make_condition_from_norm(params_true_norm)

        # Encode + prédiction initiale (normée)
        latent, _ = self.backbone(noisy.unsqueeze(1))
        feat_shared = self.feature_head(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared, cond_norm=cond_norm)

        # Tenseur des paramètres "fournis" (physiques)
        if len(self.provided_params) > 0:
            provided_cols = [params_true_norm[:, self.name_to_idx[n]] for n in self.provided_params]
            provided_norm_tensor = torch.stack(provided_cols, dim=1)
            provided_phys_tensor = self._denorm_params_subset(provided_norm_tensor, self.provided_params)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((noisy.shape[0], 0))

        # Planning raffinement
        e = self.current_epoch
        effective_refine_steps = 0 if e < self.refine_warmup_epochs else self.refine_steps
        if self._override_stage is not None:
            if self._override_stage == 'A':   effective_refine_steps = 0
            elif self._override_stage in ('B1', 'B2'): effective_refine_steps = self.refine_steps

        target_for_resid = noisy if self.refine_target == "noisy" else clean

        # Boucle de raffinement (avec échelles par-paramètre prises en charge dans le raffineur)
        # ======= BOUCLE DE RAFFINEMENT AVEC t_step =======
        spectra_recon = None
        for s in range(effective_refine_steps):
            pred_phys  = self._denorm_params_subset(params_pred_norm, self.predict_params)
            y_phys_full= self._compose_full_phys(pred_phys, provided_phys_tensor)
            spectra_recon = self._physics_reconstruction(y_phys_full, clean.device, scale=None)
            resid = spectra_recon - target_for_resid

            # Raffineur → delta par paramètre (bounded + scaled inside)
            delta = self.refiner(
                noisy=noisy,
                resid=resid,
                params_pred_norm=params_pred_norm,
                cond_norm=cond_norm,
                feat_shared=feat_shared,
                t_step=s,                    
            )
            params_pred_norm = (params_pred_norm + delta).clamp(1e-4, 1-1e-4)

        # Reconstruction finale
        pred_phys   = self._denorm_params_subset(params_pred_norm, self.predict_params)
        y_phys_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
        spectra_recon = self._physics_reconstruction(y_phys_full, clean.device, scale=None)

        # --- Pertes physiques : MSE + Huber (mélangées) + corrélation dérivée/SG si activée ---
        #loss_phys_mse   = F.mse_loss(spectra_recon, clean)
        loss_phys_huber = F.smooth_l1_loss(spectra_recon, clean, beta=self.huber_beta)
        loss_phys_comb  = loss_phys_huber 

        loss_phys_corr = self._pearson_corr_loss(
            spectra_recon, clean,
            derivative=self.corr_mode,
            savgol_win=self.corr_savgol_win,
            savgol_poly=self.corr_savgol_poly
        )

        # --- Pertes paramètres (groupe ReLoBRaLo inchangé) ---
        per_param_losses = []
        for j, name in enumerate(self.predict_params):
            true_j = params_true_norm[:, self.name_to_idx[name]]
            mult = self.weight_mf if name == "mf_CH4" else 1.0
            lp = mult * F.mse_loss(params_pred_norm[:, j], true_j)
            per_param_losses.append(lp)

        if len(per_param_losses) > 0:
            per_param_tensor = torch.stack(per_param_losses)
            w_params = self.relo_params.compute_weights(per_param_tensor)
            w_params_norm = w_params / (w_params.sum() + 1e-12)
            loss_param_group = torch.sum(w_params_norm * per_param_tensor)
        else:
            loss_param_group = torch.tensor(0.0, device=clean.device)

        # --- Agrégation top (garde 3 têtes pour rester compatible avec self.relo_top) ---
        top_vec = torch.stack([loss_phys_comb, loss_phys_corr, loss_param_group])  # [phys, corr, params]
        w_top = self.relo_top.compute_weights(top_vec)
        priors_top = torch.tensor([self.alpha_phys, self.alpha_phys, self.alpha_param],
                                device=top_vec.device, dtype=top_vec.dtype)
        w_top = w_top * priors_top
        w_top = 3.0 * w_top / (w_top.sum() + 1e-12)
        loss = torch.sum(w_top * top_vec)

        # Logs
        self.log(f"{step_name}_loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_phys_huber",loss_phys_huber, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_phys_corr", loss_phys_corr,  on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_param_group", loss_param_group, on_epoch=True, sync_dist=True)
        if len(per_param_losses) > 0:
            self.log(f"{step_name}_loss_param", torch.stack(per_param_losses).mean(), on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_w_top_phys",        w_top[0], on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_w_top_phys_corr",   w_top[1], on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_w_top_param_group", w_top[2], on_epoch=True, sync_dist=True)
        for j, name in enumerate(self.predict_params):
            self.log(f"{step_name}_loss_param_{name}", per_param_losses[j], on_epoch=True, sync_dist=True)

        return loss


    def training_step(self, batch, batch_idx): return self._common_step(batch, "train")
    def validation_step(self, batch, batch_idx): self._common_step(batch, "val")

    def on_train_epoch_start(self):
        # Stage override : si défini, on ignore le scheduler A/B1/B2 basé sur epochs
        if self._override_stage is not None:
            stage = self._override_stage
            if stage == 'A':
                self._set_requires_grad(self.refiner, False)
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None),
                                         getattr(self, "out_heads", None), self.film], True)
            elif stage == 'B1':
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None),
                                         getattr(self, "out_heads", None), self.film], False)
                self._set_requires_grad(self.refiner, True)
            elif stage == 'B2':
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None),
                                         getattr(self, "out_heads", None), self.film, self.refiner], True)
            return

        # sinon: plan automatique (optionnel si tu utilises un seul fit)
        e = self.current_epoch
        stage3_start = self.refine_warmup_epochs + self.freeze_base_epochs

        if e < self.refine_warmup_epochs:
            self._set_requires_grad(self.refiner, False)
            self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None),
                                     getattr(self, "out_heads", None), self.film], True)
            self._froze_base = False
        elif e < stage3_start:
            if not self._froze_base:
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None),
                                         getattr(self, "out_heads", None), self.film], False)
                self._set_requires_grad(self.refiner, True)
                self._froze_base = True
        else:
            self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None),
                                     getattr(self, "out_heads", None), self.film, self.refiner], True)
            if e == stage3_start:
                if hasattr(self.trainer, "optimizers") and len(self.trainer.optimizers) > 0:
                    opt = self.trainer.optimizers[0]
                    for pg in opt.param_groups: pg["lr"] *= self.stage3_lr_shrink
                if self.stage3_refine_steps is not None: self.refine_steps = int(self.stage3_refine_steps)
                if self.stage3_delta_scale is not None: self.refiner.delta_scale = float(self.stage3_delta_scale)
                if self.stage3_alpha_phys is not None:  self.alpha_phys = float(self.stage3_alpha_phys)
                if self.stage3_alpha_param is not None: self.alpha_param = float(self.stage3_alpha_param)

    def _set_requires_grad(self, modules, flag: bool):
        if modules is None: return
        if not isinstance(modules, (list, tuple)): modules = [modules]
        for m in modules:
            if m is None: continue
            for p in m.parameters(): p.requires_grad_(flag)

    def configure_optimizers(self):
        # --- groupes de paramètres (comme dans ton code) ---
        base_params = list(self.backbone.parameters()) + list(self.shared_head.parameters())
        if hasattr(self, "out_head"):  base_params += list(self.out_head.parameters())
        if hasattr(self, "out_heads"): base_params += list(self.out_heads.parameters())
        if self.film is not None:      base_params += list(self.film.parameters())
        refiner_params = list(self.refiner.parameters())

        param_groups = [
            {"params": base_params,    "lr": float(getattr(self, "base_lr", self.lr))},
            {"params": refiner_params, "lr": float(getattr(self, "refiner_lr", self.lr))},
        ]

        # --- sélection d'optimiseur ---
        opt_name = getattr(self.hparams, "optimizer", "adamw").lower()
        weight_decay = getattr(self, "weight_decay", 1e-4)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        elif opt_name == "lion":
            # betas par défaut Lion : (0.9, 0.99); tu peux les passer via self.hparams.betas si tu veux
            betas = getattr(self.hparams, "betas", (0.9, 0.99))
            optimizer = Lion(param_groups, betas=betas, weight_decay=weight_decay)
        elif opt_name == "radam":
            import torch_optimizer as optim_plus
            optimizer = optim_plus.RAdam(param_groups, weight_decay=weight_decay)
        elif opt_name == "adabelief":
            import torch_optimizer as optim_plus
            optimizer = optim_plus.AdaBelief(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # --- scheduler identique à avant ---
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer is not None else 100,
            eta_min=1e-11
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    @torch.no_grad()
    def infer(
        self,
        spectra: torch.Tensor,
        provided_phys: dict,
        *,
        refine: bool = True,
        resid_target: str = "input",
        scale: Optional[torch.Tensor] = None,
    ):
        """
        Inférence cohérente avec la pipeline de normalisation:

        - 'spectra' (noisy) est déjà normalisé en entrée (par fenêtre gaussienne OU LOWESS),
        on NE le re-normalise PAS ici.
        - La reconstruction physique est renvoyée avec max(recon)=1 (cf. self.recon_max1=True).
        - Le résidu utilisé par le raffineur est: recon(max=1) - spectra(normalisé).

        Args:
            spectra        : [B, N] (noisy déjà normalisé au chargement des données)
            provided_phys  : dict des paramètres fournis (doit couvrir self.provided_params)
            refine         : active le raffineur si True
            resid_target   : "input"/"noisy" => cible du résidu = 'spectra' tel quel ; autre => pas de raffinage
            scale          : ignoré ici (compat), conservé pour rétro-compatibilité

        Returns:
            dict:
                - "params_pred_norm": [B, len(predict_params)] (dans [0,1])
                - "y_phys_full"    : [B, len(PARAMS)] (physiques dénormalisés)
                - "spectra_recon"  : [B, N] (reconstruction physique à max=1)
                - "norm_scale"     : [B]    échelle d'entrée estimée (info/log uniquement)
        """
        self.eval()
        device = spectra.device
        B = spectra.shape[0]

        # 1) Vérif couverture des paramètres fournis
        missing = [n for n in self.provided_params if n not in provided_phys]
        assert not missing, f"Manque des paramètres fournis: {missing}"

        # 2) Condition FiLM (normalisée) depuis 'provided_phys'
        cond_norm = self._make_condition_from_phys(provided_phys, device, dtype=torch.float32)

        # 3) Prédiction initiale (normée) depuis l'encodeur
        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat_shared = self.feature_head(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared, cond_norm=cond_norm)

        # 4) Tensor des paramètres fournis (physiques)
        if len(self.provided_params) > 0:
            provided_list = [provided_phys[n].to(device) for n in self.provided_params]
            provided_phys_tensor = torch.stack(provided_list, dim=1)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((B, 0))

        # 5) Cible du résidu: on prend 'spectra' tel quel (déjà normalisé en amont)
        spectra_target = spectra if resid_target in ("input", "noisy") else None

        # 6) Estimation d'échelle d'entrée (info/log) avec le MÊME estimateur que le dataset
        scale_est = lowess_value(spectra, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)

        # 7) Boucle de raffinement optionnelle
        if refine and self.refine_steps > 0:
            for s in range(self.refine_steps):
                pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
                y_full    = self._compose_full_phys(pred_phys, provided_phys_tensor)
                recon     = self._physics_reconstruction(y_full, device, scale=None)  # max=1

                if spectra_target is None:
                    break

                resid = recon - spectra_target  # résidu dans le repère cohérent

                delta = self.refiner(
                    noisy=spectra,
                    resid=resid,
                    params_pred_norm=params_pred_norm,
                    cond_norm=cond_norm,
                    feat_shared=feat_shared,
                    t_step=s,                
                )
                params_pred_norm = params_pred_norm.add(delta).clamp(1e-4, 1-1e-4)

        # 8) Passage final par la physique
        pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
        y_full    = self._compose_full_phys(pred_phys, provided_phys_tensor)
        recon     = self._physics_reconstruction(y_full, device, scale=None)  

        return {
            "params_pred_norm": params_pred_norm,
            "y_phys_full": y_full,
            "spectra_recon": recon,
            "norm_scale": scale_est,
        }


# ============================================================
# 8) Callbacks visu & epoch sync dataset
# ============================================================
class PlotAndMetricsCallback(Callback):
    """
    Génère des figures et les sauvegarde en PNG à la fin de chaque epoch de validation.
    - Aucune dépendance à IPython
    - Rank 0 uniquement (multi-noeuds / multi-GPU safe)
    - Dossier de sortie: ./figs_<SLURM_JOB_ID>/ par défaut
    """
    def __init__(self, val_loader, param_names, num_examples: int = 1,
                 save_dir: str | None = None, stage_tag: str | None = None):
        super().__init__()
        self.val_loader = val_loader
        self.param_names = list(param_names)
        self.num_examples = int(num_examples)
        if save_dir is None:
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            save_dir = f"./figs_{job_id}"
        self.stage_tag = stage_tag or "stage"
        self.save_dir = os.path.join(save_dir, self.stage_tag)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not is_rank0():  # ne trace qu'une fois
            return

        pl_module.eval()
        device = pl_module.device
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        noisy  = batch['noisy_spectra'][:self.num_examples].to(device)
        clean  = batch['clean_spectra'][:self.num_examples].to(device)
        params_true_norm = batch['params'][:self.num_examples].to(device)

        # Construire le dict des paramètres "fournis" (cohérent avec le modèle)
        provided_phys = {}
        for name in getattr(pl_module, "provided_params", []):
            idx = pl_module.name_to_idx[name]
            v_norm = params_true_norm[:, idx]
            v_phys = pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]
            provided_phys[name] = v_phys

        out = pl_module.infer(noisy, provided_phys=provided_phys, refine=True, resid_target="input")
        spectra_recon = out["spectra_recon"].detach().cpu()

        noisy_cpu, clean_cpu = noisy.detach().cpu(), clean.detach().cpu()
        x = np.arange(clean_cpu.shape[1])

        # Récup métriques (safe)
        m = trainer.callback_metrics
        def get_metric(name, default="-"):
            v = m.get(name, None)
            try: return f"{float(v):.6g}"
            except Exception: return default

        val_loss        = get_metric("val_loss")
        val_phys_huber  = get_metric("val_loss_phys_huber")
        val_phys_corr   = get_metric("val_loss_phys_corr")
        val_param_group = get_metric("val_loss_param_group")
        train_loss      = get_metric("train_loss")

        per_param_lines = []
        for p in getattr(pl_module, "predict_params", self.param_names):
            per_param_lines.append(f"{p:10s} : {get_metric(f'val_loss_param_{p}')}")
        per_param_text = "\n".join(per_param_lines)

        # Figure compacte avec specs + résidus + tableau de métriques
        fig = plt.figure(figsize=(11, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.8], height_ratios=[3, 1], hspace=0.25, wspace=0.3)
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_spec)
        ax_tbl  = fig.add_subplot(gs[:, 1]); ax_tbl.axis("off")

        i = 0  # on trace le premier exemple (tu peux faire une boucle si tu veux en sauver plusieurs)
        ax_spec.plot(x, noisy_cpu[i],         label="Noisy",       lw=1, alpha=0.7)
        ax_spec.plot(x, clean_cpu[i],         label="Clean (réel)",lw=1.5)
        ax_spec.plot(x, spectra_recon[i],     label="Reconstruit", lw=1.2, ls="--")
        ax_spec.set_ylabel("Transmission")
        ax_spec.set_title(f"Epoch {trainer.current_epoch}")
        ax_spec.legend(frameon=False, fontsize=9)

        resid_clean = spectra_recon[i] - clean_cpu[i]
        resid_noisy = spectra_recon[i] - noisy_cpu[i]
        ax_res.plot(x, resid_noisy, lw=1,   label="Reconstruit - Noisy")
        ax_res.plot(x, resid_clean, lw=1.2, label="Reconstruit - Clean")
        ax_res.axhline(0, ls=":", lw=0.8)
        ax_res.set_xlabel("Points spectraux"); ax_res.set_ylabel("Résidu")
        ax_res.legend(frameon=False, fontsize=9)

        header = f"Métriques (epoch {trainer.current_epoch})"
        lines  = [
            f"train_loss : {train_loss}",
            f"val_loss   : {val_loss}",
            f"val_phys_huber : {val_phys_huber}",
            f"val_corr   : {val_phys_corr}",
            f"val_param_group : {val_param_group}",
            "", "Pertes par paramètre (val) :", per_param_text
        ]
        ax_tbl.text(0.02, 0.98, header, va="top", ha="left", fontsize=12, fontweight="bold")
        ax_tbl.text(0.02, 0.90, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

        for ax in (ax_spec, ax_res): ax.grid(alpha=0.25)

        out_png = os.path.join(
            self.save_dir, f"{self.stage_tag}_val_epoch{trainer.current_epoch:04d}.png"
        )
        save_fig(fig, out_png)

class UpdateEpochInDataset(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        ds = trainer.train_dataloaders.dataset if hasattr(trainer, "train_dataloaders") else trainer.train_dataloader.dataset
        if hasattr(ds, "set_epoch"): ds.set_epoch(trainer.current_epoch)

class AdvanceDistributedSamplerEpoch(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        try:
            sampler = loader.sampler
        except Exception:
            sampler = getattr(trainer, "train_dataloader", None).sampler if hasattr(trainer, "train_dataloader") else None
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)

# ============================================================
# 9) Helpers ranges
# ============================================================
def expand_interval(a, b, factor):
    c = 0.5*(a+b); half = 0.5*(b-a)*float(factor)
    return float(c-half), float(c+half)

def map_ranges(base: dict, fn, per_param: dict | None = None) -> dict:
    out = {}
    for k, (a, b) in base.items():
        f = per_param.get(k, per_param.get("_default", 1.0)) if per_param else 1.0
        out[k] = fn(a, b, f)
    return out

def assert_subset(child: dict, parent: dict, name_child="child", name_parent="parent"):
    bad = [k for k in child if not (parent[k][0] <= child[k][0] and child[k][1] <= parent[k][1])]
    if bad:
        raise ValueError(f"{name_child} ⊄ {name_parent} pour: {bad}")

# ============================================================
# 10) Stages d'entraînement (A / B1 / B2), contrôle fin
# ============================================================
def _freeze_all(model: PhysicallyInformedAE):
    for p in model.parameters(): p.requires_grad_(False)

def _set_trainable_heads(model: PhysicallyInformedAE, names: Optional[List[str]]):
    if model.head_mode == "single":
        # on ne sait pas geler par colonne proprement → tout ON
        for p in model.out_head.parameters(): p.requires_grad_(True)
        return
    # multi-head : on active uniquement les heads désirés
    want = set(names) if names is not None else set(model.predict_params)
    for n, head in model.out_heads.items():
        req = (n in want)
        for p in head.parameters(): p.requires_grad_(req)

def _apply_stage_freeze(model: PhysicallyInformedAE,
                        train_base: bool, train_heads: bool, train_film: bool, train_refiner: bool,
                        heads_subset: Optional[List[str]]):
    # tout OFF
    _freeze_all(model)
    # backbone + shared head
    if train_base:
        for p in model.backbone.parameters(): p.requires_grad_(True)
        for p in model.shared_head.parameters(): p.requires_grad_(True)
    # heads
    if model.head_mode == "single":
        if train_heads:
            for p in model.out_head.parameters(): p.requires_grad_(True)
    else:
        _set_trainable_heads(model, heads_subset if train_heads else [])
    # FiLM
    if model.film is not None and train_film:
        for p in model.film.parameters(): p.requires_grad_(True)
    # refiner
    if train_refiner:
        for p in model.refiner.parameters(): p.requires_grad_(True)

def _load_weights_if_any(model, ckpt_path):
    import os, torch
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[INFO] No checkpoint at {ckpt_path}; skipping.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    model_sd = model.state_dict()
    compatible = {}
    mismatched, unexpected = [], []

    for k, v in state_dict.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                compatible[k] = v
            else:
                mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            unexpected.append(k)

    if not compatible:
        print("[WARN] Checkpoint is architecturally incompatible (0 matching shapes). "
              "Skipping weight loading for this run.")
        return

    # load the matching subset without erroring on the rest
    model_sd.update(compatible)
    model.load_state_dict(model_sd, strict=False)

    print(f"[INFO] Loaded {len(compatible)} tensors from checkpoint; "
          f"{len(mismatched)} mismatched, {len(unexpected)} unexpected, "
          f"{len(model_sd) - len(compatible)} missing.")


def _save_checkpoint(trainer: pl.Trainer, ckpt_out: Optional[str]):
    if ckpt_out and on_rank_zero():
        trainer.save_checkpoint(ckpt_out)
        print(f"✓ checkpoint sauvegardé: {ckpt_out}")

def get_worker_info():
    rank = int(os.environ.get("SLURM_PROCID") or os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or 0)
    world = int(os.environ.get("SLURM_NTASKS") or os.environ.get("WORLD_SIZE") or 1)
    return rank, world

def wait_for_file(path, check_every=5, timeout=24*3600):
    import time
    t0 = time.time()
    while not os.path.isfile(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Timeout en attendant {path}")
        time.sleep(check_every)
    return path

def train_stage_custom(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    stage_name: str,
    epochs: int,
    base_lr: float,
    refiner_lr: float,
    train_base: bool,
    train_heads: bool,
    train_film: bool,
    train_refiner: bool,
    refine_steps: int,
    delta_scale: float,
    use_film: Optional[bool] = None,
    film_subset: Optional[List[str]] = None,
    heads_subset: Optional[List[str]] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    **trainer_kwargs,
):
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import Strategy

    print(f"\n===== Stage {stage_name} =====")

    ckpt_in  = trainer_kwargs.pop("ckpt_in",  None)
    ckpt_out = trainer_kwargs.pop("ckpt_out", None)

    _load_weights_if_any(model, ckpt_in)

    # Nettoyage de la barre de progression si demandé
    try:
        from pytorch_lightning.callbacks.progress import TQDMProgressBar
        if callbacks is not None and enable_progress_bar is False:
            callbacks = [cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)]
    except Exception:
        pass

    # Overrides runtime
    model.base_lr = float(base_lr)
    model.refiner_lr = float(refiner_lr)
    model.set_stage_mode(stage_name, refine_steps=refine_steps, delta_scale=delta_scale)
    if use_film is not None: model.set_film_usage(bool(use_film))
    if film_subset is not None: model.set_film_subset(film_subset)

    _apply_stage_freeze(
        model,
        train_base=train_base, train_heads=train_heads, train_film=train_film, train_refiner=train_refiner,
        heads_subset=heads_subset,
    )

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    # Si l’appelant a fourni un objet Strategy, laisse Lightning déduire l’accélérateur
    strat = trainer_kwargs.get("strategy", None)
    if isinstance(strat, Strategy):
        trainer_kwargs.pop("accelerator", None)

    # ⚙️ IMPORTANT: DataLoaders distribués explicites (évite un remplacement auto par PL)
    train_loader, val_loader = ensure_distributed_samplers(train_loader, val_loader)

    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)
    return model



# --- helpers JSON best artefacts ---
from pathlib import Path
import json

def _dump_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))

def _save_best_params_json(stage: str, best_trial, ckpt_path: str, stage_dir: Path) -> str:
    """
    Écrit checkpoints/<stage>_best_params.json avec:
      - stage, score, params (hyperparams Optuna), ckpt (chemin best .ckpt)
    """
    payload = {
        "stage": stage,
        "score": float(best_trial.value),
        "params": dict(best_trial.params),
        "ckpt": str(ckpt_path),
    }
    out = stage_dir / "checkpoints" / f"{stage}_best_params.json"
    _dump_json(out, payload)
    return str(out)


# Facades conviviales
def train_stage_A(model, train_loader, val_loader, **kw):
    # A = base + heads (+/- FiLM), pas de raffineur
    defaults = dict(
        stage_name="A", epochs=20,
        base_lr=2e-4, refiner_lr=1e-6,
        train_base=True, train_heads=True, train_film=False, train_refiner=False,
        refine_steps=0, delta_scale=0.1,
        use_film=False, film_subset=None, heads_subset=None,
         enable_progress_bar=False
    ); defaults.update(kw)
    return train_stage_custom(model, train_loader, val_loader, **defaults)

def train_stage_B1(model, train_loader, val_loader, **kw):
    # B1 = raffineur seul
    defaults = dict(
        stage_name="B1", epochs=12,
        base_lr=1e-6, refiner_lr=1e-5,
        train_base=False, train_heads=False, train_film=False, train_refiner=True,
        refine_steps=2, delta_scale=0.12,
        use_film=True, film_subset=["T"], heads_subset=None,
         enable_progress_bar=False
    ); defaults.update(kw)
    return train_stage_custom(model, train_loader, val_loader, **defaults)

def train_stage_B2(model, train_loader, val_loader, **kw):
    # B2 = fine-tune global (petits LR), raffinement ON
    defaults = dict(
        stage_name="B2", epochs=15,
        base_lr=3e-5, refiner_lr=3e-6,
        train_base=True, train_heads=True, train_film=True, train_refiner=True,
        refine_steps=2, delta_scale=0.08,
        use_film=True, film_subset=["P","T"], heads_subset=None,
         enable_progress_bar=False
    ); defaults.update(kw)
    return train_stage_custom(model, train_loader, val_loader, **defaults)

def export_param_errors_files(ckpt_path: str, which: str, *, refine: bool, split: str = "val",
                              robust_smape: bool = False) -> dict:
    """
    Charge le modèle + données, restaure le checkpoint, puis écrit:
      - eval/<which>_<split>_metrics.csv          (agrégé)
      - eval/<which>_<split>_errors_per_sample.csv (par-échantillon)
    Retourne les chemins des fichiers.
    """
    from pathlib import Path
    run_dir = Path(make_run_dir())
    stage_dir = get_stage_dir(run_dir, which)
    out_dir = stage_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recrée un modèle compatible + loaders (puis load poids)
    model, train_loader, val_loader = build_data_and_model()
    _load_weights_if_any(model, ckpt_path)
    loader = val_loader if split == "val" else train_loader
    tag = f"{which}_{split}"

    # Sauvegardes CSV à l’intérieur de evaluate_and_plot
    _ = evaluate_and_plot(
        model, loader,
        n_show=3, refine=refine, robust_smape=robust_smape,
        save_dir=str(out_dir), tag=tag, save_per_sample=True
    )

    return {
        "metrics_csv": str(out_dir / f"{tag}_metrics.csv"),
        "per_sample_csv": str(out_dir / f"{tag}_errors_per_sample.csv"),
    }


@torch.no_grad()
def evaluate_and_plot(
    model: PhysicallyInformedAE, loader: DataLoader, n_show: int = 5,
    refine: bool = True, robust_smape: bool = False, eps: float = 1e-12, seed: int = 123,
    baseline_correction: dict | None = None,
    save_dir: str | None = None, tag: str | None = None,
    save_per_sample: bool = True,        # ← NEW: écrit aussi un CSV par-échantillon
):
    model.eval()
    device = model.device
    rng = random.Random(seed)

    pred_names = list(getattr(model, "predict_params", []))
    if not pred_names:
        raise RuntimeError("Aucun paramètre à évaluer.")

    per_param_err = {p: [] for p in pred_names}
    show_examples = []
    per_sample_rows = []                    # ← NEW
    sample_idx = 0                          # ← NEW compteur d’échantillons globaux

    for batch in loader:
        noisy  = batch["noisy_spectra"].to(device)
        clean  = batch["clean_spectra"].to(device)
        p_norm = batch["params"].to(device)
        B = noisy.size(0)

        provided_phys = {}
        if len(model.provided_params) > 0:
            cols = [p_norm[:, model.name_to_idx[n]] for n in model.provided_params]
            provided_norm_tensor = torch.stack(cols, dim=1)
            provided_phys_tensor = model._denorm_params_subset(provided_norm_tensor, model.provided_params)
            for j, name in enumerate(model.provided_params):
                provided_phys[name] = provided_phys_tensor[:, j]

        out = model.infer(noisy, provided_phys=provided_phys, refine=refine, resid_target="input")
        recon = out["spectra_recon"]
        y_full_pred = out["y_phys_full"].clone()

        true_cols = [p_norm[:, model.name_to_idx[n]] for n in pred_names]
        true_norm_tensor = torch.stack(true_cols, dim=1)
        true_phys = model._denorm_params_subset(true_norm_tensor, pred_names)
        pred_phys = torch.stack([y_full_pred[:, model.name_to_idx[n]] for n in pred_names], dim=1)

        if robust_smape:
            denom = pred_phys.abs() + true_phys.abs() + eps
            err_pct = 100.0 * 2.0 * (pred_phys - true_phys).abs() / denom
        else:
            denom = torch.clamp(true_phys.abs(), min=eps)
            err_pct = 100.0 * (pred_phys - true_phys).abs() / denom

        # agrégation par-paramètre
        for j, name in enumerate(pred_names):
            per_param_err[name].append(err_pct[:, j].detach().cpu())

        # NEW: enregistrement par-échantillon
        if save_per_sample:
            for i in range(B):
                row = {"sample": sample_idx + i}
                for j, name in enumerate(pred_names):
                    row[name] = float(err_pct[i, j])
                per_sample_rows.append(row)
            sample_idx += B

        # (… le bloc show_examples inchangé …)
        for i in range(B):
            if len(show_examples) < n_show:
                show_examples.append({
                    "noisy":  noisy[i].detach().cpu(),
                    "clean":  clean[i].detach().cpu(),
                    "recon":  recon[i].detach().cpu(),
                    "pred":   {name: float(pred_phys[i, j]) for j, name in enumerate(pred_names)},
                    "true":   {name: float(true_phys[i, j]) for j, name in enumerate(pred_names)},
                    "errpct": {name: float(err_pct[i, j])   for j, name in enumerate(pred_names)},
                })
        if len(show_examples) >= n_show: break

    # === Stats agrégées ===
    rows = []
    for name in pred_names:
        if len(per_param_err[name]) == 0: continue
        v = torch.cat(per_param_err[name])
        mean = v.mean().item(); med  = v.median().item()
        try:
            p90  = torch.quantile(v, torch.tensor(0.90)).item()
            p95  = torch.quantile(v, torch.tensor(0.95)).item()
        except Exception:
            vv = v.numpy(); p90 = float(np.quantile(vv, 0.90)); p95 = float(np.quantile(vv, 0.95))
        rows.append({"param": name, "mean_%": mean, "median_%": med, "p90_%": p90, "p95_%": p95})

    df = pd.DataFrame(rows).set_index("param").sort_index()
    print("\n=== Erreurs en % (globales, sur le loader) ==="); print(df.round(4))

    # === Sauvegardes CSV ===
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        tag = tag or "eval"
        csv_path = os.path.join(save_dir, f"{tag}_metrics.csv")
        df.round(6).to_csv(csv_path)
        print(f"✓ Metrics CSV: {csv_path}")

        if save_per_sample and len(per_sample_rows) > 0:
            df_samples = pd.DataFrame(per_sample_rows)
            csv_path2 = os.path.join(save_dir, f"{tag}_errors_per_sample.csv")
            df_samples.to_csv(csv_path2, index=False)
            print(f"✓ Per-sample CSV: {csv_path2}")

    # (… le reste de la fonction, les figures, inchangé …)
    return df

# ============================================================
# 12) Build data & modèle (exemple par défaut)
# ============================================================
def build_data_and_model(
    *,
    seed=42, n_points=800, n_train=100000, n_val=500, batch_size=16,
    train_ranges=None, val_ranges=None, noise_train=None, noise_val=None,
    predict_list=None, film_list=None, lrs=(1e-4, 1e-5),
    backbone_variant="s", refiner_variant="s",
    backbone_width_mult=1.0, backbone_depth_mult=1.0,
    refiner_width_mult=1.0,  refiner_depth_mult=1.0,
    backbone_stem_channels=None, refiner_stem_channels=None,
    backbone_drop_path=0.0, refiner_drop_path=0.0,
    backbone_se_ratio=0.25,  refiner_se_ratio=0.25,
    refiner_feature_pool="avg", refiner_shared_hidden_scale=0.5,
    refiner_time_embed_dim=None,
    huber_beta=0.002,  # pour la Huber
):
    pl.seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_windows = sys.platform == "win32"
    num_workers = 0 if is_windows else 4

    poly_freq_CH4 = [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]
    transitions_ch4_str = """6;1;3085.861015;1.013E-19;0.06;0.078;219.9411;0.73;-0.00712;0.0;0.0221;0.96;0.584;1.12
6;1;3085.832038;1.693E-19;0.0597;0.078;219.9451;0.73;-0.00712;0.0;0.0222;0.91;0.173;1.11
6;1;3085.893769;1.011E-19;0.0602;0.078;219.9366;0.73;-0.00711;0.0;0.0184;1.14;-0.516;1.37
6;1;3086.030985;1.659E-19;0.0595;0.078;219.9197;0.73;-0.00711;0.0;0.0193;1.17;-0.204;0.97
6;1;3086.071879;1.000E-19;0.0585;0.078;219.9149;0.73;-0.00703;0.0;0.0232;1.09;-0.0689;0.82
6;1;3086.085994;6.671E-20;0.055;0.078;219.9133;0.70;-0.00610;0.0;0.0300;0.54;0.00;0.0"""
    transitions_dict = {'CH4': parse_csv_transitions(transitions_ch4_str)}

    # Ranges par défaut
    # default_val = {
    #     'sig0': (3085.43, 3085.46),
    #     'dsig': (0.001521, 0.00154),
    #     'mf_CH4': (2e-6, 100e-6),
    #     'baseline0': (1, 1.00000001),
    #     'baseline1': (-0.0004, -0.0003),
    #     'baseline2': (-4.0565E-08, -3.07117E-08),
    #     'P': (400, 800),
    #     'T': (273.15 + 0, 273.15 + 50),
    # }
    # expand_factors = {"_default": 1.0, 'sig0': 5.0, 'dsig': 7.0, 'mf_CH4': 2.0,
    #                   "baseline0": 1, "baseline1": 10.0, "baseline2": 15.0, "P": 2.0, "T": 2.0}
    
    default_val = {
        'sig0': (3085.43, 3085.46),
        'dsig': (0.001521, 0.00154),
        'mf_CH4': (2e-6, 50e-6),
        'baseline0': (0.99, 1.01),
        'baseline1': (-0.0004, -0.0003),
        'baseline2': (-4.0565E-08, -3.07117E-08),
        'P': (400, 600),
        'T': (273.15 + 30, 273.15 + 40),
    }
    expand_factors = {"_default": 1.0, 'sig0': 5.0, 'dsig': 7.0, 'mf_CH4': 2.0,
                      "baseline0": 1, "baseline1": 3.0, "baseline2": 8.0, "P": 2.0, "T": 2.0}
    

    default_train = map_ranges(default_val, expand_interval, per_param=expand_factors)

    # Plancher log
    lo, hi = default_train['mf_CH4']; default_train['mf_CH4'] = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR*10))
    lo, hi = default_val['mf_CH4'];   default_val['mf_CH4']   = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR*10))

    global NORM_PARAMS
    VAL_RANGES = val_ranges or default_val
    TRAIN_RANGES = train_ranges or default_train
    assert_subset(VAL_RANGES, TRAIN_RANGES, "VAL", "TRAIN")
    NORM_PARAMS = TRAIN_RANGES

    NOISE_TRAIN = noise_train or dict(
        std_add_range=(0, 1e-3), std_mult_range=(0, 1e-3),
        p_drift=0.1, drift_sigma_range=(10.0, 120.0), drift_amp_range=(0.004, 0.05),
        p_fringes=0.2, n_fringes_range=(1, 2), fringe_freq_range=(0.3, 50.0), fringe_amp_range=(0.001, 0.015),
        p_spikes=0.1, spikes_count_range=(1, 6), spike_amp_range=(0.002, 1), spike_width_range=(1.0, 200.0),
        clip=(0.0, 1.1),
    )
    NOISE_VAL = noise_val or dict(
        std_add_range=(0, 1e-5), std_mult_range=(0, 1e-5),
        p_drift=0, drift_sigma_range=(20.0, 120.0), drift_amp_range=(0.0, 0.01),
        p_fringes=0, n_fringes_range=(1, 2), fringe_freq_range=(0.5, 10.0), fringe_amp_range=(0.0, 0.004),
        p_spikes=0.0, spikes_count_range=(1, 2), spike_amp_range=(0.0, 0.01), spike_width_range=(1.0, 3.0),
        clip=(0.0, 1.1),
    )

    dataset_train = SpectraDataset(n_samples=n_train, num_points=n_points,
                                   poly_freq_CH4=poly_freq_CH4, transitions_dict=transitions_dict,
                                   sample_ranges=TRAIN_RANGES, strict_check=True,
                                   with_noise=True, noise_profile=NOISE_TRAIN, freeze_noise=False)
    dataset_val = SpectraDataset(n_samples=n_val, num_points=n_points,
                                 poly_freq_CH4=poly_freq_CH4, transitions_dict=transitions_dict,
                                 sample_ranges=VAL_RANGES, strict_check=True,
                                 with_noise=True, noise_profile=NOISE_VAL, freeze_noise=True)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device == 'cuda'))

    # baseline0 n'est PAS prédit (normalisation via max LOWESS)
    predict_list = predict_list or ["sig0", "dsig", "mf_CH4", "P", "T", "baseline1", "baseline2"]
    film_list = film_list or []

    model = PhysicallyInformedAE(
        n_points=n_points, param_names=PARAMS, poly_freq_CH4=poly_freq_CH4, transitions_dict=transitions_dict,
        lr=lrs[0], alpha_param=0.3, alpha_phys=0.7, head_mode="multi",
        predict_params=predict_list, film_params=film_list,
        refine_steps=1, refine_delta_scale=0.1, refine_target="noisy",
        refine_warmup_epochs=30, freeze_base_epochs=20,
        base_lr=lrs[0], refiner_lr=lrs[1],
        recon_max1=True,
        corr_mode="none",
        corr_savgol_win=15,
        corr_savgol_poly=3,
        huber_beta=huber_beta,
        # ==== NOUVEAUX hyperparamètres vers backbone/refiner ====
        backbone_variant=backbone_variant,
        refiner_variant=refiner_variant,
        backbone_width_mult=backbone_width_mult,
        backbone_depth_mult=backbone_depth_mult,
        refiner_width_mult=refiner_width_mult,
        refiner_depth_mult=refiner_depth_mult,
        backbone_stem_channels=backbone_stem_channels,
        refiner_stem_channels=refiner_stem_channels,
        backbone_drop_path=backbone_drop_path,
        refiner_drop_path=refiner_drop_path,
        backbone_se_ratio=backbone_se_ratio,
        refiner_se_ratio=refiner_se_ratio,
        refiner_feature_pool=refiner_feature_pool,
        refiner_shared_hidden_scale=refiner_shared_hidden_scale,
        refiner_time_embed_dim=refiner_time_embed_dim,
        ranges_train=TRAIN_RANGES,
        ranges_val=VAL_RANGES,
        noise_train=NOISE_TRAIN,
        noise_val=NOISE_VAL,
    )

    model.hparams.optimizer = "lion"        # "adamw" | "lion" | "radam" | "adabelief"
    model.hparams.betas = (0.9, 0.99)       # Optionnel, seulement pour Lion
    model.weight_decay = 1e-4    

    def _serializable_ranges(d):
        return {k: [float(d[k][0]), float(d[k][1])] for k in d}

    extra_hparams = {
        "ranges_train": _serializable_ranges(TRAIN_RANGES),
        "ranges_val":   _serializable_ranges(VAL_RANGES),
        "noise_train":  _to_serializable(NOISE_TRAIN),
        "noise_val":    _to_serializable(NOISE_VAL),
    }

    try:
        model.save_hyperparameters(extra_hparams)
    except Exception:
        if hasattr(model, "hparams"):
            for k, v in extra_hparams.items():
                setattr(model.hparams, k, v)


    return model, train_loader, val_loader

# ------------- Utilitaires HPC -------------
def get_master_addr_and_port(default_port=12910):
    # MASTER_ADDR pour SLURM : 1er hostname de la nodelist
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
        if nodelist:
            # méthode robuste: scontrol peut ne pas être dispo à l’intérieur; fallback simple
            try:
                import subprocess, shlex
                cmd = f"scontrol show hostnames {shlex.quote(nodelist)} | head -n 1"
                master_addr = subprocess.check_output(cmd, shell=True).decode().strip()
            except Exception:
                # dernier recours: hostname courant
                master_addr = socket.gethostname()
        else:
            master_addr = socket.gethostname()
    master_port = int(os.environ.get("MASTER_PORT", default_port))
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    return master_addr, master_port

def choose_precision():
    if torch.cuda.is_available():
        # bf16 si supporté (A100/H100/Grace-Hopper…)
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        # sinon fp16 mixte
        if torch.cuda.get_device_capability(0)[0] >= 7:  # Volta+
            return "16-mixed"
    return "32-true"

from torch.utils.data import DataLoader, DistributedSampler

def _with_sampler(loader, shuffle_default):
    if loader is None: return None
    if isinstance(getattr(loader, "sampler", None), DistributedSampler):
        return loader
    ds = loader.dataset
    sampler = DistributedSampler(ds, shuffle=shuffle_default)
    return DataLoader(
        ds,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
        sampler=sampler,       
        shuffle=False,
    )

def ensure_distributed_samplers(train_loader, val_loader):
    # On ne se base QUE sur l'init réelle de torch.distributed (pas les env SLURM).
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return train_loader, val_loader
    return _with_sampler(train_loader, True), _with_sampler(val_loader, False)

def trainer_common_kwargs():
    import pytorch_lightning as pl
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # ► Trials mono-GPU seulement (1 tâche = 1 GPU)
    devices = 1
    num_nodes = 1

    precision = choose_precision()
    default_dir = os.environ.get("SLURM_JOB_ID", "runs_local")
    default_root_dir = f"./lightning_logs/{default_dir}"

    # ► Pas de DDP pour les trials mono-GPU
    strategy = "auto"

    return dict(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        default_root_dir=default_root_dir,
        log_every_n_steps=50,
        enable_progress_bar=False,
        deterministic=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy=strategy,
    )


def on_rank_zero():
    # Simple helper: True uniquement sur le rank 0 global
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def _ensure_stage_structure(stage_dir: Path):
    for sub in ("trials", "checkpoints", "logs", "figs", "eval", "retrain"):
        (stage_dir / sub).mkdir(parents=True, exist_ok=True)


def make_run_dir(base="runs"):
    existing = os.environ.get("PHYS_AE_RUN_DIR")
    if existing:
        Path(existing).mkdir(parents=True, exist_ok=True)
        _ensure_stage_structure(Path(existing) / "A")
        _ensure_stage_structure(Path(existing) / "B")
        (Path(existing) / "finetune" / "checkpoints").mkdir(parents=True, exist_ok=True)
        (Path(existing) / "finetune" / "logs").mkdir(parents=True, exist_ok=True)
        (Path(existing) / "finetune" / "figs").mkdir(parents=True, exist_ok=True)
        return existing

    job = os.environ.get("SLURM_JOB_ID")
    if job:
        run_dir = Path(base) / f"study_{job}"
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(base) / f"study_local_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for stage in ("A", "B"):
        _ensure_stage_structure(run_dir / stage)

    finetune_dir = run_dir / "finetune"
    for sub in ("checkpoints", "logs", "figs"):
        (finetune_dir / sub).mkdir(parents=True, exist_ok=True)

    os.environ["PHYS_AE_RUN_DIR"] = str(run_dir)
    return str(run_dir)


def get_stage_dir(run_dir: Path, stage: str) -> Path:
    stage_dir = Path(run_dir) / stage.upper()
    _ensure_stage_structure(stage_dir)
    return stage_dir

import yaml
import numpy as np
import torch

# --- A) rendre sérialisable: tuples -> listes, numpy/torch -> python ---
def _to_serializable(x):
    if isinstance(x, tuple):
        return [_to_serializable(v) for v in x]
    if isinstance(x, list):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (np.generic,)):   # numpy scalars
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.ndim else x.item()
    return x

class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def save_config(run_dir_path, **cfg):
    path = os.path.join(run_dir_path, "config.yaml")
    clean = _to_serializable(cfg)  # plus de !!python/tuple / objets non serializables
    with open(path, "w") as f:
        yaml.dump(
            clean, f,
            Dumper=_NoAliasDumper,   # plus de &id/*id
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False
        )
    return path


"""
Optuna tuning pour les étapes A (backbone+heads) et B1 (refiner) de ton pipeline.

Pré-requis: place ton gros fichier (celui que tu as collé) sous le nom `pipeline.py`
dans le même dossier que ce script, OU adapte `PIPELINE_MODULE` ci‑dessous.

Ce script:
  1) optimise Stage A → écrit RUN_DIR/checkpoints/A_opt.ckpt
  2) optimise Stage B1 en repartant du meilleur A → écrit RUN_DIR/checkpoints/B_opt.ckpt
  3) (optionnel) lance un court B2 (fine‑tune global) en repartant de B1 best → RUN_DIR/checkpoints/B2_final.ckpt

Usage minimal:
  python optuna_tuner.py --trials-a 20 --trials-b 20 --epochs-a 50 --epochs-b 30

Tu peux relancer: les studies sont stockées en SQLite dans RUN_DIR.
"""
import os
import math
import shutil
import argparse
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages.journal import JournalFileBackend
from optuna.storages import JournalStorage


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import json, os, shutil, time
from pathlib import Path

# ---- callback à mettre près de ton objective() ----
import torch
import optuna
import pytorch_lightning as pl

class LossHistory(pl.callbacks.Callback):
    def __init__(self, trial: optuna.trial.Trial):
        self.trial = trial
        self.train_hist = []
        self.val_hist = []

    @staticmethod
    def _to_float(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            x = x.item() if x.ndim == 0 else float(x.mean().item())
        return float(x)

    def on_train_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics
        # suivant ta config, la clé peut être train_loss_epoch ou train_loss
        v = cm.get("train_loss_epoch", cm.get("train_loss", None))
        self.train_hist.append(self._to_float(v))

    def on_validation_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics
        # idem pour la val : val_loss_epoch ou val_loss
        v = cm.get("val_loss_epoch", cm.get("val_loss", None))
        v = self._to_float(v)
        self.val_hist.append(v)

        # (optionnel) reporter la val_loss à Optuna pour pruning/traçage
        step = len(self.val_hist) - 1
        if v is not None:
            self.trial.report(v, step)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    def on_fit_end(self, trainer, pl_module):
        # stocke les historiques dans le trial → récupérable après l’étude
        self.trial.set_user_attr("train_loss_history", self.train_hist)
        self.trial.set_user_attr("val_loss_history", self.val_hist)


class OptunaCSVLogger:
    """Enregistre chaque trial Optuna dans un CSV et maintient le top 5."""

    def __init__(self, stage_dir: Path, stage: str, top_k: int = 5):
        self.stage_dir = Path(stage_dir)
        self.stage = stage.upper()
        self.top_k = top_k
        self.csv_path = self.stage_dir / "trials.csv"
        self.top_path = self.stage_dir / "top5.json"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        self._append_trial(trial)
        self._update_top(study)

    def _append_trial(self, trial: optuna.trial.FrozenTrial):
        record = {
            "trial": trial.number,
            "timestamp": datetime.utcnow().isoformat(),
            "state": trial.state.name if trial.state else "UNKNOWN",
            "value": "" if trial.value is None else float(trial.value),
            "duration_s": trial.duration.total_seconds() if trial.duration else "",
            "params": json.dumps(trial.params, sort_keys=True, ensure_ascii=False),
        }

        file_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def _update_top(self, study: optuna.study.Study):
        try:
            trials = study.get_trials(deepcopy=False, states=None)
        except Exception:
            trials = study.trials

        completed = [
            t for t in trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if not completed:
            return

        direction = getattr(study, "direction", None)
        if direction is None:
            directions = getattr(study, "directions", None)
            direction = directions[0] if directions else optuna.study.StudyDirection.MINIMIZE

        reverse = direction == optuna.study.StudyDirection.MAXIMIZE
        completed.sort(key=lambda t: t.value, reverse=reverse)
        top = completed[: self.top_k]

        ckpt_attr = "best_ckpt_A" if self.stage.startswith("A") else "best_ckpt_B1"
        payload = [
            {
                "rank": rank,
                "trial": t.number,
                "value": float(t.value),
                "params": dict(t.params),
                "checkpoint": t.user_attrs.get(ckpt_attr, ""),
            }
            for rank, t in enumerate(top, start=1)
        ]

        self.top_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def get_worker_info():
    rank = int(os.environ.get("SLURM_PROCID") or os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or 0)
    world = int(os.environ.get("SLURM_NTASKS") or os.environ.get("WORLD_SIZE") or 1)
    return rank, world

def wait_for_file(path, check_every=5, timeout=24*3600):
    t0 = time.time()
    while not os.path.isfile(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Timeout en attendant {path}")
        time.sleep(check_every)
    return path

from typing import Tuple, Union
from pathlib import Path
import json, os, shutil, time

def _atomic_update_best_ckpt(
    src_ckpt: Union[str, Path],
    new_score: float,
    dest_ckpt: Union[str, Path],
    meta_path: Union[str, Path],
    direction: str = "min",
) -> Tuple[bool, str]:
    """
    Copie src_ckpt -> dest_ckpt si new_score est meilleur que 'score' dans meta_path.
    direction: 'min' (par défaut) ou 'max'.
    Retourne (is_new_best, str(dest_ckpt)).
    """
    if not src_ckpt:
        return False, str(dest_ckpt)

    dest_ckpt = Path(dest_ckpt)
    meta_path = Path(meta_path)
    dest_ckpt.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # petit verrou (optionnel) pour éviter les races entre ranks
    lock_fh = None
    lock_path = meta_path.with_suffix(meta_path.suffix + ".lock")
    try:
        try:
            import fcntl
            lock_fh = open(lock_path, "w")
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except Exception:
            lock_fh = None

        cur = None
        if meta_path.exists():
            try:
                cur = float(json.loads(meta_path.read_text()).get("score", None))
            except Exception:
                cur = None

        if direction not in ("min", "max"):
            raise ValueError("direction must be 'min' or 'max'")

        is_better = (cur is None) or (
            (direction == "min" and new_score < cur) or
            (direction == "max" and new_score > cur)
        )

        if is_better:
            tmp = dest_ckpt.with_suffix(dest_ckpt.suffix + f".tmp.{os.getpid()}")
            shutil.copy2(str(src_ckpt), tmp)
            os.replace(tmp, dest_ckpt)
            meta_path.write_text(json.dumps({
                "score": float(new_score),
                "path": str(dest_ckpt),
                "time": time.time()
            }))
            return True, str(dest_ckpt)

        return False, str(dest_ckpt) if dest_ckpt.exists() else ""

    finally:
        if lock_fh:
            try:
                import fcntl
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
                lock_fh.close()
            except Exception:
                pass


def _cleanup_trial_checkpoint(path: Union[str, Path]):
    if not path:
        return

    p = Path(path)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

    parent = p.parent
    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        pass


def _cleanup_trial_dir(root: Union[str, Path]):
    if not root:
        return

    root = Path(root)
    try:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass

    parent = root.parent
    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        pass


def _register_trial_best(stage_dir: Path, stage_name: str, best_path: Optional[str], best_score: float,
                         direction: str = "min") -> str:
    if not best_path:
        return ""

    dest_ckpt = stage_dir / "checkpoints" / f"{stage_name}_optuna_best.ckpt"
    meta_path = stage_dir / "checkpoints" / f"{stage_name}_optuna_best.json"
    is_new_best, final_ckpt = _atomic_update_best_ckpt(best_path, best_score, dest_ckpt, meta_path, direction=direction)

    _cleanup_trial_checkpoint(best_path)

    if final_ckpt and os.path.isfile(final_ckpt):
        return final_ckpt

    return ""


def _trial_dirs(run_dir: Path, stage: str, trial_number: int) -> dict[str, Path]:
    stage_dir = get_stage_dir(run_dir, stage)
    root = stage_dir / "trials" / f"trial_{trial_number:04d}"
    ck   = root / "ckpts"
    fig  = root / "figs"
    for p in (root, ck, fig):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "ckpts": ck, "figs": fig}


def _common_callbacks(stage: str, val_loader, fig_dir: Path, monitor: str = "val_loss",
                      patience: int = 15, ckpt_dir: Optional[Path] = None) -> list:
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckp = ModelCheckpoint(
        monitor=monitor,
        mode="min",
        save_top_k=1,
        filename=f"best-{stage}",
        save_last=False,
        auto_insert_metric_name=False,
        dirpath=str(ckpt_dir) if ckpt_dir is not None else None,
    )
    early = EarlyStopping(monitor=monitor, mode="min", patience=patience)
    plot = PlotAndMetricsCallback(val_loader, PARAMS, num_examples=1, save_dir=str(fig_dir), stage_tag=f"stage_{stage}")
    return [ckp, early, plot, UpdateEpochInDataset(), AdvanceDistributedSamplerEpoch()]


def _score_from_callbacks(callbacks: list, fallback: float | None = None) -> tuple[float, str | None]:
    ckps = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
    if ckps and ckps[0].best_model_path:
        score = ckps[0].best_model_score
        if score is not None:
            try:
                return float(score.cpu().item()), ckps[0].best_model_path
            except Exception:
                return float(score), ckps[0].best_model_path
    # fallback (ex: aucune amélioration)
    return (float("inf") if fallback is None else float(fallback), None)


def _copy_ckpt(src_path: str | Path, dst_path: str | Path):
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(src_path), str(dst_path))


def _cleanup_artifacts(paths: Iterable[Path | str]):
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if path.is_file():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


# ===================== OBJECTIVE: STAGE A =====================

def objective_stage_A(trial: optuna.Trial, *, run_dir: Path, epochs: int, seed: int,
                      n_train_samples: int = 100_000, n_val_samples: int = 500) -> float:
    # --- Search space ---
    base_lr = trial.suggest_float("base_lr", 1e-5, 5e-4, log=True)
    backbone_variant = trial.suggest_categorical("backbone_variant", ["s"])  # compacité vs capacité
    backbone_width_mult = trial.suggest_float("backbone_width_mult", 0.8, 1.3)
    backbone_depth_mult = trial.suggest_float("backbone_depth_mult", 0.85, 1.35)
    backbone_drop_path  = trial.suggest_float("backbone_drop_path", 0.0, 0.2)
    backbone_se_ratio   = trial.suggest_float("backbone_se_ratio", 0.15, 0.35)
    huber_beta          = trial.suggest_float("huber_beta", 5e-4, 5e-3, log=True)
    batch_size          = trial.suggest_categorical("batch_size", [8, 12, 16, 24, 32])

    # Dossiers dédiés au trial
    stage_dir = get_stage_dir(run_dir, "A")
    dirs = _trial_dirs(run_dir, stage="A", trial_number=trial.number)

    # Construire data + modèle avec HPs trial
    model, train_loader, val_loader = build_data_and_model(
        seed=seed,
        n_train=n_train_samples,
        n_val=n_val_samples,
        batch_size=batch_size,
        huber_beta=huber_beta,
        backbone_variant=backbone_variant,
        backbone_width_mult=backbone_width_mult,
        backbone_depth_mult=backbone_depth_mult,
        backbone_drop_path=backbone_drop_path,
        backbone_se_ratio=backbone_se_ratio,
    )

    # Trainer kwargs (on force 1 device pour éviter des essais multi-GPU par trial)
    tkw = trainer_common_kwargs()
    tkw["devices"] = 1
    tkw["default_root_dir"] = str(dirs["root"] / "logs")

    # Callbacks
    callbacks = _common_callbacks("A", val_loader, fig_dir=dirs["figs"], patience=12, ckpt_dir=dirs["ckpts"])
    callbacks = [LossHistory(trial), *callbacks] 
    # Lancement stage A
    train_stage_A(
        model, train_loader, val_loader,
        epochs=epochs,
        base_lr=base_lr,
        callbacks=callbacks,
        # ckpt_out facultatif ici (ModelCheckpoint gère l'early/best)
        **tkw,
    )

    # Récup meilleure val_loss + ckpt
    # Récup meilleure val_loss + ckpt
    best_score, best_path = _score_from_callbacks(callbacks)
    global_ckpt = _register_trial_best(stage_dir, "A", best_path, best_score, direction="min")

    trial.set_user_attr("best_ckpt_A", global_ckpt)
    trial.set_user_attr("best_val_A", best_score)

    _cleanup_trial_dir(dirs.get("root"))
    return best_score




# ===================== OBJECTIVE: STAGE B1 =====================

def objective_stage_B1(trial: optuna.Trial, *, run_dir: Path, epochs: int, seed: int, ckpt_A: str,
                       n_train_samples: int = 100_000, n_val_samples: int = 500) -> float:
    # --- Search space (refiner-centric) ---
    refiner_lr = trial.suggest_float("refiner_lr", 1e-6, 3e-4, log=True)
    refine_steps = trial.suggest_int("refine_steps", 1, 1)
    delta_scale  = trial.suggest_float("delta_scale", 0.02, 0.15)
    refiner_variant = trial.suggest_categorical("refiner_variant", ["s"])
    refiner_width_mult = trial.suggest_float("refiner_width_mult", 0.85, 1.35)
    refiner_depth_mult = trial.suggest_float("refiner_depth_mult", 0.85, 1.35)
    refiner_feature_pool = trial.suggest_categorical("refiner_feature_pool", ["avg"])
    refiner_shared_hidden_scale = trial.suggest_float("refiner_shared_hidden_scale", 0.35, 0.8)
    refiner_time_embed_dim = trial.suggest_categorical("refiner_time_embed_dim", [16, 24, 32, 48, 64])

    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24, 32])

    stage_dir = get_stage_dir(run_dir, "B")
    dirs = _trial_dirs(run_dir, stage="B", trial_number=trial.number)

    # IMPORTANT: reconstruit le modèle avec les HP refiner (le backbone peut rester celui du meilleur A)
    model, train_loader, val_loader = build_data_and_model(
        seed=seed,
        n_train=n_train_samples,
        n_val=n_val_samples,
        batch_size=batch_size,
        refiner_variant=refiner_variant,
        refiner_width_mult=refiner_width_mult,
        refiner_depth_mult=refiner_depth_mult,
        refiner_feature_pool=refiner_feature_pool,
        refiner_shared_hidden_scale=refiner_shared_hidden_scale,
        refiner_time_embed_dim=refiner_time_embed_dim,
    )

    # Trainer kwargs
    tkw = trainer_common_kwargs()
    tkw["devices"] = 1
    tkw["default_root_dir"] = str(dirs["root"] / "logs")

    callbacks = _common_callbacks("B1", val_loader, fig_dir=dirs["figs"], patience=10, ckpt_dir=dirs["ckpts"])
    callbacks = [LossHistory(trial), *callbacks] 


    # Lancement B1 en repartant de A
    train_stage_B1(
        model, train_loader, val_loader,
        epochs=epochs,
        refiner_lr=refiner_lr,
        refine_steps=refine_steps,
        delta_scale=delta_scale,
        callbacks=callbacks,
        ckpt_in=ckpt_A,
        **tkw,
    )

    # Récup meilleure val_loss + ckpt
    best_score, best_path = _score_from_callbacks(callbacks)
    global_ckpt = _register_trial_best(stage_dir, "B1", best_path, best_score, direction="min")

    trial.set_user_attr("best_ckpt_B1", global_ckpt)
    trial.set_user_attr("best_val_B1", best_score)

    _cleanup_trial_dir(dirs.get("root"))
    return best_score



# ===================== RUNNERS =====================

def run_optuna_stage_A(n_trials: int, epochs: int, seed: int,
                       n_train_samples: int, n_val_samples: int) -> tuple[str, float, dict]:
    run_dir = Path(make_run_dir())
    stage_dir = get_stage_dir(run_dir, "A")
    optuna_dir = stage_dir / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    journal_path = optuna_dir / "journal.log"
    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    study = optuna.create_study(
        study_name="stage_A_opt",
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=0),
        load_if_exists=True,
    )

    def _obj(trial: optuna.Trial) -> float:
        return objective_stage_A(
            trial,
            run_dir=run_dir,
            epochs=epochs,
            seed=seed,
            n_train_samples=n_train_samples,
            n_val_samples=n_val_samples,
        )

    csv_logger = OptunaCSVLogger(stage_dir, stage="A")
    study.optimize(_obj, n_trials=n_trials, callbacks=[csv_logger], show_progress_bar=True)

    # ► meilleur global (tous workers) + artefacts
    best = study.best_trial
    best_ckpt_src = best.user_attrs.get("best_ckpt_A", "")
    if not best_ckpt_src or not os.path.isfile(best_ckpt_src):
        raise RuntimeError("Aucune checkpoint valide trouvée après l'optimisation de A.")

    dest_ckpt = stage_dir / "checkpoints" / "A_opt.ckpt"
    meta_path = stage_dir / "checkpoints" / "A_best.json"
    _atomic_update_best_ckpt(best_ckpt_src, float(best.value), dest_ckpt, meta_path, direction="min")

    # ► JSON des meilleurs hyperparams d'A
    if on_rank_zero():
        _save_best_params_json("A", best, str(dest_ckpt), stage_dir)

    print(f"✓ A_opt prêt: {dest_ckpt}")
    return str(dest_ckpt), float(best.value), dict(best.params)

def run_optuna_stage_B1(n_trials: int, epochs: int, seed: int, ckpt_A_path: str,
                        n_train_samples: int, n_val_samples: int) -> tuple[str, float, dict]:
    run_dir = Path(make_run_dir())
    stage_dir = get_stage_dir(run_dir, "B")
    optuna_dir = stage_dir / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    journal_path = optuna_dir / "journal.log"
    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    study = optuna.create_study(
        study_name="stage_B1_opt",
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=0),
        load_if_exists=True,
    )

    def _obj(trial: optuna.Trial) -> float:
        return objective_stage_B1(
            trial,
            run_dir=run_dir,
            epochs=epochs,
            seed=seed,
            ckpt_A=ckpt_A_path,
            n_train_samples=n_train_samples,
            n_val_samples=n_val_samples,
        )

    csv_logger = OptunaCSVLogger(stage_dir, stage="B")
    study.optimize(_obj, n_trials=n_trials, callbacks=[csv_logger], show_progress_bar=True)

    best = study.best_trial
    best_ckpt_src = best.user_attrs.get("best_ckpt_B1", "")
    if not best_ckpt_src or not os.path.isfile(best_ckpt_src):
        raise RuntimeError("Aucune checkpoint valide trouvée après l'optimisation de B1.")

    dest_ckpt = stage_dir / "checkpoints" / "B_opt.ckpt"
    meta_path = stage_dir / "checkpoints" / "B_best.json"
    _atomic_update_best_ckpt(best_ckpt_src, float(best.value), dest_ckpt, meta_path, direction="min")

    # ► JSON des meilleurs hyperparams de B (B1)
    if on_rank_zero():
        _save_best_params_json("B", best, str(dest_ckpt), stage_dir)

    print(f"✓ B_opt prêt: {dest_ckpt}")
    return str(dest_ckpt), float(best.value), dict(best.params)


def _best_source_from_callbacks(callbacks: list, fallback_ckpt: Optional[Path]) -> tuple[str, float]:
    score, path = _score_from_callbacks(callbacks)
    if path and os.path.isfile(path):
        return path, float(score)
    if fallback_ckpt is not None and fallback_ckpt.exists():
        return str(fallback_ckpt), float(score)
    return "", float(score)


def retrain_stage_A(best_params: dict, *, epochs: int, seed: int, n_train: int = 1_000_000) -> tuple[str, float]:
    run_dir = Path(make_run_dir())
    stage_dir = get_stage_dir(run_dir, "A")
    retrain_dir = stage_dir / "retrain"
    logs_dir = retrain_dir / "logs"
    figs_dir = retrain_dir / "figs"
    ckpts_dir = retrain_dir / "ckpts"
    for p in (logs_dir, figs_dir, ckpts_dir):
        p.mkdir(parents=True, exist_ok=True)

    batch_size = int(best_params.get("batch_size", 16))
    model, train_loader, val_loader = build_data_and_model(
        seed=seed,
        n_train=n_train,
        batch_size=batch_size,
        huber_beta=float(best_params.get("huber_beta", 0.002)),
        backbone_variant=best_params.get("backbone_variant", "s"),
        backbone_width_mult=float(best_params.get("backbone_width_mult", 1.0)),
        backbone_depth_mult=float(best_params.get("backbone_depth_mult", 1.0)),
        backbone_drop_path=float(best_params.get("backbone_drop_path", 0.0)),
        backbone_se_ratio=float(best_params.get("backbone_se_ratio", 0.25)),
    )

    tkw = trainer_common_kwargs()
    tkw["default_root_dir"] = str(logs_dir)

    callbacks = _common_callbacks("A_retrain", val_loader, fig_dir=figs_dir, patience=15, ckpt_dir=ckpts_dir)
    ckpt_last = retrain_dir / "A_fine_tuned_last.ckpt"

    train_stage_A(
        model, train_loader, val_loader,
        epochs=epochs,
        base_lr=float(best_params.get("base_lr", 2e-4)),
        callbacks=callbacks,
        ckpt_out=str(ckpt_last),
        **tkw,
    )

    best_src, best_score = _best_source_from_callbacks(callbacks, ckpt_last)

    dest_ckpt = stage_dir / "checkpoints" / "A_fine_tuned.ckpt"
    meta_path = stage_dir / "checkpoints" / "A_fine_tuned.json"
    if best_src:
        _atomic_update_best_ckpt(best_src, best_score, dest_ckpt, meta_path, direction="min")

    _cleanup_artifacts([ckpt_last, ckpts_dir])

    _dump_json(stage_dir / "checkpoints" / "A_fine_tuned_params.json", {
        "params": dict(best_params),
        "epochs": epochs,
        "n_train": n_train,
        "batch_size": batch_size,
        "source_checkpoint": best_src,
    })

    return str(dest_ckpt), best_score


def retrain_stage_B(best_params: dict, stage_a_params: dict, stage_a_ckpt: str, *, epochs: int, seed: int,
                    n_train: int = 1_000_000) -> tuple[str, float]:
    run_dir = Path(make_run_dir())
    stage_dir = get_stage_dir(run_dir, "B")
    retrain_dir = stage_dir / "retrain"
    logs_dir = retrain_dir / "logs"
    figs_dir = retrain_dir / "figs"
    ckpts_dir = retrain_dir / "ckpts"
    for p in (logs_dir, figs_dir, ckpts_dir):
        p.mkdir(parents=True, exist_ok=True)

    batch_size = int(best_params.get("batch_size", stage_a_params.get("batch_size", 16)))
    model, train_loader, val_loader = build_data_and_model(
        seed=seed,
        n_train=n_train,
        batch_size=batch_size,
        huber_beta=float(stage_a_params.get("huber_beta", 0.002)),
        backbone_variant=stage_a_params.get("backbone_variant", "s"),
        backbone_width_mult=float(stage_a_params.get("backbone_width_mult", 1.0)),
        backbone_depth_mult=float(stage_a_params.get("backbone_depth_mult", 1.0)),
        backbone_drop_path=float(stage_a_params.get("backbone_drop_path", 0.0)),
        backbone_se_ratio=float(stage_a_params.get("backbone_se_ratio", 0.25)),
        refiner_variant=best_params.get("refiner_variant", "s"),
        refiner_width_mult=float(best_params.get("refiner_width_mult", 1.0)),
        refiner_depth_mult=float(best_params.get("refiner_depth_mult", 1.0)),
        refiner_feature_pool=best_params.get("refiner_feature_pool", "avg"),
        refiner_shared_hidden_scale=float(best_params.get("refiner_shared_hidden_scale", 0.5)),
        refiner_time_embed_dim=best_params.get("refiner_time_embed_dim", None),
    )

    tkw = trainer_common_kwargs()
    tkw["default_root_dir"] = str(logs_dir)

    callbacks = _common_callbacks("B_retrain", val_loader, fig_dir=figs_dir, patience=12, ckpt_dir=ckpts_dir)
    ckpt_last = retrain_dir / "B_fine_tuned_last.ckpt"

    train_stage_B1(
        model, train_loader, val_loader,
        epochs=epochs,
        refiner_lr=float(best_params.get("refiner_lr", 1e-5)),
        refine_steps=int(best_params.get("refine_steps", 1)),
        delta_scale=float(best_params.get("delta_scale", 0.1)),
        callbacks=callbacks,
        ckpt_in=stage_a_ckpt,
        ckpt_out=str(ckpt_last),
        **tkw,
    )

    best_src, best_score = _best_source_from_callbacks(callbacks, ckpt_last)

    dest_ckpt = stage_dir / "checkpoints" / "B_fine_tuned.ckpt"
    meta_path = stage_dir / "checkpoints" / "B_fine_tuned.json"
    if best_src:
        _atomic_update_best_ckpt(best_src, best_score, dest_ckpt, meta_path, direction="min")

    _cleanup_artifacts([ckpt_last, ckpts_dir])

    _dump_json(stage_dir / "checkpoints" / "B_fine_tuned_params.json", {
        "params": dict(best_params),
        "epochs": epochs,
        "n_train": n_train,
        "batch_size": batch_size,
        "stage_a_checkpoint": stage_a_ckpt,
        "source_checkpoint": best_src,
    })

    return str(dest_ckpt), best_score


def finetune_ensemble(ckpt_B_path: str, best_a_params: dict, best_b_params: dict, *, epochs: int, seed: int,
                      n_train: int = 1_000_000) -> tuple[str, float]:
    run_dir = Path(make_run_dir())
    finetune_dir = Path(run_dir) / "finetune"
    logs_dir = finetune_dir / "logs"
    figs_dir = finetune_dir / "figs"
    tmp_ckpt_dir = finetune_dir / "tmp_ckpts"
    for p in (logs_dir, figs_dir):
        p.mkdir(parents=True, exist_ok=True)

    batch_size = int(best_b_params.get("batch_size", best_a_params.get("batch_size", 16)))
    model, train_loader, val_loader = build_data_and_model(
        seed=seed,
        n_train=n_train,
        batch_size=batch_size,
        huber_beta=float(best_a_params.get("huber_beta", 0.002)),
        backbone_variant=best_a_params.get("backbone_variant", "s"),
        backbone_width_mult=float(best_a_params.get("backbone_width_mult", 1.0)),
        backbone_depth_mult=float(best_a_params.get("backbone_depth_mult", 1.0)),
        backbone_drop_path=float(best_a_params.get("backbone_drop_path", 0.0)),
        backbone_se_ratio=float(best_a_params.get("backbone_se_ratio", 0.25)),
        refiner_variant=best_b_params.get("refiner_variant", "s"),
        refiner_width_mult=float(best_b_params.get("refiner_width_mult", 1.0)),
        refiner_depth_mult=float(best_b_params.get("refiner_depth_mult", 1.0)),
        refiner_feature_pool=best_b_params.get("refiner_feature_pool", "avg"),
        refiner_shared_hidden_scale=float(best_b_params.get("refiner_shared_hidden_scale", 0.5)),
        refiner_time_embed_dim=best_b_params.get("refiner_time_embed_dim", None),
    )

    tkw = trainer_common_kwargs()
    tkw["default_root_dir"] = str(logs_dir)

    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = _common_callbacks("B2_finetune", val_loader, fig_dir=figs_dir, patience=10, ckpt_dir=tmp_ckpt_dir)
    ckpt_last = finetune_dir / "checkpoints" / "finetune_last.ckpt"

    train_stage_B2(
        model, train_loader, val_loader,
        epochs=epochs,
        base_lr=float(best_a_params.get("base_lr", 3e-5)),
        refiner_lr=float(best_b_params.get("refiner_lr", 3e-6)),
        refine_steps=int(best_b_params.get("refine_steps", 1)),
        delta_scale=float(best_b_params.get("delta_scale", 0.1)),
        callbacks=callbacks,
        ckpt_in=ckpt_B_path,
        ckpt_out=str(ckpt_last),
        **tkw,
    )

    best_src, best_score = _best_source_from_callbacks(callbacks, ckpt_last)

    dest_ckpt = finetune_dir / "checkpoints" / "ensemble_best.ckpt"
    meta_path = finetune_dir / "checkpoints" / "ensemble_best.json"
    if best_src:
        _atomic_update_best_ckpt(best_src, best_score, dest_ckpt, meta_path, direction="min")

    _dump_json(finetune_dir / "checkpoints" / "ensemble_finetune_params.json", {
        "stage_A_params": dict(best_a_params),
        "stage_B_params": dict(best_b_params),
        "epochs": epochs,
        "n_train": n_train,
        "batch_size": batch_size,
        "source_checkpoint": ckpt_B_path,
        "selected_checkpoint": best_src,
    })

    return str(dest_ckpt), best_score


def optional_finetune_B2(ckpt_B1_path: str, epochs: int = 20):
    """Fine-tune global court (B2) en repartant du meilleur B1."""
    run_dir = Path(make_run_dir())

    model, train_loader, val_loader = build_data_and_model()

    tkw = trainer_common_kwargs()
    tkw["devices"] = 1
    ft_dir = Path(run_dir) / "finetune"
    tkw["default_root_dir"] = str(ft_dir / "logs")

    tmp_ckpt_dir = ft_dir / "tmp_ckpts"
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_path = ft_dir / "checkpoints" / "B2_optional.ckpt"
    callbacks = _common_callbacks("B2_optional", val_loader, fig_dir=ft_dir / "figs", patience=8, ckpt_dir=tmp_ckpt_dir)

    train_stage_B2(
        model, train_loader, val_loader,
        epochs=epochs,
        base_lr=3e-5, refiner_lr=3e-6,
        refine_steps=1, delta_scale=0.08,
        callbacks=callbacks,
        ckpt_in=ckpt_B1_path,
        ckpt_out=str(out_path),
        **tkw,
    )

    best_score, best_path = _score_from_callbacks(callbacks)
    print(f"\n✓ B2 terminé — best val_loss={best_score:.6g}; ckpt={best_path or out_path}")

if __name__ == "__main__":
    import argparse, math, json, os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Optuna tuner pour stages A et B1")
    parser.add_argument("--trials-a", type=int, default=200, help="Nombre d'essais Optuna pour A")
    parser.add_argument("--trials-b", type=int, default=200, help="Nombre d'essais Optuna pour B1")
    parser.add_argument("--epochs-a", type=int, default=50, help="Epochs par essai pour A")
    parser.add_argument("--epochs-b", type=int, default=50, help="Epochs par essai pour B1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=100_000,
                        help="Nombre d'échantillons synthétiques pour chaque essai (train)")
    parser.add_argument("--val-samples", type=int, default=500,
                        help="Nombre d'échantillons synthétiques pour la validation des essais")
    parser.add_argument("--retrain-samples", type=int, default=1_000_000,
                        help="Nombre d'échantillons synthétiques pour les ré-entraînements finaux")
    # Par défaut on NE lance PAS B2 ; pour l'activer, passer --run-b2
    parser.add_argument("--run-b2", action="store_true", help="Lancer le fine-tune B2 final")
    args = parser.parse_args()

    run_dir = Path(make_run_dir())
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Répartition des trials entre workers (un process ~ un GPU)
    rank, world = get_worker_info()
    world = max(1, world)

    trials_a_each = max(1, math.ceil(args.trials_a / world))
    trials_b_each = max(1, math.ceil(args.trials_b / world))

    if on_rank_zero():
        print(f"[stage A] essais totaux demandés={args.trials_a} → par worker={trials_a_each}")
    a_ckpt, a_best, a_params = run_optuna_stage_A(
        n_trials=trials_a_each,
        epochs=args.epochs_a,
        seed=args.seed,
        n_train_samples=args.train_samples,
        n_val_samples=args.val_samples,
    )

    # S'assure que le meilleur ckpt A est bien écrit (cas multi-workers)
    wait_for_file(a_ckpt)

    if on_rank_zero():
        print(f"[stage B] essais totaux demandés={args.trials_b} → par worker={trials_b_each}")
    b_ckpt, b_best, b_params = run_optuna_stage_B1(
        n_trials=trials_b_each,
        epochs=args.epochs_b,
        seed=args.seed,
        ckpt_A_path=a_ckpt,
        n_train_samples=args.train_samples,
        n_val_samples=args.val_samples,
    )

    wait_for_file(b_ckpt)

    if on_rank_zero():
        print("\n========== RÉ-ENTRAÎNEMENT STAGE A ==========")
        a_re_ckpt, a_re_score = retrain_stage_A(
            a_params, epochs=50, seed=args.seed, n_train=args.retrain_samples
        )
        wait_for_file(a_re_ckpt)

        print("\n========== RÉ-ENTRAÎNEMENT STAGE B ==========")
        b_re_ckpt, b_re_score = retrain_stage_B(
            b_params, a_params, a_re_ckpt, epochs=50, seed=args.seed, n_train=args.retrain_samples
        )
        wait_for_file(b_re_ckpt)

        print("\n========== FINE-TUNE ENSEMBLE ==========")
        ensemble_ckpt, ensemble_score = finetune_ensemble(
            b_re_ckpt, a_params, b_params, epochs=50, seed=args.seed, n_train=args.retrain_samples
        )
        wait_for_file(ensemble_ckpt)

        stage_dir_A = get_stage_dir(run_dir, "A")
        stage_dir_B = get_stage_dir(run_dir, "B")

        a_err_files = {}
        b_err_files = {}
        try:
            a_err_files = export_param_errors_files(
                a_re_ckpt, "A", refine=False, split="val", robust_smape=False
            )
        except Exception as e:
            print(f"[warn] export erreurs A a échoué: {e}")
        try:
            b_err_files = export_param_errors_files(
                ensemble_ckpt, "B", refine=True, split="val", robust_smape=False
            )
        except Exception as e:
            print(f"[warn] export erreurs B a échoué: {e}")

        summary = {
            "study_root": str(run_dir),
            "A": {
                "optuna": {
                    "ckpt": a_ckpt,
                    "best_val_loss": a_best,
                    "best_params": a_params,
                },
                "retrain": {
                    "ckpt": a_re_ckpt,
                    "best_val_loss": a_re_score,
                    "epochs": 50,
                    "n_train": args.retrain_samples,
                },
                "trials_csv": str(stage_dir_A / "trials.csv"),
                "top5_params": str(stage_dir_A / "top5.json"),
                "errors_files": a_err_files,
            },
            "B": {
                "optuna": {
                    "ckpt": b_ckpt,
                    "best_val_loss": b_best,
                    "best_params": b_params,
                },
                "retrain": {
                    "ckpt": b_re_ckpt,
                    "best_val_loss": b_re_score,
                    "epochs": 50,
                    "n_train": args.retrain_samples,
                },
                "trials_csv": str(stage_dir_B / "trials.csv"),
                "top5_params": str(stage_dir_B / "top5.json"),
                "errors_files": b_err_files,
            },
            "ensemble": {
                "ckpt": ensemble_ckpt,
                "best_val_loss": ensemble_score,
                "epochs": 50,
                "n_train": args.retrain_samples,
            },
        }

        out_summary = run_dir / "summary.json"
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))

        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))

    # Fine-tune B2 optionnel, seulement sur le rank 0
    if args.run_b2 and rank == 0:
        print("\n========== OPTIONAL FINE-TUNE B2 ==========")
        optional_finetune_B2(b_ckpt, epochs=20)
