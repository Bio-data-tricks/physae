from __future__ import annotations

import math, random, sys, os, socket, pickle, json, yaml
from typing import Optional, List, Dict
from pathlib import Path
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

from lion_pytorch import Lion

import matplotlib as mpl
mpl.use("Agg")  # important: définir le backend AVANT d'importer pyplot
try:
    mpl.rcParams['font.family'] = ['DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
except Exception:
    pass
import matplotlib.pyplot as plt  # import unique

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
PARAMS = ['sig0', 'dsig', 'mf_CH4',  'baseline0', 'baseline1', 'baseline2', 'P', 'T'] #'mf_H2O',
PARAM_TO_IDX = {n:i for i,n in enumerate(PARAMS)}
LOG_SCALE_PARAMS = {'mf_CH4'} #'mf_H2O'
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


# ---------- LOWESS unifié ----------
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
# 2) Moteur physique fidèle (HITRAN+TIPS QTpy / Pine / LM) & parsing
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

# ==================== PHYSIQUE (cgs) + TIPS_2021 (QTpy) ====================
# Constantes cgs
C     = 2.99792458e10           # cm/s
NA    = 6.02214129e23
KB    = 1.380649e-16            # erg/K
R     = NA * KB                  # erg/(mol.K)
P0    = 1013.25                  # mbar
T0    = 273.15                   # K
TREF  = 296.0
L0    = 2.6867773e19            # cm^-3 (Loschmidt)
C2    = 1.438776877             # cm.K
SQRT_LN2    = math.sqrt(math.log(2.0))
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

# Paramètres moléculaires (g/mol, longueur m)
MOLECULE_PARAMS = {
    'CH4': {'M': 16.04,     'PL': 15.12},
    'H2O': {'M': 18.01528,  'PL': 15.12},
}

def find_qtpy_dir(pref: str | Path) -> Path:
    p = Path(pref)
    if p.exists() and p.is_dir(): return p.resolve()
    here = Path.cwd()
    for cand in (here / "QTpy", here.parent / "QTpy"):
        if cand.exists(): return cand.resolve()
    raise FileNotFoundError(f"Dossier QTpy introuvable (essayé: {pref}, ./QTpy, ../QTpy).")

class Tips2021QTpy:
    """
    Lecteur QTpy (pickle HITRAN TIPS_2021) + interpolation linéaire en T (entiers).
    """
    def __init__(self, qtpy_dir: str | Path, device: str = 'cpu'):
        self.base = Path(qtpy_dir).resolve()
        if not self.base.exists():
            raise FileNotFoundError(f"Dossier QTpy introuvable: {self.base}")
        self.device = device
        self.cache_dict = {}
        self.cache_table = {}
        self.cache_tmax = {}

    def _path_for(self, mid: int, iso: int) -> Path:
        return self.base / f"{int(mid)}_{int(iso)}.QTpy"

    def _load_one(self, mid: int, iso: int):
        key = (int(mid), int(iso))
        if key in self.cache_dict:
            return
        p = self._path_for(mid, iso)
        if not p.exists():
            raise FileNotFoundError(f"Fichier QTpy manquant pour (mol={mid}, iso={iso}): {p}")
        with open(p, "rb") as h:
            d = pickle.loads(h.read())
        dd = {int(k): float(v) for k, v in d.items()}
        tmax = int(max(dd.keys()))
        table = np.zeros(tmax, dtype=np.float64)
        for T in range(1, tmax + 1):
            if T in dd:
                table[T-1] = dd[T]
            else:
                prev = max([k for k in dd.keys() if k < T], default=min(dd.keys()))
                nxt  = min([k for k in dd.keys() if k > T], default=max(dd.keys()))
                if nxt == prev: table[T-1] = dd[prev]
                else:
                    a = (T - prev) / (nxt - prev)
                    table[T-1] = dd[prev] + a*(dd[nxt] - dd[prev])
        self.cache_dict[key]  = dd
        self.cache_table[key] = table
        self.cache_tmax[key]  = tmax

    def q_scalar(self, mid: int, iso: int, T: float) -> float:
        self._load_one(mid, iso)
        key = (int(mid), int(iso))
        table = self.cache_table[key]; tmax = self.cache_tmax[key]
        if T <= 1: return float(table[0])
        if T >= tmax: return float(table[-1])
        t0 = int(np.floor(T)); t1 = t0 + 1
        f  = (T - t0) / (t1 - t0)
        q1, q2 = table[t0-1], table[t1-1]
        return float(q1 + f*(q2 - q1))

    def q_torch(self, mid: int, iso: int, T: torch.Tensor) -> torch.Tensor:
        self._load_one(mid, iso)
        key = (int(mid), int(iso))
        table = self.cache_table[key]; tmax = self.cache_tmax[key]
        Ts = T.detach().to(dtype=torch.float64, device=self.device)
        Ts = torch.clamp(Ts, 1.0, float(tmax))
        t0 = torch.floor(Ts); t1 = torch.clamp(t0 + 1.0, max=float(tmax))
        f  = (Ts - t0) / torch.clamp(t1 - t0, min=1e-12)
        i0 = (t0.to(torch.int64) - 1).clamp(0, tmax-1)
        i1 = (t1.to(torch.int64) - 1).clamp(0, tmax-1)
        tab = torch.from_numpy(table).to(dtype=torch.float64, device=self.device)
        q1 = tab[i0]; q2 = tab[i1]
        return q1 + f*(q2 - q1)

# ---------- wofz (Humlíček), Pine, line mixing ----------
_b = torch.tensor(
    [-0.0173-0.0463j, -0.7399+0.8395j,  5.8406+0.9536j, -5.5834-11.2086j],
    dtype=torch.cdouble
)
_b = torch.cat((_b, _b.conj()))
_c = torch.tensor(
    [ 2.2377-1.626j ,  1.4652-1.7896j,  0.8393-1.892j ,  0.2739-1.9418j],
    dtype=torch.cdouble
)
_c = torch.cat((_c, -_c.conj()))

def wofz_torch(z: torch.Tensor) -> torch.Tensor:
    b_loc = _b.to(device=z.device, dtype=z.dtype)
    c_loc = _c.to(device=z.device, dtype=z.dtype)
    w_pos = (b_loc / (z.unsqueeze(-1) - c_loc)).sum(dim=-1) * (1j * INV_SQRT_PI)
    w_neg = (b_loc / ((-z).unsqueeze(-1) - c_loc)).sum(dim=-1) * (1j * INV_SQRT_PI)
    return torch.where(z.imag < 0, 2.0*torch.exp(-(z**2)) - w_neg, w_pos)

def pine_profile_torch_complex(x, sigma_hwhm, gamma, g_dicke):
    xh = SQRT_LN2 * x / sigma_hwhm
    yh = SQRT_LN2 * gamma / sigma_hwhm
    zD = SQRT_LN2 * g_dicke / sigma_hwhm
    z  = xh + 1j * (yh + zD)
    k  = -wofz_torch(z)
    k_r, k_i = k.real, k.imag
    pi_sqrt = math.sqrt(math.pi)
    denom = (1 - zD * pi_sqrt * k_r)**2 + (zD * pi_sqrt * k_i)**2
    real = (k_r - zD * pi_sqrt * (k_r**2 + k_i**2)) / denom
    imag = k_i / denom
    factor = math.sqrt(math.log(2.0) / math.pi) / sigma_hwhm  # facteur HWHM (comme script de réf.)
    return real * factor, imag * factor

def apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T, P, *, PREF=P0, TREF_=TREF):
    flm = lmf * ((TREF_ / T) ** nlmf) * (P / PREF)  # dépendance P et T
    return -(real_prof + imag_prof * flm)  # signe absorption

def polyval_torch(coeffs, x):
    powers = torch.arange(coeffs.shape[1], device=coeffs.device, dtype=coeffs.dtype)
    return torch.sum(coeffs.unsqueeze(2) * x.unsqueeze(0).pow(powers.view(1, -1, 1)), dim=1)

def ST_hitran_with_qtpy(
    Sref,                # intensité à Tref (HITRAN, 296 K) — [1,L,1] ou [B,L,1]
    nu0,                 # fréquence de transition (cm^-1)   — [1,L,1] ou [B,L,1]
    e0,                  # énergie d'état bas (cm^-1)        — [1,L,1] ou [B,L,1]
    abundance,           # (conservé pour compat)             — [1,L,1] ou [B,L,1]
    def_abundance,       # (conservé pour compat)             — [1,L,1] ou [B,L,1]
    mid_arr,             # ID molécule par ligne              — [1,L,1] (long)
    iso_arr,             # ID isotopologue par ligne          — [1,L,1] (long)
    T_exp,               # température expérimentale          — [B,1,1] ou [B] (K)
    mf=None,             # ignoré (pour compat signature)
    tipspy=None,         # objet Tips2021QTpy (requis)
    Tref: float = 296.0, # température de référence HITRAN
    device=None,
):
    """
    Retourne S(T) pour chaque échantillon du batch, en utilisant Q(T) issu de QTpy/TIPS_2021,
    calculé **par ligne spectrale et par échantillon** (via tipspy.q_torch).
    Shapes de sortie: [B, L, 1] (diffuse correctement avec le reste de la physique).
    """
    import torch

    if tipspy is None:
        raise RuntimeError("tipspy (QTpy/TIPS) est requis pour ST_hitran_with_qtpy().")

    # -- Device/dtype cohérents
    if device is None:
        if torch.is_tensor(Sref):
            device = Sref.device
        elif torch.is_tensor(T_exp):
            device = T_exp.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = Sref.dtype if torch.is_tensor(Sref) else torch.float32

    def to_tensor(x, dtype_=dtype):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype_, non_blocking=True)
        return torch.as_tensor(x, device=device, dtype=dtype_)

    Sref = to_tensor(Sref)
    nu0  = to_tensor(nu0)
    e0   = to_tensor(e0)
    abundance     = to_tensor(abundance)
    def_abundance = to_tensor(def_abundance)
    mid_arr = to_tensor(mid_arr, dtype_=torch.long)
    iso_arr = to_tensor(iso_arr, dtype_=torch.long)
    T_exp   = to_tensor(T_exp)

    # Mise en forme T_exp -> [B,1,1] et extraction Ts -> [B]
    if T_exp.ndim == 1:
        T_exp = T_exp.view(-1, 1, 1)
    elif T_exp.ndim == 2:
        T_exp = T_exp.view(T_exp.shape[0], 1, 1)
    B = T_exp.shape[0]

    # Déduire L (nb de lignes)
    # Sref/nu0/e0 sont typiquement [1,L,1] (ou [B,L,1] par broadcast). On lit la deuxième dim.
    if Sref.ndim == 3:
        L = Sref.shape[1]
    else:
        # fallback si fourni 1D : on tente via mid_arr
        L = mid_arr.view(-1).numel()

    # Constante c2 en dtype/device correct
    c2 = torch.tensor(1.438776877, device=device, dtype=dtype)

    # === Q(T) et Q(Tref) vectorisés par ligne & par échantillon ===
    # Indices (L) des paires (mid, iso)
    mid_L = mid_arr.view(-1).to(torch.long)[:L]
    iso_L = iso_arr.view(-1).to(torch.long)[:L]
    key   = mid_L * 100 + iso_L
    uniq  = torch.unique(key)

    # Sorties [B, L, 1]
    Q_T    = torch.empty((B, L, 1), device=device, dtype=dtype)
    Q_refT = torch.empty((B, L, 1), device=device, dtype=dtype)

    # Températures échantillon [B]
    Ts = T_exp.view(B)

    # Pour chaque groupe (molécule, isotopologue), on appelle tipspy.q_torch une seule fois
    for k in uniq:
        k = int(k.item())
        mid_i = k // 100
        iso_i = k % 100
        cols  = (key == k).nonzero(as_tuple=True)[0]  # indices colonnes de ce groupe, shape [K]

        # q_torch retourne [B] (float64 côté QTpy) → cast en dtype/device cible
        qT_b  = tipspy.q_torch(mid_i, iso_i, Ts).to(device=device, dtype=dtype)          # [B]
        qRef_b= tipspy.q_torch(mid_i, iso_i, torch.full_like(Ts, float(Tref)))\
                    .to(device=device, dtype=dtype)                                       # [B]

        # Répliquer sur les colonnes de ce groupe : [B, K, 1]
        Q_T[:, cols, 0]    = qT_b.view(B, 1).expand(B, cols.numel())
        Q_refT[:, cols, 0] = qRef_b.view(B, 1).expand(B, cols.numel())

    # === Facteurs de Boltzmann et ratio (broadcast [B,L,1]) ===
    Tref_t = torch.tensor(float(Tref), device=device, dtype=dtype)
    invT    = 1.0 / T_exp
    invTref = 1.0 / Tref_t

    expo_fac = torch.exp(-c2 * e0 * (invT - invTref))
    num_fac  = 1.0 - torch.exp(-c2 * nu0 * invT)
    den_fac  = 1.0 - torch.exp(-c2 * nu0 * invTref)

    # Sécurités numériques
    eps = torch.tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
    den_fac = torch.where(den_fac == 0, eps, den_fac)
    Q_T     = torch.where(Q_T == 0,     eps, Q_T)

    # abundance = torch.where(abundance == 0, eps, abundance)
    # def_abundance = torch.where(def_abundance == 0, eps, def_abundance)
    # abundance_ratio = (abundance / def_abundance)
    # S_T = Sref * (Q_refT / Q_T) * expo_fac * (num_fac / den_fac) * abundance_ratio

    # Intensité à T — shape [B, L, 1]
    S_T = Sref * (Q_refT / Q_T) * expo_fac * (num_fac / den_fac)
    return S_T

def batch_physics_forward_multimol_vgrid(
    sig0, dsig, poly_freq, v_grid_idx, baseline_coeffs,
    transitions_dict, P, T, mf_dict, *, tipspy: Tips2021QTpy, device='cpu', USE_LM: bool=True
):
    """
    Implémentation fidèle (script Spectro CH4):
      - shift_air appliqué
      - Pine + Dicke effectif
      - LM avec flm ∝ (Tref/T)^nlmf * (P/P0), signe absorption
      - S(T) HITRAN avec Q(T) via QTpy
      - Colonne: (P/P0)*(T0/T)*L0*PL*100
      - transmission = exp(total_profile) (profile déjà signé)
    """
    B, N = sig0.shape[0], v_grid_idx.shape[0]
    v_grid_idx = v_grid_idx.to(device=device, dtype=torch.float64)
    sig0 = sig0.to(dtype=torch.float64, device=device).unsqueeze(1)
    dsig = dsig.to(dtype=torch.float64, device=device).unsqueeze(1)
    P    = P.to(dtype=torch.float64, device=device).unsqueeze(1)
    T    = T.to(dtype=torch.float64, device=device).unsqueeze(1)

    if baseline_coeffs.dim() == 1: baseline_coeffs = baseline_coeffs.unsqueeze(0)
    baseline_coeffs = baseline_coeffs.to(dtype=torch.float64, device=device)

    poly_freq_torch = torch.tensor(poly_freq, dtype=torch.float64, device=device).unsqueeze(0).expand(B, -1)
    coeffs = torch.cat([sig0, dsig, poly_freq_torch], dim=1)
    v_grid_batch = polyval_torch(coeffs, v_grid_idx)  # (B,N)

    total_profile = torch.zeros((B, N), device=device, dtype=torch.float64)

    P0_t   = torch.tensor(P0,   dtype=torch.float64, device=device)
    T0_t   = torch.tensor(T0,   dtype=torch.float64, device=device)
    TREF_t = torch.tensor(TREF, dtype=torch.float64, device=device)
    L0_t   = torch.tensor(L0,   dtype=torch.float64, device=device)
    C_t    = torch.tensor(C,    dtype=torch.float64, device=device)
    R_t    = torch.tensor(R,    dtype=torch.float64, device=device)

    for mol, trans in transitions_dict.items():
        (amp, center, ga, gs, na, sa, gd, nd, lmf, nlmf) = [
            t.to(dtype=torch.float64, device=device).view(1, -1, 1)
            for t in transitions_to_tensors(trans, device)
        ]
        e0  = torch.tensor([t['e0']        for t in trans], dtype=torch.float64, device=device).view(1, -1, 1)
        abn = torch.tensor([t['abundance'] for t in trans], dtype=torch.float64, device=device).view(1, -1, 1)
        def_abn = torch.ones_like(abn)
        mid_arr = torch.tensor([t['mid']   for t in trans], dtype=torch.int64, device=device).view(1, -1, 1)
        iso_arr = torch.tensor([t['lid']   for t in trans], dtype=torch.int64, device=device).view(1, -1, 1)

        mf   = mf_dict[mol].to(dtype=torch.float64, device=device).view(B, 1, 1)  # fraction molaire
        Mmol = torch.tensor(MOLECULE_PARAMS[mol]['M'],  dtype=torch.float64, device=device)
        PL   = torch.tensor(MOLECULE_PARAMS[mol]['PL'], dtype=torch.float64, device=device)

        T_exp = T.view(B, 1, 1)
        P_exp = P.view(B, 1, 1)
        v_exp = v_grid_batch.view(B, 1, N)

        # shift de pression
        x = v_exp - (center + sa * (P_exp / P0_t))

        # HWHM doppler
        sigma_HWHM = (center / C_t) * torch.sqrt(2.0 * R_t * T_exp * math.log(2.0) / Mmol)

        # élargissement collisionnel + Dicke effectif
        gamma  = (P_exp / P0_t) * (TREF_t / T_exp) ** na * (ga * (1 - mf) + gs * mf)
        gN_eff = gd * (P_exp / P0_t) * (TREF_t / T_exp) ** nd

        real_prof, imag_prof = pine_profile_torch_complex(x, sigma_HWHM, gamma, gN_eff)
        if USE_LM:
            profile = apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T=T_exp, P=P_exp)
        else:
            profile = -real_prof  # signe absorption sans LM

        # S(T) exact (HITRAN + Q(T))
        S_T = ST_hitran_with_qtpy(
            Sref=amp, nu0=center, e0=e0, abundance=abn, def_abundance=def_abn,
            mid_arr=mid_arr, iso_arr=iso_arr, T_exp=T_exp, mf=mf, tipspy=tipspy,
            device=device,
        )

        # Colonne (m -> cm via *100)
        col = (P_exp / P0_t) * (T0_t / T_exp) * L0_t * PL * 100.0 * mf

        band = profile * S_T * col  # (B, L, N) → somme sur L
        total_profile += band.sum(dim=1)

    transmission = torch.exp(total_profile)  # profile déjà signé

    # baseline polynomiale sur l'index
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
        # Normalisation : on retire le biais DC puis on met à l'échelle
        fringes = fringes - fringes.mean(dim=-1, keepdim=True)
        std = fringes.std(dim=-1, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        fringes = fringes / std
        span = (y.max(dim=-1, keepdim=True).values - y.min(dim=-1, keepdim=True).values).clamp_min(1e-6)
        final_amp = r(*fringe_amp_range)
        fringes = fringes * (final_amp * span)
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
                 freeze_noise: bool = False, tipspy: Tips2021QTpy | None = None):
        self.n_samples, self.num_points = n_samples, num_points
        self.poly_freq_CH4, self.transitions_dict = poly_freq_CH4, transitions_dict
        self.sample_ranges = sample_ranges if sample_ranges is not None else NORM_PARAMS
        self.with_noise = bool(with_noise)
        self.noise_profile = dict(noise_profile or {})
        self.freeze_noise = bool(freeze_noise)
        self.tipspy = tipspy
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

        # Tirage des paramètres (dico générique)
        sampled = {k: torch.empty(1, dtype=dtype).uniform_(*self.sample_ranges[k]) for k in PARAMS}
        sig0 = sampled['sig0']; dsig = sampled['dsig']
        b0 = sampled['baseline0']; b1 = sampled['baseline1']; b2 = sampled['baseline2']
        P = sampled['P']; T = sampled['T']

        baseline_coeffs = torch.cat([b0, b1, b2]).unsqueeze(0)
        v_grid_idx = torch.arange(self.num_points, dtype=dtype, device=device)

        # Fractions molaires utilisées par la physique
        mf_dict = {}
        if 'CH4' in self.transitions_dict:
            mf_dict['CH4'] = sampled['mf_CH4']
        if 'H2O' in self.transitions_dict:
            mf_dict['H2O'] = sampled['mf_H2O']  # <-- varie dans la simu

        # Vecteur normalisé
        params_norm = torch.tensor([norm_param_value(k, sampled[k].item()) for k in PARAMS], dtype=torch.float32)

        spectra_clean, _ = batch_physics_forward_multimol_vgrid(
            sig0, dsig, self.poly_freq_CH4, v_grid_idx,
            baseline_coeffs, self.transitions_dict, P, T, mf_dict,
            tipspy=self.tipspy, device=device
        )
        spectra_clean = spectra_clean.to(torch.float32)

        if self.with_noise:
            g = self._make_generator(idx)
            spectra_noisy = add_noise_variety(spectra_clean, generator=g, **self.noise_profile)
        else:
            spectra_noisy = spectra_clean

        # --- ÉCHELLE POUR L'ENTRÉE (noisy) ---
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

# --- replace / étendre ton ConvBNAct1d ---
class ConvBNAct1d(nn.Sequential):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, bias=False, act=True, d=1):
        if p is None:
            p = ((k - 1) // 2) * d
        mods = [nn.Conv1d(in_c, out_c, k, s, p, dilation=d, groups=g, bias=bias), Norm1d(out_c)]
        if act: mods.append(SiLU())
        super().__init__(*mods)

# --- nouveau: BlurPool anti-alias ---
class BlurPool1D(nn.Module):
    def __init__(self, c, kernel=(1,4,6,4,1)):
        super().__init__()
        k = torch.tensor(kernel, dtype=torch.float32)
        k = (k / k.sum()).view(1,1,-1).repeat(c,1,1)
        self.register_buffer('k', k)
    def forward(self, x):  # x: [B,C,N]
        return F.conv1d(x, self.k, stride=2, padding=self.k.shape[-1]//2, groups=x.size(1))


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




class ChannelAttention1d(nn.Module):
    """Squeeze-and-Excitation amélioré avec contexte global"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class GatedSharedHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, p_drop: float):
        super().__init__()
        self.short_head = nn.Linear(feat_dim, hidden)
        self.deep_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p_drop),
        )
        self.gate_head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_head(x)
        return g * self.deep_head(x) + (1 - g) * self.short_head(x)


class SpatialAttention1d(nn.Module):
    """Attention sur la dimension temporelle/spectrale"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))

# class CBAM1d(nn.Module):
#     """Convolutional Block Attention Module pour 1D"""
#     def __init__(self, channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.channel_att = ChannelAttention1d(channels, reduction)
#         self.spatial_att = SpatialAttention1d(kernel_size)
    
#     def forward(self, x):
#         x = self.channel_att(x)
#         x = self.spatial_att(x)
#         return x

# class MBConv1d(nn.Module):
#     def __init__(self, in_c, out_c, k, s, expand_ratio, se_ratio=0.25, 
#                  drop_path=0.0, use_cbam=True):  # <- NEW
#         super().__init__()
#         mid = int(in_c * expand_ratio)
#         self.use_res = (s == 1 and in_c == out_c)
        
#         layers = []
#         if expand_ratio != 1:
#             layers.append(ConvBNAct1d(in_c, mid, k=1, s=1, act=True))
        
#         layers.append(ConvBNAct1d(mid, mid, k=k, s=s, p=k//2, g=mid, act=True))
        
#         # CHOIX: SE1d OU CBAM (CBAM généralement meilleur)
#         if use_cbam:
#             layers.append(CBAM1d(mid, reduction=max(4, mid//16)))
#         elif se_ratio is not None and se_ratio > 0:
#             layers.append(SE1d(mid, se_ratio=se_ratio))
        
#         layers.append(ConvBNAct1d(mid, out_c, k=1, s=1, act=False))
#         self.block = nn.Sequential(*layers)
#         self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
#     def forward(self, x):
#         y = self.block(x)
#         if self.use_res:
#             y = self.drop(y) + x
#         return y

# MBConc normal (sans CBAM)

class FusedMBConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, s, expand_ratio, drop_path=0.0):
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
        # LayerScale (gamma) — démarre proche de l'identité
        self.gamma = nn.Parameter(torch.ones(1, out_c, 1) * 1e-3)

    def forward(self, x):
        y = self.fused(x)
        if self.use_res:
            y = x + self.drop(self.gamma * y)
        return y


class MBConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, s, expand_ratio, se_ratio=0.25, drop_path=0.0, dw_dilation: int = 1):
        super().__init__()
        mid = int(in_c * expand_ratio)
        self.use_res = (s == 1 and in_c == out_c)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct1d(in_c, mid, k=1, s=1, act=True))

        # depthwise (dilatation modérée configurable)
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

_EFFV2_CFGS = {
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
        mb_dilate_from_stage: int = 3,   # à partir de ce "stage" (0-index) pour les 'mb'
        mb_dilation_value: int = 2,
    ):
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
        total_blocks = sum(int(math.ceil(r * depth_mult)) for (_, r, *_ ) in cfg)
        b_idx = 0
        stage_idx = -1
        for (kind, repeats, out_c, k, s, exp) in cfg:
            stage_idx += 1
            out_c = _make_divisible(out_c * width_mult, 8)
            repeats = int(math.ceil(repeats * depth_mult))

            for i in range(repeats):
                stride = s if i == 0 else 1
                dp = drop_path_rate * b_idx / max(1, total_blocks - 1)

                # --- anti-alias downsample: BlurPool avant le bloc, et bloc en stride=1
                pre = []
                if use_blurpool and stride == 2:
                    pre.append(BlurPool1D(in_c))
                    stride = 1  # le bloc reste stride=1 grâce au BlurPool

                # --- dilatation sur certains MBConv profonds
                dw_dil = 1
                if kind == "mb" and stage_idx >= mb_dilate_from_stage:
                    dw_dil = mb_dilation_value  # ex: 2

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

    def forward(self, x):  # x: [B, C, N]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x, None


# ============================================================
# 5) ReLoBRaLo
# ============================================================
class ReLoBRaLoLoss:
    """
    ReLoBRaLo avec tirage Torch déterministe (DDP friendly).
    Supporte à la fois num_losses (int) et loss_names (list) pour compatibilité.
    """
    def __init__(self, num_losses=None, loss_names=None, alpha=0.9, tau=1.0, history_len=10, seed=12345):
        """
        Args:
            num_losses: Nombre de losses (int) - recommandé
            loss_names: Liste des noms (list) - pour compatibilité ancienne API
            alpha: Facteur de lissage EMA
            tau: Température pour softmax
            history_len: Longueur de l'historique
            seed: Seed pour reproductibilité
        """
        # ✅ Compatibilité avec les deux signatures
        if num_losses is not None:
            self.num_losses = int(num_losses)
            self.loss_names = [f"loss_{i}" for i in range(self.num_losses)]
        elif loss_names is not None:
            self.loss_names = list(loss_names)
            self.num_losses = len(self.loss_names)
        else:
            raise ValueError("Fournir soit num_losses (int) soit loss_names (list)")
        
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.history_len = int(history_len)
        self.loss_history = {name: [] for name in self.loss_names}
        self.weights = torch.ones(self.num_losses, dtype=torch.float32)
        self.last_weights = torch.ones(self.num_losses, dtype=torch.float32) / self.num_losses  # ← AJOUTÉ
        self._g = torch.Generator(device="cpu")
        if seed is not None:
            self._g.manual_seed(int(seed))
    
    def set_seed(self, seed: int):
        self._g.manual_seed(int(seed))
    
    def to(self, device=None, dtype=None):
        if device is not None or dtype is not None:
            self.weights = self.weights.to(
                device=device or self.weights.device,
                dtype=dtype or self.weights.dtype
            )
            self.last_weights = self.last_weights.to(  # ← AJOUTÉ
                device=device or self.last_weights.device,
                dtype=dtype or self.last_weights.dtype
            )
        return self
    
    def _append_history(self, current_losses):
        """Ajoute les losses courantes à l'historique"""
        for i, name in enumerate(self.loss_names):
            if i < len(current_losses):
                self.loss_history[name].append(float(current_losses[i].detach().cpu()))
                if len(self.loss_history[name]) > self.history_len:
                    self.loss_history[name].pop(0)
    
    @torch.no_grad()
    def compute_weights(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Calcule les poids adaptatifs pour les losses.
        
        Args:
            current_losses: [N] tensor des losses individuelles
        
        Returns:
            weights: [N] poids pondérés
        """
        device = current_losses.device
        dtype = current_losses.dtype
        
        # ✅ Vérifier la cohérence
        if current_losses.shape[0] != self.num_losses:
            raise ValueError(
                f"Nombre de losses incohérent: attendu {self.num_losses}, "
                f"reçu {current_losses.shape[0]}"
            )
        
        self._append_history(current_losses)
        
        # Besoin d'au moins 2 valeurs
        if len(self.loss_history[self.loss_names[0]]) < 2:
            w = self.weights.to(device=device, dtype=dtype)
            self.last_weights = w.detach().cpu()  # ← AJOUTÉ
            return w
        
        ratios = []
        for name in self.loss_names:
            hist = self.loss_history[name]
            j = int(torch.randint(low=0, high=len(hist)-1, size=(), generator=self._g).item())
            num = float(hist[-1])
            den = float(hist[j]) + 1e-8
            ratios.append(num / den)
        
        ratios_t = torch.tensor(ratios, device=device, dtype=dtype)
        mean_rel = ratios_t.mean()
        balancing = mean_rel / (ratios_t + 1e-8)
        
        K = self.num_losses
        new_w = K * torch.softmax(balancing / self.tau, dim=0)
        
        w_old = self.weights.to(device=device, dtype=dtype)
        w_new = self.alpha * w_old + (1.0 - self.alpha) * new_w
        
        self.weights = w_new.detach().cpu()
        self.last_weights = w_new.detach().cpu()  # ← AJOUTÉ
        
        return w_new
    
    def __call__(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Calcule la loss totale pondérée par ReLoBRaLo.
        
        Args:
            current_losses: [N] tensor des losses individuelles
        
        Returns:
            total_loss: Scalar tensor de la loss totale pondérée
        """
        weights = self.compute_weights(current_losses)
        total_loss = (weights * current_losses).sum()
        return total_loss
    
    def forward(self, current_losses: torch.Tensor) -> torch.Tensor:
        """Alias pour __call__ (compatibilité nn.Module style)"""
        return self.__call__(current_losses)
# ============================================================
# 6) Raffineur type EfficientNet (paramétrable)
# ============================================================
class EfficientNetRefiner(nn.Module):
    """
    Raffineur simplifié : prend UNIQUEMENT le résidu en entrée.
    Prédit directement les corrections delta à partir du résidu.
    """
    def __init__(
        self,
        m_params: int,
        *,
        delta_scale: float = 0.1,
        encoder_variant: str = "s",
        encoder_width_mult: float = 1.0,
        encoder_depth_mult: float = 1.0,
        encoder_stem_channels: int | None = None,
        encoder_drop_path: float = 0.1,
        encoder_se_ratio: float = 0.25,
        feature_pool: str = "avg",
        hidden_dim: int = 128,
        mlp_dropout: float = 0.10,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.m_params = int(m_params)
        self.mlp_dropout = float(mlp_dropout)

        # ✅ Encodeur : 1 seul canal (résidu)
        self.encoder = EfficientNetEncoder(
            in_channels=1,
            variant=encoder_variant,
            width_mult=encoder_width_mult,
            depth_mult=encoder_depth_mult,
            se_ratio=encoder_se_ratio,
            drop_path_rate=encoder_drop_path,
            stem_channels=encoder_stem_channels,
        )

        D = self.encoder.feat_dim
        pool = feature_pool.lower()
        self._feature_pool_mode = pool
        
        if pool == "avg":
            self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        elif pool == "max":
            self.feature_head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        elif pool == "avgmax":
            self.feature_head = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
        else:
            raise ValueError(f"feature_pool inconnu: {feature_pool}")

        in_dim = D if pool != "avgmax" else 2 * D
        H = hidden_dim

        # ✅ MLP : features → hidden
        self.mlp_hidden = nn.Sequential(
            nn.Linear(in_dim, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            
            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
        )
        
        # ✅ Gate adaptatif par paramètre (APRÈS le MLP)
        self.scale_gate = nn.Linear(H, m_params)
        
        # ✅ Tête de prédiction des deltas
        self.delta_head = nn.Linear(H, m_params)

    def _pool_features(self, latent: torch.Tensor) -> torch.Tensor:
        if self._feature_pool_mode == "avg":
            return self.feature_head(latent)
        if self._feature_pool_mode == "max":
            return self.feature_head(latent)
        avgp, maxp = self.feature_head
        a = avgp(latent).flatten(1)
        m = maxp(latent).flatten(1)
        return torch.cat([a, m], dim=1)

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        """
        ✅ SIMPLE : uniquement le résidu en entrée
        
        Args:
            resid: [B, N] résidu (recon - target)
        
        Returns:
            delta: [B, M] corrections des paramètres
        """
        # Encoder le résidu
        x = resid.unsqueeze(1)  # [B, 1, N]
        latent, _ = self.encoder(x)  # [B, D, N']
        feat = self._pool_features(latent)  # [B, in_dim]
        
        # Passer par le MLP
        h = self.mlp_hidden(feat)  # [B, H]
        
        # Gate adaptatif (contrôle l'amplitude par paramètre)
        gate = torch.sigmoid(self.scale_gate(h))  # [B, M]
        scale = self.delta_scale * gate
        
        # Prédire les corrections brutes
        raw_delta = self.delta_head(h)  # [B, M]
        
        # Corrections bornées
        delta = torch.tanh(raw_delta) * scale  # [B, M]
        return delta

class ResidualBlock1D(nn.Module):
    def __init__(self, c, k=7, d=1, p=None):
        super().__init__()
        if p is None: p = (k - 1) // 2 * d
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
    Débruiteur simple, *longueur conservée* (stride=1), sortie = correction du résidu.
    """
    def __init__(self, in_ch=1, base_ch=64, depth=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3, bias=False),
            Norm1d(base_ch),
            SiLU(),
        )
        blocks = []
        # échelle de dilatations: 1,2,4,8... modérée
        for i in range(depth):
            d = 2**(i % 4)   # cycle 1,2,4,8
            blocks.append(ResidualBlock1D(base_ch, k=7, d=d))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(base_ch, 1, kernel_size=1, bias=True)

    def forward(self, resid):    # resid: [B, N]  → correction: [B, N]
        y = self.stem(resid.unsqueeze(1))
        y = self.blocks(y)
        y = self.head(y).squeeze(1)
        return y


def _design_poly3(n: int, device, dtype):
    # colonnes: [1, x, x^2, x^3] avec x centré-échelonné pour stabilité
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    xc = x - x.mean()
    X = torch.stack([torch.ones_like(xc), xc, xc**2, xc**3], dim=1)  # [N,4]
    return X

def baseline_poly3_from_edges(resid: torch.Tensor, left_frac: float = 0.20, right_start: float = 0.75):
    """
    resid: [B, N]  (résidu recon - cible)
    Retourne: resid_corr [B,N] où un polynôme d'ordre 3 ajusté sur (0..20%) U (75..100%) a été soustrait.
    """
    B, N = resid.shape
    device, dtype = resid.device, resid.dtype
    Xfull = _design_poly3(N, device, resid.dtype)              # [N,4]
    iL = torch.arange(0, max(1, int(N*left_frac)), device=device)
    iR = torch.arange(int(N*right_start), N, device=device)
    idx = torch.cat([iL, iR], dim=0)                           # [M]
    X = Xfull[idx]                                             # [M,4]
    # (X^T X)^{-1} X^T y, batched
    Xt = X.t()                                                 # [4,M]
    XtX = Xt @ X                                               # [4,4]
    # régularisation légère pour stabilité
    lam = 1e-6
    XtX = XtX + lam * torch.eye(4, device=device, dtype=dtype)
    XtX_inv = torch.linalg.inv(XtX)                            # [4,4]
    P = XtX_inv @ Xt                                           # [4,M]
    y_edges = resid[:, idx]                                    # [B,M]
    coeff = (P @ y_edges.T).T                                  # [B,4]
    baseline = (Xfull @ coeff.transpose(0,1)).transpose(0,1)   # [B,N]
    resid_corr = resid - baseline
    return resid_corr, baseline

# ============================================================
# LOSS FUNCTIONS AMÉLIORÉES - CONFIGURATION MINIMALE
# ============================================================

class SpectralAngleLoss(nn.Module):
    """
    Mesure l'angle entre spectres (insensible à l'échelle).
    Plus robuste que MSE pour les variations d'amplitude globales.
    Remplace: w_corr_raw, w_corr_d1, w_corr_d2
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, N] spectres prédits
            target: [B, N] spectres cibles
        Returns:
            loss scalaire (moyenne sur le batch)
        """
        # Normalisation L2
        pred_norm = F.normalize(pred, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=1, eps=self.eps)
        
        # Similarité cosinus
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        
        # Clamp pour stabilité numérique
        cosine_sim = torch.clamp(cosine_sim, -1.0 + self.eps, 1.0 - self.eps)
        
        # Convertir en loss (1 - similarité)
        return (1.0 - cosine_sim).mean()

# ============================================================
# APRÈS LES IMPORTS, AVANT LES AUTRES CLASSES
# ============================================================
from torch.optim.lr_scheduler import _LRScheduler

# ============================================================
# CLASSE DU SCHEDULER (à ajouter au début de votre fichier principal)
# ============================================================

class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    """
    Cosine Annealing avec Warm Restarts, Decay progressif et Linear Warmup
    """
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: float = 1,
        eta_min: float = 0,
        decay_factor: float = 1.0,
        warmup_epochs: int = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.T_0 = T_0
        self.T_i = float(T_0)
        self.T_mult = float(T_mult)
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_max_lrs = self.base_lrs.copy()
        self.T_cur = 0
        self.restart_count = 0
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        current_epoch = self.last_epoch
        
        # Phase warmup
        if current_epoch < self.warmup_epochs:
            warmup_factor = (current_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Phase cosine
        adjusted_epoch = current_epoch - self.warmup_epochs
        
        if adjusted_epoch == 0:
            self.T_cur = 0
            self.T_i = float(self.T_0)
        elif self.T_cur >= int(self.T_i):
            self.restart_count += 1
            self.T_cur = 0
            self.current_max_lrs = [lr * self.decay_factor for lr in self.current_max_lrs]
            self.T_i = self.T_i * self.T_mult
        
        lrs = []
        for max_lr in self.current_max_lrs:
            lr = self.eta_min + (max_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / self.T_i)
            ) / 2
            lrs.append(lr)
        
        self.T_cur += 1
        return lrs
    
    def get_schedule_info(self, max_epochs: int) -> dict:
        """Simule le schedule pour visualisation"""
        original_state = {
            'last_epoch': self.last_epoch,
            'T_cur': self.T_cur,
            'T_i': self.T_i,
            'restart_count': self.restart_count,
            'current_max_lrs': self.current_max_lrs.copy(),
        }
        
        self.last_epoch = -1
        self.T_cur = 0
        self.T_i = float(self.T_0)
        self.restart_count = 0
        self.current_max_lrs = self.base_lrs.copy()
        
        epochs = []
        lrs = []
        restart_epochs = []
        
        for epoch in range(max_epochs):
            self.step()
            epochs.append(epoch)
            lrs.append(self.get_last_lr()[0])
            
            adjusted_epoch = epoch - self.warmup_epochs
            if adjusted_epoch > 0 and self.T_cur == 1:
                restart_epochs.append(epoch)
        
        # Restaurer l'état
        self.last_epoch = original_state['last_epoch']
        self.T_cur = original_state['T_cur']
        self.T_i = original_state['T_i']
        self.restart_count = original_state['restart_count']
        self.current_max_lrs = original_state['current_max_lrs']
        
        return {
            'epochs': epochs,
            'lrs': lrs,
            'restart_epochs': restart_epochs,
            'warmup_end': self.warmup_epochs - 1 if self.warmup_epochs > 0 else None,
        }



import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakWeightedMSELoss(nn.Module):
    """
    Pondération par détection de pics via dérivées (100% PyTorch) :
      - lissage gaussien AVANT dérivées,
      - dérivées one-sided aux bords + atténuation cosine,
      - carte de saillance 'soft' (0..1) lissée (+ gamma),
      - élargissement optionnel (gaussian / maxpool),
      - conversion en poids puis MSE pondérée.
    """
    def __init__(
        self,
        peak_weight: float = 4.0,
        baseline_weight: float = 1.0,
        pre_smooth_sigma: float = 1.3,        # points
        salience_smooth_sigma: float = 1.8,   # points
        peak_kind: str = "min",               # "min" (absorption) ou "max"
        curv_scale_k: float = 2.2,
        border_policy: str = "taper",         # "taper" ou "zero"
        border_extra_margin: int = 2,
        weight_normalize: str = "mean",       # "mean" ou "none"
        renorm_after_smoothing: bool = True,
        salience_gamma: float = 0.9,
        spread_kind: str | None = "gaussian", # "gaussian", "maxpool", ou None
        spread_sigma: float = 2.5,            # si gaussian (← un peu large par défaut)
        spread_kernel: int = 11,              # si maxpool (impair)
    ):
        super().__init__()
        self.peak_weight = float(peak_weight)
        self.baseline_weight = float(baseline_weight)
        self.pre_smooth_sigma = float(pre_smooth_sigma)
        self.salience_smooth_sigma = float(salience_smooth_sigma)
        self.peak_kind = str(peak_kind)
        self.curv_scale_k = float(curv_scale_k)
        self.border_policy = str(border_policy)
        self.border_extra_margin = int(border_extra_margin)
        self.weight_normalize = str(weight_normalize).lower()
        self.renorm_after_smoothing = bool(renorm_after_smoothing)
        self.salience_gamma = float(salience_gamma)
        self.spread_kind = (None if spread_kind is None else str(spread_kind).lower())
        self.spread_sigma = float(spread_sigma)
        self.spread_kernel = int(spread_kernel)

    @staticmethod
    def _gaussian_kernel_1d(sigma: float, device, dtype):
        if sigma <= 0:
            return torch.ones(1,1,1, device=device, dtype=dtype)
        k = int(6 * sigma + 1)
        if k % 2 == 0: k += 1
        if k < 3: k = 3
        x = torch.arange(k, device=device, dtype=dtype) - (k // 2)
        g = torch.exp(-(x**2) / (2 * (sigma**2)))
        g = g / (g.sum() + 1e-12)
        return g.view(1,1,-1)

    def _gaussian_smooth1d(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0: return x
        g = self._gaussian_kernel_1d(sigma, x.device, x.dtype)
        pad = g.shape[-1] // 2
        x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
        return F.conv1d(x_pad, g).squeeze(1)

    def _central_one_sided_diffs(self, x: torch.Tensor):
        B, N = x.shape
        d1 = torch.zeros_like(x)
        d2 = torch.zeros_like(x)
        if N >= 2:
            if N > 2:
                d1[:,1:-1] = 0.5 * (x[:,2:] - x[:, :-2])
            d1[:,0]  = (x[:,1] - x[:,0])
            d1[:, -1]= (x[:, -1] - x[:, -2])
        if N >= 3:
            d2[:,1:-1] = x[:,2:] - 2.0 * x[:,1:-1] + x[:, :-2]
            d2[:,0]    = x[:,0] - 2.0 * x[:,1] + x[:,2]
            d2[:, -1]  = x[:, -3] - 2.0 * x[:, -2] + x[:, -1]
        return d1, d2

    @staticmethod
    def _robust_scale_central(t: torch.Tensor, margin: int) -> torch.Tensor:
        B, N = t.shape
        m = max(0, min(margin, (N - 1)//2))
        tc = t[:, m:N-m] if (N - 2*m >= 3) else t
        med = torch.median(tc, dim=-1, keepdim=True).values
        mad = torch.median(torch.abs(tc - med), dim=-1, keepdim=True).values + 1e-12
        return 1.4826 * mad

    @staticmethod
    def _border_taper_window(N: int, device, dtype, margin: int):
        if margin <= 0 or 2*margin >= N:
            return torch.ones(1,N, device=device, dtype=dtype)
        t = torch.linspace(0,1, steps=margin, device=device, dtype=dtype)
        ramp = 0.5 * (1 - torch.cos(torch.pi * t))
        w = torch.ones(N, device=device, dtype=dtype)
        w[:margin]  = ramp
        w[-margin:] = torch.flip(ramp, dims=[0])
        return w.view(1,-1)

    def _detect(self, spectra: torch.Tensor):
        B, N = spectra.shape
        device, dtype = spectra.device, spectra.dtype

        # marge = somme des pads utilisés → évite faux pics aux bords
        m_pre = int(3 * max(self.pre_smooth_sigma, 0.0))
        m_sal = int(3 * max(self.salience_smooth_sigma, 0.0))
        m_spread = int(3 * max(self.spread_sigma if self.spread_kind=="gaussian" else 0.0, 0.0))
        border_margin = max(2, m_pre + m_sal + m_spread + self.border_extra_margin)

        x = self._gaussian_smooth1d(spectra, self.pre_smooth_sigma)
        d1, d2 = self._central_one_sided_diffs(x)
        curv = d2 if self.peak_kind.lower()=="min" else (-d2)

        sigma1 = self._robust_scale_central(d1, border_margin)
        gate = torch.exp(- (d1/(3.0*sigma1))**2)

        sigma2 = self._robust_scale_central(curv, border_margin)
        s_raw = F.softplus(curv / (self.curv_scale_k*sigma2)) * gate

        w_border = self._border_taper_window(N, device, dtype, border_margin)
        if self.border_policy.lower()=="taper":
            s_raw = s_raw * w_border
        elif self.border_policy.lower()=="zero":
            s_raw = s_raw.clone(); s_raw[:, :border_margin]=0; s_raw[:, -border_margin:]=0

        s_raw = s_raw / (s_raw.amax(dim=-1, keepdim=True) + 1e-12)
        s_smooth = self._gaussian_smooth1d(s_raw, self.salience_smooth_sigma).clamp_(0,1)
        if self.salience_gamma != 1.0:
            s_smooth = s_smooth ** self.salience_gamma
        if self.border_policy.lower()=="taper":
            s_smooth = s_smooth * w_border

        s_wide = s_smooth
        if self.spread_kind=="gaussian" and self.spread_sigma>0:
            s_wide = self._gaussian_smooth1d(s_wide, self.spread_sigma)
        elif self.spread_kind=="maxpool":
            k = max(3, self.spread_kernel | 1)
            pad = k // 2
            s_wide = F.max_pool1d(s_wide.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)

        if self.border_policy.lower()=="taper":
            s_wide = s_wide * w_border
        if self.renorm_after_smoothing:
            s_wide = s_wide / (s_wide.amax(dim=-1, keepdim=True) + 1e-12)

        weights = self.baseline_weight + s_wide * (self.peak_weight - self.baseline_weight)
        if self.weight_normalize == "mean":
            weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-12)
        return weights, s_wide, border_margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_debug: bool=False):
        assert pred.shape == target.shape and pred.ndim == 2
        weights, mask_soft, border_margin = self._detect(target)
        se = (pred - target) ** 2
        loss = (se * weights).sum() / (weights.sum() + 1e-12)
        if return_debug:
            return loss, {"weights": weights, "mask_soft": mask_soft, "border_margin": border_margin}
        return loss

class PhysicallyInformedAE(pl.LightningModule):
    def __init__(
        self,
        n_points: int,
        param_names: List[str],
        poly_freq_CH4,
        transitions_dict,
        mlp_dropout: float = 0.10,
        refiner_mlp_dropout: float = 0.10,

        # --- optims & pondérations globales ---
        lr: float = 1e-4,
        # --- têtes de sortie ---
        head_mode: str = "multi",
        predict_params: Optional[List[str]] = None,
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
        # --- reconstruction / pertes (existantes) ---
        recon_max1: bool = False,
        corr_mode: str = "savgol",
        corr_savgol_win: int = 11,
        corr_savgol_poly: int = 3,
        huber_beta: float = 0.002,
        weight_mf: float = 2.0,
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
        # --- PHYSIQUE / TIPS ---
        tipspy: Tips2021QTpy | None = None,
        use_relobralo_top: bool = True,     # ← NOUVEAU: activer ReLoBRaLo pour losses principales

        # --- débruitage ---
        use_denoiser: bool = False,
        denoiser_lr: float = 1e-4,
        denoiser_width: int = 64,

        # --- POIDS DE PERTES (CONFIGURATION MINIMALE) ---
    # --- POIDS DE PERTES (CONFIGURATION COMPLÈTE) ---
        w_pw_raw: float = 1.0,
        w_spectral_angle: float = 0.5,
        w_peak: float = 0.5,
        w_params: float = 1.0,              # ← AJOUTÉ
        pw_cfg: dict | None = None,

        scheduler_T_0: int = 10,
        scheduler_T_mult: float = 1.2,
        scheduler_eta_min: float = 1e-9,
        scheduler_decay_factor: float = 0.75,
        scheduler_warmup_epochs: int = 5,



    ):
        super().__init__()

        self.param_names = list(param_names); self.n_params = len(self.param_names)
        self.n_points = n_points
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict
        self.tipspy = tipspy
        self.save_hyperparameters(ignore=["transitions_dict", "poly_freq_CH4", "tipspy"])
        self.mlp_dropout = float(mlp_dropout)
        self.refiner_mlp_dropout = float(refiner_mlp_dropout)

        # --- optims / pondérations globales ---
        self.lr = float(lr)
        self.huber_beta  = float(huber_beta)

        # --- raffinement ---
        self.refine_steps = int(refine_steps)
        self.refine_target = refine_target.lower(); assert self.refine_target in {"noisy", "clean"}
        self.refine_warmup_epochs = int(refine_warmup_epochs)
        self.freeze_base_epochs   = int(freeze_base_epochs)
        self.base_lr    = float(base_lr)    if base_lr    is not None else self.lr
        self.refiner_lr = float(refiner_lr) if refiner_lr is not None else self.lr
        self._froze_base = False

        self.stage3_lr_shrink   = float(stage3_lr_shrink)
        self.stage3_refine_steps= stage3_refine_steps
        self.stage3_delta_scale = stage3_delta_scale

        # --- pertes / métriques ---
        self.weight_mf = float(weight_mf)
        self.corr_mode = str(corr_mode).lower()
        self.corr_savgol_win  = int(corr_savgol_win)
        self.corr_savgol_poly = int(corr_savgol_poly)
        self.recon_max1 = bool(recon_max1)

        self.w_pw_raw = float(w_pw_raw)
        self.w_spectral_angle = float(w_spectral_angle)
        self.w_peak = float(w_peak)
        self.w_params = float(w_params)              # ← AJOUTÉ
        self.use_relobralo_top = bool(use_relobralo_top) 
        
        # --- instanciation des modules de loss ---
        self.spectral_angle_loss = SpectralAngleLoss(eps=1e-8)
  

        # ✅ PeakWeightedMSELoss (torch-only), config par défaut + override via pw_cfg
        _pw_default = dict(
            peak_weight=4.0, baseline_weight=1.0,
            pre_smooth_sigma=1.3, salience_smooth_sigma=1.8,
            peak_kind="min", curv_scale_k=2.2,
            border_policy="taper", border_extra_margin=2,
            weight_normalize="mean", renorm_after_smoothing=True,
            salience_gamma=0.9,
            spread_kind="gaussian", spread_sigma=2.5, spread_kernel=11
        )
        if pw_cfg is not None:
            _pw_default.update(pw_cfg)
        self.peak_weighted_loss = PeakWeightedMSELoss(**_pw_default)



        # --- débruiteur (optionnel) ---
        self.use_denoiser = bool(use_denoiser)
        self.denoiser_lr = float(denoiser_lr)
        self.denoiser = Denoiser1D(in_ch=1, base_ch=int(denoiser_width), depth=6)

        # --- paramètres à prédire / fournis ---
        if predict_params is None:
            predict_params = self.param_names
        self.predict_params = list(predict_params)
        self.provided_params = [p for p in self.param_names if p not in self.predict_params]
        unknown = set(self.predict_params) - set(self.param_names)
        assert not unknown, f"Paramètres inconnus: {unknown}"
        assert len(self.predict_params) >= 1

        self.name_to_idx = {n: i for i, n in enumerate(self.param_names)}
        self.predict_idx = [self.name_to_idx[p] for p in self.predict_params]
        self.provided_idx = [self.name_to_idx[p] for p in self.provided_params]

        self.scheduler_T_0 = int(scheduler_T_0)
        self.scheduler_T_mult = float(scheduler_T_mult)
        self.scheduler_eta_min = float(scheduler_eta_min)
        self.scheduler_decay_factor = float(scheduler_decay_factor)
        self.scheduler_warmup_epochs = int(scheduler_warmup_epochs)

        # ===== Backbone EfficientNet 1D =====
        self.backbone = EfficientNetEncoder(
            in_channels=1,
            variant=backbone_variant,
            width_mult=backbone_width_mult,
            depth_mult=backbone_depth_mult,
            se_ratio=backbone_se_ratio,
            drop_path_rate=backbone_drop_path,
            stem_channels=backbone_stem_channels,
        )

        # ===== Backbone déjà construit au-dessus =====
        self.head_mode = str(head_mode).lower()
        assert self.head_mode in {"single", "multi"}, f"head_mode invalide: {self.head_mode}"

        # AvgMax pooling
        self.feature_head = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
        feat_dim = 2 * self.backbone.feat_dim
        hidden   = feat_dim // 2

        def _pool_features(z: torch.Tensor) -> torch.Tensor:
            a = self.feature_head[0](z).flatten(1)
            m = self.feature_head[1](z).flatten(1)
            return torch.cat([a, m], dim=1)
        self._pool_features = _pool_features

        # --- shared head: skip + gating encapsulé dans un nn.Module ---
        class GatedSharedHead(nn.Module):
            def __init__(self, feat_dim: int, hidden: int, p_drop: float):
                super().__init__()
                self.short_head = nn.Linear(feat_dim, hidden)
                self.deep_head = nn.Sequential(
                    nn.Linear(feat_dim, hidden),
                    nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p_drop),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p_drop),
                )
                self.gate_head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                g = self.gate_head(x)
                return g * self.deep_head(x) + (1 - g) * self.short_head(x)

        self.shared_head = GatedSharedHead(feat_dim, hidden, p_drop=self.mlp_dropout)

        # --- têtes de sortie (assure l’existence des deux attributs) ---
        if self.head_mode == "single":
            self.out_head = nn.Linear(hidden, len(self.predict_params))
            self.out_heads = None
        else:
            self.out_heads = nn.ModuleDict({pname: nn.Linear(hidden, 1) for pname in self.predict_params})
            self.out_head = None


            
        # ===== Raffineurs B/C/D =====
# ===== Raffineurs B/C/D (VERSION SIMPLE) =====
        base_refiner = EfficientNetRefiner(
            m_params=len(self.predict_params),
            delta_scale=refine_delta_scale,
            encoder_variant=refiner_variant,
            encoder_width_mult=refiner_width_mult,
            encoder_depth_mult=refiner_depth_mult,
            encoder_stem_channels=refiner_stem_channels,
            encoder_drop_path=refiner_drop_path,
            encoder_se_ratio=refiner_se_ratio,
            feature_pool=refiner_feature_pool,
            hidden_dim=max(64, int(self.backbone.feat_dim * refiner_shared_hidden_scale)),
            mlp_dropout=self.refiner_mlp_dropout
        )
        self.refiner = base_refiner
        self.cascade_stages = 3

        extra_refiners = [EfficientNetRefiner(
                m_params=len(self.predict_params),
                delta_scale=refine_delta_scale,
                encoder_variant=refiner_variant,
                encoder_width_mult=refiner_width_mult,
                encoder_depth_mult=refiner_depth_mult,
                encoder_stem_channels=refiner_stem_channels,
                encoder_drop_path=refiner_drop_path,
                encoder_se_ratio=refiner_se_ratio,
                feature_pool=refiner_feature_pool,
                hidden_dim=max(64, int(self.backbone.feat_dim * refiner_shared_hidden_scale)),
                mlp_dropout=self.refiner_mlp_dropout
            )
            for _ in range(self.cascade_stages - 1)
        ]
        self.refiners = nn.ModuleList([base_refiner] + extra_refiners)

        # ===== ReLoBRaLo =====
        self.loss_names_params = [f"param_{p}" for p in self.predict_params]
        self.relo_params = ReLoBRaLoLoss(num_losses=len(self.predict_params), alpha=0.9, tau=1.0, history_len=10)
        if self.use_relobralo_top:
            self.loss_names_top = ["phys_pointwise", "phys_spectral", "phys_peak", "param_group"]
            self.relo_top = ReLoBRaLoLoss(
                loss_names=self.loss_names_top, 
                alpha=0.9,      # Momentum pour EMA
                tau=1.0,        # Température pour softmax
                history_len=10  # Historique pour stabilité
            )
        else:
            self.relo_top = None 

        # ----- Stage override -----
        self._override_stage: Optional[str] = None
        self._override_refine_steps: Optional[int] = None
        self._override_delta_scale: Optional[float] = None

        # ----- Masques de raffinement -----
        def _target_mask(names: List[str]) -> torch.Tensor:
            m = torch.zeros(len(self.predict_params), dtype=torch.float32)
            idxs = [self.predict_params.index(n) for n in names if n in self.predict_params]
            if idxs:
                m[torch.tensor(idxs, dtype=torch.long)] = 1.0
            return m

        base_targets = ["sig0", "dsig", "mf_CH4", "mf_H2O"]
        pt_targets   = base_targets + ["P", "T"]

        self.register_buffer("refine_mask_base", _target_mask(base_targets))
        self.register_buffer("refine_mask_with_PT", _target_mask(pt_targets))

    # ==== Helpers baseline poly3 sur bords ====
    @staticmethod
    def _design_poly3(n: int, device, dtype):
        x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        xc = x - x.mean()
        X = torch.stack([torch.ones_like(xc), xc, xc**2, xc**3], dim=1)  # [N,4]
        return X

    @classmethod
    def _baseline_poly3_from_edges(cls, resid: torch.Tensor, left_frac: float = 0.20, right_start: float = 0.75):
        B, N = resid.shape
        device, dtype = resid.device, resid.dtype
        Xfull = cls._design_poly3(N, device, dtype)                # [N,4]
        iL = torch.arange(0, max(1, int(N*left_frac)), device=device)
        iR = torch.arange(int(N*right_start), N, device=device)
        idx = torch.cat([iL, iR], dim=0)                           # [M]
        X = Xfull[idx]                                             # [M,4]
        Xt = X.t()                                                 # [4,M]
        XtX = Xt @ X                                               # [4,4]
        XtX = XtX + 1e-6 * torch.eye(4, device=device, dtype=dtype)
        P = torch.linalg.solve(XtX, Xt)                            # [4,M]
        y_edges = resid[:, idx]                                    # [B,M]
        coeff = (P @ y_edges.T).T                                  # [B,4]
        baseline = (Xfull @ coeff.transpose(0,1)).transpose(0,1)   # [B,N]
        resid_corr = resid - baseline
        return resid_corr, baseline

    def set_stage_mode(self, mode: Optional[str], refine_steps: Optional[int]=None, delta_scale: Optional[float]=None):
        if mode is not None:
            mode = mode.upper(); assert mode in {'A','B1','B2','DEN'}
        self._override_stage = mode
        self._override_refine_steps = refine_steps
        self._override_delta_scale = delta_scale
        if delta_scale is not None:
            for r in self.refiners: r.delta_scale = float(delta_scale)
        if refine_steps is not None:
            self.refine_steps = int(refine_steps)

    # ---- utils tête ----
    def _predict_params_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.shared_head(feat)
        if self.head_mode == "multi" and self.out_heads is not None:
            outs = []
            for name in self.predict_params:
                y = self.out_heads[name](h)
                outs.append(torch.sigmoid(y))
            return torch.cat(outs, dim=1)
        elif self.head_mode == "single" and self.out_head is not None:
            y = self.out_head(h)
            return torch.sigmoid(y)
        else:
            raise RuntimeError("Configuration des têtes incohérente avec head_mode.")

        
    def encode(self, spectra: torch.Tensor, pooled: bool=True, detach: bool=False):
        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat = self._pool_features(latent) if pooled else latent
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
            """
            Reconstruit le spectre depuis les paramètres physiques dénormalisés.
            VERSION CORRIGÉE : Gère baseline0 via LOWESS quand il n'est pas prédit.
            """
            B = y_phys_full.shape[0]
            p = {k: y_phys_full[:, i] for i, k in enumerate(self.param_names)}
            
            v_grid_idx = torch.arange(self.n_points, dtype=torch.float64, device=device)
            
            # Préparer les fractions molaires
            mf_dict = {}
            for mol in self.transitions_dict.keys():
                key = f"mf_{mol}"
                mf_dict[mol] = p[key] if key in p else torch.zeros_like(p['P'])
            
            # ============================================================
            # CORRECTION : Gestion de baseline0
            # ============================================================
            
            # Vérifier si baseline0 est dans les paramètres prédits
            if 'baseline0' in self.predict_params:
                # CAS 1 : baseline0 est PRÉDIT - on l'utilise directement
                b0_idx = self.name_to_idx['baseline0']
                baseline_coeffs = y_phys_full[:, b0_idx:b0_idx+3]
                
            else:
                # CAS 2 : baseline0 est CALCULÉ via LOWESS (c'est le cas actuel)
                b1_idx = self.name_to_idx['baseline1']
                b2_idx = self.name_to_idx['baseline2']
                
                # Utiliser baseline0 = 1.0 (sera normalisé après)
                baseline_coeffs = torch.stack([
                    torch.ones(B, device=device, dtype=torch.float64),  # b0 = 1.0
                    y_phys_full[:, b1_idx].to(torch.float64),           # b1 prédit
                    y_phys_full[:, b2_idx].to(torch.float64)            # b2 prédit
                ], dim=1)
            
            # ============================================================
            # Génération du spectre
            # ============================================================
            spectra, _ = batch_physics_forward_multimol_vgrid(
                p['sig0'], p['dsig'], self.poly_freq_CH4, v_grid_idx,
                baseline_coeffs, self.transitions_dict, p['P'], p['T'], mf_dict,
                tipspy=self.tipspy, device=device
            )
            spectra = spectra.to(torch.float32)
            
            # ============================================================
            # Normalisation finale par LOWESS
            # ============================================================
            scale_recon = lowess_value(spectra, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)
            spectra = spectra / scale_recon
            
            return spectra

    # ---------- Savitzky–Golay derivative (torch) ----------
    def _savgol_coeffs(self, window_length: int, polyorder: int, deriv: int = 1, delta: float = 1.0, device=None, dtype=torch.float64) -> torch.Tensor:
        assert deriv >= 0
        W = int(window_length)
        P = int(polyorder)
        if W % 2 == 0: W += 1
        if W < 3: W = 3
        if P >= W: P = W - 1
        m = (W - 1) // 2
        dev = device or self.device
        x = torch.arange(-m, m + 1, device=dev, dtype=dtype)
        A = torch.stack([x**j for j in range(P + 1)], dim=1)
        pinv = torch.linalg.pinv(A)
        coeff = math.factorial(deriv) * pinv[deriv, :] / (delta ** deriv)
        return coeff.to(dtype=torch.float32)

    def _savgol_deriv(self, y: torch.Tensor, window_length: int, polyorder: int, deriv: int = 1) -> torch.Tensor:
        B, N = y.shape
        W = int(window_length)
        if W % 2 == 0: W += 1
        if W > N:
            W = N if (N % 2 == 1) else (N - 1)
        W = max(W, 3)
        P = min(int(polyorder), W - 1)
        coeff = self._savgol_coeffs(W, P, deriv=deriv, device=y.device, dtype=torch.float64)  # [W]
        coeff = coeff.view(1, 1, -1)
        pad = (W - 1) // 2
        y1 = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
        out = F.conv1d(y1, coeff).squeeze(1)
        return out

    def _pearson_corr_basic(self, y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        y_hat = y_hat.float(); y = y.float()
        y_hat_c = y_hat - y_hat.mean(dim=1, keepdim=True)
        y_c     = y - y.mean(dim=1, keepdim=True)
        num = (y_hat_c * y_c).sum(dim=1)
        den = torch.sqrt((y_hat_c.pow(2).sum(dim=1) + eps) * (y_c.pow(2).sum(dim=1) + eps))
        corr = num / den
        return (1.0 - corr).mean()

    @staticmethod
    def _to_pdf(y: torch.Tensor, smooth_win: int = 0, eps: float = 1e-12) -> torch.Tensor:
        if smooth_win and smooth_win > 1:
            pad = (smooth_win - 1) // 2
            k = torch.ones(1, 1, smooth_win, device=y.device, dtype=y.dtype) / float(smooth_win)
            yy = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
            y = F.conv1d(yy, k).squeeze(1)
        y = y.clamp_min(0)
        return y / (y.sum(dim=1, keepdim=True) + eps)

    @staticmethod
    def _kl(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        a = a.clamp_min(eps); b = b.clamp_min(eps)
        return (a * (a.log() - b.log())).sum(dim=1)

    def _js(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        m = 0.5 * (p + q)
        return 0.5 * self._kl(p, m, eps) + 0.5 * self._kl(q, m, eps)


    def _common_step(self, batch, step_name: str):
        """Step commun pour train/val/test."""
        noisy, clean, params_true_norm = batch['noisy_spectra'], batch['clean_spectra'], batch['params']
        scale = batch.get('scale', None)
        if scale is not None:
            scale = scale.to(clean.device)

        # ========== FORWARD PASS BASE MODEL A ==========
        latent, _ = self.backbone(noisy.unsqueeze(1))
        feat_shared = self._pool_features(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared)

        # Paramètres fournis
        if len(self.provided_params) > 0:
            provided_cols = [params_true_norm[:, self.name_to_idx[n]] for n in self.provided_params]
            provided_norm_tensor = torch.stack(provided_cols, dim=1)
            provided_phys_tensor = self._denorm_params_subset(provided_norm_tensor, self.provided_params)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((noisy.shape[0], 0))

        # Récupérer les paramètres expérimentaux
        use_exp_params_in_A = getattr(self, '_use_exp_params_in_A', [])

        # ========== STAGE A: Reconstruction ==========
        spectra_recon = None
        
        if self._override_stage == 'A' and len(use_exp_params_in_A) > 0:
            params_for_recon_norm = params_pred_norm.clone()
            for param_name in use_exp_params_in_A:
                if param_name in self.predict_params:
                    idx = self.predict_params.index(param_name)
                    params_for_recon_norm[:, idx] = params_true_norm[:, self.name_to_idx[param_name]]
            
            pred_phys_recon = self._denorm_params_subset(params_for_recon_norm, self.predict_params)
            y_phys_full_recon = self._compose_full_phys(pred_phys_recon, provided_phys_tensor)
            spectra_recon = self._physics_reconstruction(y_phys_full_recon, clean.device, scale=None)

        # ========== RAFFINEMENT ==========
        e = self.current_epoch
        effective_refine_steps = 0 if e < self.refine_warmup_epochs else self.refine_steps
        
        if self._override_stage is not None:
            if self._override_stage == 'A':   
                effective_refine_steps = 0
            elif self._override_stage in ('B1', 'B2', 'C', 'D'): 
                effective_refine_steps = self.refine_steps

        target_for_resid = noisy if self.refine_target == "noisy" else clean
        n_stages = min(effective_refine_steps, len(self.refiners))
        mask_now = self.refine_mask_base.to(clean.device, dtype=params_pred_norm.dtype)

        use_exp_params_for_resid = getattr(self, '_use_exp_params_for_resid', [])
        
        for k in range(n_stages):
            if len(use_exp_params_for_resid) > 0:
                params_for_resid_norm = params_pred_norm.clone()
                for param_name in use_exp_params_for_resid:
                    if param_name in self.predict_params:
                        idx = self.predict_params.index(param_name)
                        params_for_resid_norm[:, idx] = params_true_norm[:, self.name_to_idx[param_name]]
                
                pred_phys_for_resid = self._denorm_params_subset(params_for_resid_norm, self.predict_params)
                y_phys_full_for_resid = self._compose_full_phys(pred_phys_for_resid, provided_phys_tensor)
                spectra_recon_for_resid = self._physics_reconstruction(y_phys_full_for_resid, clean.device, scale=None)
                resid = spectra_recon_for_resid - target_for_resid
            else:
                pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
                y_phys_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
                spectra_recon = self._physics_reconstruction(y_phys_full, clean.device, scale=None)
                resid = spectra_recon - target_for_resid

            resid_corr, _ = self._baseline_poly3_from_edges(resid, left_frac=0.20, right_start=0.75)

            resid_for_refiner = resid_corr
            if self.use_denoiser and self._override_stage in (None, 'B1', 'B2'):
                with torch.no_grad():
                    resid_for_refiner = self.denoiser(resid_corr)

            delta = self.refiners[k](resid_for_refiner)
            params_pred_norm = (params_pred_norm + delta * mask_now).clamp(1e-4, 1-1e-4)

        # ========== RECONSTRUCTION FINALE ==========
        if spectra_recon is None:
            pred_phys   = self._denorm_params_subset(params_pred_norm, self.predict_params)
            y_phys_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
            spectra_recon = self._physics_reconstruction(y_phys_full, clean.device, scale=None)

        # ========== CALCUL DES LOSSES BRUTES ==========
        yh = spectra_recon
        yt = clean

        # Losses spectrales (brutes, sans poids)
        loss_pointwise_raw = F.mse_loss(yh, yt)
        loss_spectral_raw = self.spectral_angle_loss(yh, yt)
        
        loss_peak_raw = torch.tensor(0.0, device=clean.device)
        if self.w_peak > 0:
            loss_peak_raw = self.peak_weighted_loss(yh, yt)

        # ✅ Loss params (brute)
        per_param_losses = []
        per_param_names = []
        
        for j, name in enumerate(self.predict_params):
            if len(use_exp_params_in_A) > 0 and name in use_exp_params_in_A:
                continue
            
            true_j = params_true_norm[:, self.name_to_idx[name]]
            mult = self.weight_mf if name in ("mf_CH4", "mf_H2O") else 1.0
            lp = mult * F.mse_loss(params_pred_norm[:, j], true_j)
            per_param_losses.append(lp)
            per_param_names.append(name)

        loss_params_raw = torch.mean(torch.stack(per_param_losses)) if per_param_losses else torch.tensor(0.0, device=clean.device)

        # ========== APPLICATION DES POIDS (avec ou sans ReLoBRaLo) ==========
        if self.use_relobralo_top and step_name == "train" and self.relo_top is not None:
            # ✅ ReLoBRaLo équilibre automatiquement
            # Créer un tenseur avec les losses dans l'ordre des loss_names_top
            losses_tensor = torch.stack([
                loss_pointwise_raw,
                loss_spectral_raw,
                loss_peak_raw,
                loss_params_raw,
            ])  # Shape: [4]
            
            # Appeler ReLoBRaLo (retourne loss totale ET poids)
            loss_main = self.relo_top(losses_tensor)
            
            # Récupérer les poids pour logging
            # ReLoBRaLo stocke les derniers poids dans self.relo_top.last_weights
            if hasattr(self.relo_top, 'last_weights'):
                weights = self.relo_top.last_weights
            else:
                # Fallback si pas de last_weights
                weights = torch.ones_like(losses_tensor) / len(losses_tensor)
            
            # Pour logging (losses pondérées individuelles)
            loss_pointwise = weights[0].item() * loss_pointwise_raw
            loss_spectral = weights[1].item() * loss_spectral_raw
            loss_peak = weights[2].item() * loss_peak_raw
            loss_params = weights[3].item() * loss_params_raw
            
            # Logger les poids adaptatifs
            self.log("relo_weight_phys_pointwise", weights[0].item(), on_epoch=True, sync_dist=True)
            self.log("relo_weight_phys_spectral", weights[1].item(), on_epoch=True, sync_dist=True)
            self.log("relo_weight_phys_peak", weights[2].item(), on_epoch=True, sync_dist=True)
            self.log("relo_weight_param_group", weights[3].item(), on_epoch=True, sync_dist=True)
            
        else:
            # ✅ Poids fixes (pour validation ou si ReLoBRaLo désactivé)
            loss_pointwise = self.w_pw_raw * loss_pointwise_raw
            loss_spectral = self.w_spectral_angle * loss_spectral_raw
            loss_peak = self.w_peak * loss_peak_raw
            loss_params = self.w_params * loss_params_raw
            
            loss_main = loss_pointwise + loss_spectral + loss_peak + loss_params

        # ========== DENOISER (si applicable) ==========
        if self._override_stage == 'DEN':
            resid_den = spectra_recon - noisy
            resid_den_corr, _ = self._baseline_poly3_from_edges(resid_den, left_frac=0.20, right_start=0.75)
            resid_clean = spectra_recon - clean
            resid_clean_corr, _ = self._baseline_poly3_from_edges(resid_clean, left_frac=0.20, right_start=0.75)
            resid_hat = self.denoiser(resid_den_corr)
            denoiser_loss = F.mse_loss(resid_hat, resid_clean_corr)
            loss = denoiser_loss
            self.log(f"{step_name}_loss_denoiser", denoiser_loss, on_epoch=True, sync_dist=True)
        else:
            loss = loss_main

        # ========== LOGGING ==========
        self.log(f"{step_name}_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        
        # Losses pondérées
        self.log(f"{step_name}_loss_pointwise", loss_pointwise, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_spectral", loss_spectral, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_peak", loss_peak, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_params", loss_params, on_epoch=True, sync_dist=True)
        
        # ✅ Losses BRUTES (pour voir les magnitudes réelles)
        self.log(f"{step_name}_loss_pointwise_raw", loss_pointwise_raw, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_spectral_raw", loss_spectral_raw, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_peak_raw", loss_peak_raw, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_params_raw", loss_params_raw, on_epoch=True, sync_dist=True)
        
        if len(use_exp_params_in_A) > 0:
            self.log(f"{step_name}_n_exp_params_A", float(len(use_exp_params_in_A)), on_epoch=True, sync_dist=True)
        
        if len(use_exp_params_for_resid) > 0:
            self.log(f"{step_name}_n_exp_params_resid", float(len(use_exp_params_for_resid)), on_epoch=True, sync_dist=True)

        # Logger les losses individuelles par paramètre
        if len(per_param_losses) > 0:
            for name, lp in zip(per_param_names, per_param_losses):
                self.log(f"{step_name}_param_{name}", lp, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def on_train_epoch_start(self):
        def set_trainable(mod, train: bool):
            if mod is None: return
            mods = list(mod) if isinstance(mod, nn.ModuleList) else [mod]
            for m in mods:
                if m is None: continue
                for p in m.parameters():
                    p.requires_grad_(train)

        # --------- Stages explicites ----------
        if self._override_stage is not None:
            st = self._override_stage

            if st == 'A':
                set_trainable(self.backbone, True)
                set_trainable(self.shared_head, True)
                set_trainable(getattr(self, "out_head", None), True)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items(): set_trainable(head, True)
                set_trainable(self.refiners, False)
                set_trainable(self.denoiser, False)   # pas d’entraînement du denoiser pendant A
                return

            if st == 'DEN':
                set_trainable(self.backbone, False)
                set_trainable(self.shared_head, False)
                set_trainable(getattr(self, "out_head", None), False)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items(): set_trainable(head, False)
                set_trainable(self.refiners, False)
                set_trainable(self.denoiser, True)    # on n’entraîne QUE le denoiser
                return

            if st == 'B1':
                set_trainable(self.backbone, False)
                set_trainable(self.shared_head, False)
                set_trainable(getattr(self, "out_head", None), False)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items(): set_trainable(head, False)
                set_trainable(self.refiners, True)    # entraîne les raffineurs
                set_trainable(self.denoiser, False)   # denoiser gelé
                return

            if st == 'B2':
                set_trainable(self.backbone, True)
                set_trainable(self.shared_head, True)
                set_trainable(getattr(self, "out_head", None), True)
                if hasattr(self, "out_heads"):
                    for _n, head in self.out_heads.items(): set_trainable(head, True)
                set_trainable(self.refiners, True)
                set_trainable(self.denoiser, False)   # (mettez True si vous voulez aussi le FT)
                return

        # --------- Fallback 3 phases (A warmup → B → B2) ----------
        e = self.current_epoch
        stage3_start = self.refine_warmup_epochs + self.freeze_base_epochs

        if e < self.refine_warmup_epochs:
            self._set_requires_grad(self.refiners, False)
            self._set_requires_grad(
                [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None)],
                True
            )
            self._froze_base = False
        elif e < stage3_start:
            if not self._froze_base:
                self._set_requires_grad(
                    [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None)],
                    False
                )
                self._set_requires_grad(self.refiners, True)
                self._froze_base = True
        else:
            self._set_requires_grad(
                [self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.refiners],
                True
            )
            if e == stage3_start:
                if hasattr(self.trainer, "optimizers") and len(self.trainer.optimizers) > 0:
                    opt = self.trainer.optimizers[0]
                    for pg in opt.param_groups: pg["lr"] *= self.stage3_lr_shrink
                if self.stage3_refine_steps is not None: self.refine_steps = int(self.stage3_refine_steps)
                if self.stage3_delta_scale is not None:
                    for r in self.refiners: r.delta_scale = float(self.stage3_delta_scale)


    def _set_requires_grad(self, modules, flag: bool):
        if modules is None: return
        if isinstance(modules, nn.ModuleList):
            modules = list(modules)
        if not isinstance(modules, (list, tuple)): modules = [modules]
        for m in modules:
            if m is None: continue
            for p in m.parameters(): p.requires_grad_(flag)

    def configure_optimizers(self):
        """Configure optimizer et scheduler personnalisé"""
        
        # ========== Paramètres à optimiser ==========
        # === BASE PARAMS (backbone + shared head + heads) ===
        base_params = list(self.backbone.parameters()) + list(self.shared_head.parameters())

        if getattr(self, "out_head", None) is not None:   # ← au lieu de hasattr(...)
            base_params += list(self.out_head.parameters())

        if getattr(self, "out_heads", None) is not None:  # ← au lieu de hasattr(...)
            base_params += list(self.out_heads.parameters())

        
        refiner_params = list(self.refiners.parameters())

        param_groups = [
            {"params": base_params,    "lr": float(getattr(self, "base_lr", self.lr))},
            {"params": refiner_params, "lr": float(getattr(self, "refiner_lr", self.lr))},
        ]
        
        if getattr(self, "use_denoiser", False):
            param_groups.append({
                "params": self.denoiser.parameters(), 
                "lr": float(self.denoiser_lr)
            })

        # ========== Optimizer ==========
        opt_name = getattr(self.hparams, "optimizer", "adamw").lower()
        weight_decay = getattr(self, "weight_decay", 1e-4)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        elif opt_name == "lion":
            from lion_pytorch import Lion
            betas = getattr(self.hparams, "betas", (0.9, 0.99))
            optimizer = Lion(param_groups, betas=betas, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # ========== Scheduler personnalisé ==========
        scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
            optimizer,
            T_0=self.scheduler_T_0,
            T_mult=self.scheduler_T_mult,
            eta_min=self.scheduler_eta_min,
            decay_factor=self.scheduler_decay_factor,
            warmup_epochs=self.scheduler_warmup_epochs,
            verbose=True  # ← Afficher les restarts dans les logs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # ← Important : step à chaque epoch
                "frequency": 1,
            }
        }
    
    @torch.no_grad()
    def infer(
        self,
        spectra: torch.Tensor,
        provided_phys: dict,
        *,
        refine: bool = True,
        resid_target: str = "input",
        scale: Optional[torch.Tensor] = None,
        recon_PT: str = "pred",             # "pred" | "exp"
        Pexp: Optional[torch.Tensor] = None,
        Texp: Optional[torch.Tensor] = None,
        cascade_stages_override: Optional[int] = None,
    ):
        self.eval()
        device = spectra.device
        B = spectra.shape[0]

        if recon_PT not in ("pred", "exp"):
            raise ValueError("recon_PT doit être 'pred' ou 'exp'.")
        if recon_PT == "exp":
            if Pexp is None or Texp is None:
                raise ValueError("En mode recon_PT='exp', fournir Pexp et Texp.")
            Pexp = torch.as_tensor(Pexp, device=device, dtype=torch.float32).view(B)
            Texp = torch.as_tensor(Texp, device=device, dtype=torch.float32).view(B)

        missing = [n for n in self.provided_params if n not in provided_phys]
        assert not missing, f"Manque des paramètres fournis: {missing}"

        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat_shared = self._pool_features(latent)        
        params_pred_norm = self._predict_params_from_features(feat_shared)

        if len(self.provided_params) > 0:
            provided_list = [provided_phys[n].to(device) for n in self.provided_params]
            provided_phys_tensor = torch.stack(provided_list, dim=1)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((B, 0))

        spectra_target = spectra if resid_target in ("input", "noisy") else None
        scale_est = lowess_value(spectra, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)

        def _compose_full_with_PT_override(pred_norm_subset: torch.Tensor) -> torch.Tensor:
            pred_phys_subset = self._denorm_params_subset(pred_norm_subset, self.predict_params)
            y_full = self._compose_full_phys(pred_phys_subset, provided_phys_tensor)
            if recon_PT == "exp":
                idxP = self.name_to_idx.get("P", None)
                idxT = self.name_to_idx.get("T", None)
                if idxP is not None: y_full[:, idxP] = Pexp
                if idxT is not None: y_full[:, idxT] = Texp
            return y_full

        # masque de raffinement
        if recon_PT == "exp":
            mask_now = self.refine_mask_with_PT.to(device, dtype=params_pred_norm.dtype)
        else:
            mask_now = self.refine_mask_base.to(device, dtype=params_pred_norm.dtype)

        # cascade
        n_stages = cascade_stages_override if cascade_stages_override is not None else self.cascade_stages
        n_stages = max(0, min(int(n_stages), len(self.refiners)))

        if refine and n_stages > 0:
            for k in range(n_stages):
                y_full_k = _compose_full_with_PT_override(params_pred_norm)
                recon_k  = self._physics_reconstruction(y_full_k, device, scale=None)
                if spectra_target is None:
                    break
                resid = recon_k - spectra_target
                resid_corr, _ = self._baseline_poly3_from_edges(resid, left_frac=0.20, right_start=0.75)

                resid_for_refiner = resid_corr
                if self.use_denoiser:
                    resid_for_refiner = self.denoiser(resid_corr)
                    
                delta_k = self.refiners[k](resid_for_refiner)

                params_pred_norm = params_pred_norm.add(delta_k * mask_now)

        y_full_final = _compose_full_with_PT_override(params_pred_norm)
        recon_final  = self._physics_reconstruction(y_full_final, device, scale=None)

        return {
            "params_pred_norm": params_pred_norm,
            "y_phys_full": y_full_final,
            "spectra_recon": recon_final,
            "norm_scale": scale_est,
        }

# ============================================================
# 8) Callbacks visu & epoch sync dataset
# ============================================================

@torch.no_grad()
def preview_peak_weights_on_val(model: PhysicallyInformedAE, val_loader, save_dir="./figs_local/weights_preview"):
    import numpy as np, os, matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = model.device

    # sample 5 clean spectra du val_loader
    specs = []
    for batch in val_loader:
        c = batch["clean_spectra"].to(device)
        for i in range(c.shape[0]):
            specs.append(c[i])
            if len(specs) == 5:
                break
        if len(specs) == 5:
            break
    if len(specs) == 0:
        print("⚠️ pas d'échantillon dans val_loader"); return
    Y = torch.stack(specs, dim=0)  # [5,N]
    X = Y.clone()

    # récupère les poids (en mode debug)
    try:
        _, dbg = model.peak_weighted_loss(X, Y, return_debug=True)
    except TypeError:
        # ancienne signature: forward(pred, target) sans debug
        W = model.peak_weighted_loss._detect(Y)[0]
        dbg = {"weights": W, "border_margin": 0}
    W = dbg["weights"].detach().cpu().numpy()   # [5,N]
    N = W.shape[1]
    x = np.arange(N)

    fig, axes = plt.subplots(5, 1, figsize=(16, 10), sharex=True)
    if not isinstance(axes, (list, np.ndarray)): axes = [axes]
    for i, ax in enumerate(axes):
        ax.fill_between(x, W[i], alpha=0.55, label=f'weights spec {i}')
        ax.axhline(model.peak_weighted_loss.peak_weight, color='red', ls='--', lw=1.5, label='peak_weight')
        ax.axhline(model.peak_weighted_loss.baseline_weight, color='blue', ls='--', lw=1.2, label='baseline_weight')
        # inverser l’axe (style spectro) si croissant
        ax.set_xlim(x.min(), x.max())
        if x[0] < x[-1]:
            ax.invert_xaxis()
        if i == 0:
            ax.legend(loc='upper right', ncols=3, fontsize=9)
        ax.grid(alpha=0.3, ls='--')
        ax.set_ylabel("weight")
    axes[-1].set_xlabel("Wavenumber index (descending)")
    fig.suptitle("Peak weights (derivative-based) on 5 validation spectra", fontweight='bold')
    out = os.path.join(save_dir, "peak_weights_5spectra.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Figure sauvegardée : {out}")
    plt.show()


# ============================================================
# FONCTION DE VISUALISATION DU SCHEDULER
# ============================================================

def plot_lr_scheduler_preview(
    scheduler,
    max_epochs: int,
    save_dir: str,
    filename: str = "lr_schedule_preview.png"
):
    """
    Génère et sauvegarde la visualisation du scheduler
    
    Args:
        scheduler: Instance du scheduler
        max_epochs: Nombre d'epochs à visualiser
        save_dir: Dossier de sauvegarde (ex: "./figs_local/stageA_P_T")
        filename: Nom du fichier
    """
    # Créer le dossier si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    # Obtenir les infos du schedule
    info = scheduler.get_schedule_info(max_epochs)
    
    epochs = np.array(info['epochs'])
    lrs = np.array(info['lrs'])
    restart_epochs = info['restart_epochs']
    warmup_end = info['warmup_end']
    
    # Figure principale
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # ========== Subplot 1: Schedule complet (log scale) ==========
    ax1 = axes[0, 0]
    ax1.plot(epochs, lrs, linewidth=2.5, color='tab:blue', label='Learning Rate')
    
    if warmup_end is not None and warmup_end >= 0:
        ax1.axvspan(0, warmup_end, alpha=0.2, color='green', label='Warmup')
        ax1.axvline(warmup_end, color='green', linestyle='--', alpha=0.7, linewidth=2)
    
    for i, restart_epoch in enumerate(restart_epochs):
        ax1.axvline(restart_epoch, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        if i < 8:
            ax1.text(restart_epoch, lrs[restart_epoch], f' R{i+1}', 
                    rotation=0, va='bottom', ha='left', fontsize=9, color='red', fontweight='bold')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Rate Schedule (Log Scale)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    
    info_text = f"T₀={scheduler.T_0}, T_mult={scheduler.T_mult}, decay={scheduler.decay_factor:.2f}, warmup={scheduler.warmup_epochs}"
    ax1.text(0.02, 0.10, info_text, transform=ax1.transAxes,
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='wheat', alpha=0.7))
    
    # ========== Subplot 2: Vue linéaire complète ==========
    ax2 = axes[0, 1]
    ax2.plot(epochs, lrs, linewidth=2, color='tab:orange', marker='o', markersize=3)
    
    if warmup_end is not None:
        ax2.axvline(warmup_end, color='green', linestyle='--', alpha=0.7, linewidth=2, label='End Warmup')
    
    for restart_epoch in restart_epochs:
        ax2.axvline(restart_epoch, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title(f'{max_epochs} Epochs (Linear Scale)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    
    # ========== Subplot 3: Distribution des LR ==========
    ax3 = axes[1, 0]
    ax3.hist(lrs, bins=50, color='tab:purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axvline(np.mean(lrs), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(lrs):.2e}')
    ax3.axvline(np.median(lrs), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(lrs):.2e}')
    
    ax3.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('LR Distribution', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # ========== Subplot 4: Stats par cycle ==========
    ax4 = axes[1, 1]
    
    cycle_starts = [warmup_end + 1 if warmup_end is not None else 0] + restart_epochs
    cycle_ends = restart_epochs + [max_epochs - 1]
    
    cycle_stats = []
    for i, (start, end) in enumerate(zip(cycle_starts, cycle_ends)):
        cycle_lrs = lrs[start:end+1]
        if len(cycle_lrs) > 0:
            cycle_stats.append({
                'cycle': i + 1,
                'duration': end - start + 1,
                'mean_lr': np.mean(cycle_lrs),
                'max_lr': np.max(cycle_lrs),
                'min_lr': np.min(cycle_lrs),
            })
    
    if len(cycle_stats) > 0:
        cycles = [s['cycle'] for s in cycle_stats]
        mean_lrs = [s['mean_lr'] for s in cycle_stats]
        max_lrs = [s['max_lr'] for s in cycle_stats]
        durations = [s['duration'] for s in cycle_stats]
        
        x = np.arange(len(cycles))
        width = 0.35
        
        ax4.bar(x - width/2, max_lrs, width, label='Max LR', color='tab:red', alpha=0.8, edgecolor='black')
        ax4.bar(x + width/2, mean_lrs, width, label='Mean LR', color='tab:blue', alpha=0.8, edgecolor='black')
        
        ax4.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax4.set_title('Max & Mean LR per Cycle', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'C{c}\n({d}ep)' for c, d in zip(cycles, durations)], fontsize=9)
        ax4.set_yscale('log')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Stats globales
    stats_text = [
        f"📊 STATISTIQUES ({max_epochs} epochs)",
        f"",
        f"Restarts      : {len(restart_epochs)}",
        f"LR initial    : {scheduler.base_lrs[0]:.2e}",
        f"LR moyen      : {np.mean(lrs):.2e}",
        f"LR final      : {lrs[-1]:.2e}",
        f"",
        f"Decay cumulé  : {scheduler.decay_factor ** len(restart_epochs):.3f}",
    ]
    
    fig.text(0.99, 0.01, '\n'.join(stats_text),
            fontsize=10, ha='right', va='bottom',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.5, edgecolor='navy'))
    
    plt.suptitle(f'Learning Rate Schedule Preview - {max_epochs} Epochs', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    # Sauvegarder
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ LR Schedule preview sauvegardé: {save_path}")
    print(f"   - {len(restart_epochs)} restarts")
    if len(cycle_stats) > 0:
        print(f"   - LR max final: {max_lrs[-1]:.2e}")
        print(f"   - Decay cumulatif: {scheduler.decay_factor ** len(restart_epochs):.3f}")
    
    return cycle_stats

class LossCurvePlotCallback(pl.Callback):
    def __init__(self, save_path: str = "./figs_loss/loss_curves.png"):
        super().__init__()
        self.save_path = save_path
        self._history: dict[int, dict] = {}

    def _record(self, epoch: int, metrics: dict):
        if epoch not in self._history:
            self._history[epoch] = {}
        for key in [
            'train_loss','val_loss',
            'train_loss_pointwise','val_loss_pointwise',
            'train_loss_spectral','val_loss_spectral',
            'train_loss_peak','val_loss_peak',
            'train_loss_params','val_loss_params'
        ]:
            if key in metrics:
                v = metrics[key]
                if hasattr(v, 'detach'): v = v.detach().cpu()
                try: self._history[epoch][key] = float(v)
                except: pass
        self._plot()

    def _plot(self):
        if not self._history: return
        import numpy as np, matplotlib.pyplot as plt

        epochs_sorted = sorted(self._history.keys())
        E = [e+1 for e in epochs_sorted]
        get = lambda k,e: self._history[e].get(k, None)

        series = {name: [get(name,e) for e in epochs_sorted] for name in [
            'train_loss','val_loss',
            'train_loss_pointwise','val_loss_pointwise',
            'train_loss_spectral','val_loss_spectral',
            'train_loss_peak','val_loss_peak',
            'train_loss_params','val_loss_params'
        ]}

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Courbes d'entraînement", fontsize=14, fontweight='bold')
        ax = axes.flatten()

        # Totale
        if any(x is not None for x in series['train_loss']): ax[0].plot(E, series['train_loss'], marker='o', label='Train')
        if any(x is not None for x in series['val_loss']):   ax[0].plot(E, series['val_loss'], marker='s', label='Val')
        ax[0].set_title("Loss Totale"); ax[0].legend(); ax[0].grid(alpha=.3, ls='--')

        # Pointwise
        if any(x is not None for x in series['train_loss_pointwise']): ax[1].plot(E, series['train_loss_pointwise'], marker='o', label='Train')
        if any(x is not None for x in series['val_loss_pointwise']):   ax[1].plot(E, series['val_loss_pointwise'], marker='s', label='Val')
        ax[1].set_title("Pointwise (MSE)"); ax[1].legend(); ax[1].grid(alpha=.3, ls='--')

        # Spectral
        if any(x is not None for x in series['train_loss_spectral']): ax[2].plot(E, series['train_loss_spectral'], marker='o', label='Train')
        if any(x is not None for x in series['val_loss_spectral']):   ax[2].plot(E, series['val_loss_spectral'], marker='s', label='Val')
        ax[2].set_title("Spectral (Angle)"); ax[2].legend(); ax[2].grid(alpha=.3, ls='--')

        # Peak
        if any(x is not None for x in series['train_loss_peak']): ax[3].plot(E, series['train_loss_peak'], marker='o', label='Train')
        if any(x is not None for x in series['val_loss_peak']):   ax[3].plot(E, series['val_loss_peak'], marker='s', label='Val')
        ax[3].set_title("Peak (Weighted)"); ax[3].legend(); ax[3].grid(alpha=.3, ls='--')

        # Params
        if any(x is not None for x in series['train_loss_params']): ax[4].plot(E, series['train_loss_params'], marker='o', label='Train')
        if any(x is not None for x in series['val_loss_params']):   ax[4].plot(E, series['val_loss_params'], marker='s', label='Val')
        ax[4].set_title("Params (phys.)"); ax[4].legend(); ax[4].grid(alpha=.3, ls='--')

        # Résumé
        ax[5].axis('off')
        last = epochs_sorted[-1]
        h = self._history[last]
        txt = [
            f"Epoch {last+1}",
            f"Loss Totale     | T: {h.get('train_loss', float('nan')):.6g} | V: {h.get('val_loss', float('nan')):.6g}",
            f"Pointwise (MSE) | T: {h.get('train_loss_pointwise', float('nan')):.6g} | V: {h.get('val_loss_pointwise', float('nan')):.6g}",
            f"Spectral Angle  | T: {h.get('train_loss_spectral', float('nan')):.6g} | V: {h.get('val_loss_spectral', float('nan')):.6g}",
            f"Peak (weighted) | T: {h.get('train_loss_peak', float('nan')):.6g} | V: {h.get('val_loss_peak', float('nan')):.6g}",
            f"Params (phys.)  | T: {h.get('train_loss_params', float('nan')):.6g} | V: {h.get('val_loss_params', float('nan')):.6g}",
        ]
        ax[5].text(0.05, 0.95, "\n".join(txt), va='top', ha='left', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3))
        plt.tight_layout()
        save_fig(fig, self.save_path, dpi=150)

    def on_train_epoch_end(self, trainer, pl_module):
        self._record(trainer.current_epoch, trainer.callback_metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._record(trainer.current_epoch, trainer.callback_metrics)


class StageAwarePlotCallback(pl.Callback):
    """
    Callback de visualisation avec métriques MINIMALES:
    - loss_pointwise (MSE pixel-wise)
    - loss_spectral (Spectral Angle)
    - loss_params (paramètres physiques)
    + poids adaptatifs si activés
    """
    def __init__(self, val_loader, param_names, *,
                 num_examples: int = 1,
                 save_dir: str | None = None,
                 stage_tag: str = "stage",
                 refine: bool = True,
                 cascade_stages_override: int | None = None,
                 use_gt_for_provided: bool = True,
                 recon_PT: str = "pred",
                 Pexp: torch.Tensor | None = None,
                 Texp: torch.Tensor | None = None,
                 max_val_batches: int | None = None
                 ):
        super().__init__()
        self.val_loader = val_loader
        self.param_names = list(param_names)
        self.num_examples = int(num_examples)
        job = os.environ.get("SLURM_JOB_ID", "local")
        root = f"./figs_{job}" if save_dir is None else save_dir
        self.stage_tag = stage_tag
        self.save_dir = os.path.join(root, self.stage_tag)

        self.refine = bool(refine)
        self.cascade_stages_override = cascade_stages_override
        self.use_gt_for_provided = bool(use_gt_for_provided)
        self.recon_PT = recon_PT
        self.Pexp, self.Texp = Pexp, Texp
        self.max_val_batches = None if max_val_batches is None else int(max_val_batches)

    def _denorm_subset(self, pl_module, y_norm: torch.Tensor, names: list[str]) -> torch.Tensor:
        return pl_module._denorm_params_subset(y_norm, names)

    @torch.no_grad()
    def _compute_val_stats(self, pl_module) -> tuple[float, dict]:
        """Calcule MSE global et erreur % par paramètre sur tout le val_loader"""
        device = pl_module.device
        pred_names = list(getattr(pl_module, "predict_params", []))
        if len(pred_names) == 0:
            return float("nan"), {}

        # ✅ AJOUT : Récupérer les paramètres expérimentaux à exclure
        exp_params_A = getattr(pl_module, '_use_exp_params_in_A', [])
        exp_params_resid = getattr(pl_module, '_use_exp_params_for_resid', [])
        exp_params = set(exp_params_A) | set(exp_params_resid)
        
        # ✅ Filtrer les paramètres à évaluer
        eval_params = [p for p in pred_names if p not in exp_params]
        
        if len(eval_params) == 0:
            return float("nan"), {}

        mse_sum = 0.0
        n_points_total = 0
        err_sum = {p: 0.0 for p in eval_params}  # ← Seulement les params à évaluer
        err_cnt = 0
        eps = 1e-12

        for b_idx, batch in enumerate(self.val_loader):
            noisy  = batch['noisy_spectra'].to(device)
            clean  = batch['clean_spectra'].to(device)
            p_norm = batch['params'].to(device)
            B, N = noisy.shape

            provided_phys = {}
            if self.use_gt_for_provided:
                for name in getattr(pl_module, "provided_params", []):
                    idx = pl_module.name_to_idx[name]
                    v_phys = self._denorm_subset(pl_module, p_norm[:, idx].unsqueeze(1), [name])[:, 0]
                    provided_phys[name] = v_phys

            out = pl_module.infer(
                noisy,
                provided_phys=provided_phys,
                refine=self.refine,
                resid_target="input",
                recon_PT=self.recon_PT,
                Pexp=self.Pexp, Texp=self.Texp,
                cascade_stages_override=self.cascade_stages_override,
            )
            recon = out["spectra_recon"]
            y_full_pred = out["y_phys_full"]

            # MSE global
            diff = (recon - clean).float()
            mse_sum += float((diff * diff).sum().item())
            n_points_total += B * N

            # ✅ Erreurs % UNIQUEMENT pour les paramètres évalués (non-exp)
            true_cols = [p_norm[:, pl_module.name_to_idx[n]] for n in eval_params]
            if len(true_cols) == 0:
                continue
            true_norm_subset = torch.stack(true_cols, dim=1)
            true_phys = self._denorm_subset(pl_module, true_norm_subset, eval_params)

            pred_phys = torch.stack([y_full_pred[:, pl_module.name_to_idx[n]] for n in eval_params], dim=1)

            denom = torch.clamp(true_phys.abs(), min=eps)
            err_pct = 100.0 * (pred_phys - true_phys).abs() / denom

            for j, name in enumerate(eval_params):
                err_sum[name] += float(err_pct[:, j].sum().item())
            err_cnt += B

            if self.max_val_batches is not None and (b_idx + 1) >= self.max_val_batches:
                break

        mse_global = mse_sum / max(1, n_points_total)
        mean_pct = {k: (v / max(1, err_cnt)) for k, v in err_sum.items()}
        return mse_global, mean_pct

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not is_rank0():
            return
        pl_module.eval()
        device = pl_module.device

        # Exemple pour visualisation
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        noisy  = batch['noisy_spectra'][:self.num_examples].to(device)
        clean  = batch['clean_spectra'][:self.num_examples].to(device)
        params_true_norm = batch['params'][:self.num_examples].to(device)

        provided_phys = {}
        if self.use_gt_for_provided:
            for name in getattr(pl_module, "provided_params", []):
                idx = pl_module.name_to_idx[name]
                v_norm = params_true_norm[:, idx]
                v_phys = pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]
                provided_phys[name] = v_phys

        out = pl_module.infer(
            noisy, provided_phys=provided_phys,
            refine=self.refine,
            resid_target="input",
            recon_PT=self.recon_PT,
            Pexp=self.Pexp, Texp=self.Texp,
            cascade_stages_override=self.cascade_stages_override,
        )

        spectra_recon = out["spectra_recon"].detach().cpu()
        noisy_cpu, clean_cpu = noisy.detach().cpu(), clean.detach().cpu()
        x = np.arange(clean_cpu.shape[1])

        # Statistiques globales
        val_mse, mean_pct = self._compute_val_stats(pl_module)

        # Métriques Lightning
        m = trainer.callback_metrics
        def get_metric(name, default="-"):
            v = m.get(name, None)
            try: return f"{float(v):.6g}"
            except Exception: return default

        # ========== FIGURE ==========
        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[3, 1], hspace=0.30, wspace=0.35)
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_spec)
        ax_tbl  = fig.add_subplot(gs[:, 1]); ax_tbl.axis("off")

        i = 0
        ax_spec.plot(x, noisy_cpu[i],     label="Noisy",       lw=1,   alpha=0.7, color='gray')
        ax_spec.plot(x, clean_cpu[i],     label="Clean (GT)",  lw=1.5, color='tab:blue')
        ax_spec.plot(x, spectra_recon[i], label="Reconstruit", lw=1.2, ls="--", color='tab:orange')
        ax_spec.set_ylabel("Transmission", fontsize=10)
        ax_spec.set_title(f"{self.stage_tag} — Epoch {trainer.current_epoch}", fontsize=11, fontweight='bold')
        ax_spec.legend(frameon=False, fontsize=9, loc='upper right')
        ax_spec.grid(alpha=0.25, ls='--', lw=0.5)

        resid_clean = spectra_recon[i] - clean_cpu[i]
        resid_noisy = spectra_recon[i] - noisy_cpu[i]
        ax_res.plot(x, resid_noisy, lw=1,   label="Recon - Noisy", color='tab:gray')
        ax_res.plot(x, resid_clean, lw=1.2, label="Recon - Clean", color='tab:red')
        ax_res.axhline(0, ls=":", lw=0.8, color='black')
        ax_res.set_xlabel("Index spectral", fontsize=10)
        ax_res.set_ylabel("Résidu", fontsize=10)
        ax_res.legend(frameon=False, fontsize=9)
        ax_res.grid(alpha=0.25, ls='--', lw=0.5)

        # ✅ Identifier les paramètres expérimentaux
        exp_params_A = getattr(pl_module, '_use_exp_params_in_A', [])
        exp_params_resid = getattr(pl_module, '_use_exp_params_for_resid', [])
        exp_params_set = set(exp_params_A) | set(exp_params_resid)

        # ========== PANNEAU TEXTE (MÉTRIQUES MINIMALES) ==========
        # Ligne 1: Métriques principales
        lines = [
            "╔═══════════════════════════════════╗",
            "║   MÉTRIQUES PRINCIPALES           ║",
            "╚═══════════════════════════════════╝",
            "",
            f"train_loss       : {get_metric('train_loss')}",
            f"val_loss         : {get_metric('val_loss')}",
            "",
            "─────────────────────────────────────",
            "  Losses Individuelles (val)",
            "─────────────────────────────────────",
            f"pointwise (MSE)  : {get_metric('val_loss_pointwise')}",
            f"spectral (angle) : {get_metric('val_loss_spectral')}",
            f"peak (weighted)  : {get_metric('val_loss_peak')}",      # ← AJOUTÉ
            f"params (physique): {get_metric('val_loss_params')}",
            "",
        ]
                # ✅ Afficher les poids ReLoBRaLo si actifs
        if getattr(pl_module, 'use_relobralo_top', False):
            lines.extend([
                "─────────────────────────────────────",
                "  Poids ReLoBRaLo (adaptatifs)",
                "─────────────────────────────────────",
                f"w_pointwise : {get_metric('relo_weight_phys_pointwise')}",
                f"w_spectral  : {get_metric('relo_weight_phys_spectral')}",
                f"w_peak      : {get_metric('relo_weight_phys_peak')}",
                f"w_params    : {get_metric('relo_weight_param_group')}",
                "",
            ])


        # Ligne 3: Statistiques validation globales
        lines.extend([
            "─────────────────────────────────────",
            "  Statistiques Val Globales",
            "─────────────────────────────────────",
            f"MSE(clean,recon) : {val_mse:.6g}",
        ])

        # ✅ MODIFICATION : Afficher les erreurs + indiquer les params exp
        if len(mean_pct) > 0:
            lines.append("")
            lines.append("Erreur % moyenne par paramètre:")
            for k in sorted(mean_pct.keys()):
                lines.append(f"  {k:12s} : {mean_pct[k]:6.3f} %")
            
            # ✅ AJOUT : Indiquer les paramètres expérimentaux (non évalués)
            if len(exp_params_set) > 0:
                lines.append("")
                lines.append("Paramètres expérimentaux (0%):")
                for k in sorted(exp_params_set):
                    if k in pl_module.predict_params:
                        lines.append(f"  {k:12s} : EXP (GT utilisé)")

        # Ligne 4: Config
        lines.extend([
            "",
            "─────────────────────────────────────",
            "  Configuration",
            "─────────────────────────────────────",
            f"refine={self.refine}",
            f"cascade={self.cascade_stages_override if self.cascade_stages_override else 'auto'}",
            f"provided={'GT' if self.use_gt_for_provided else 'pred'}",
            f"recon_PT={self.recon_PT}",
        ])

        # ✅ AJOUT : Afficher les paramètres exp actifs
        if len(exp_params_A) > 0:
            lines.append(f"exp_in_A={','.join(exp_params_A)}")
        if len(exp_params_resid) > 0:
            lines.append(f"exp_in_resid={','.join(exp_params_resid)}")

        # Affichage du texte
        ax_tbl.text(
            0.05, 0.98, "\n".join(lines),
            va="top", ha="left",
            fontsize=8.5,
            family="monospace",
            transform=ax_tbl.transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.3, edgecolor='gray', linewidth=1)
        )

        # Sauvegarde
        os.makedirs(self.save_dir, exist_ok=True)
        out_png = os.path.join(self.save_dir, f"{self.stage_tag}_val_epoch{trainer.current_epoch:04d}.png")
        save_fig(fig, out_png, dpi=150)
        
        if trainer.is_global_zero:
            print(f"✅ Figure sauvegardée: {out_png}")


import pytorch_lightning as pl
from torch.utils.data import DataLoader

class UpdateEpochInDataset(pl.Callback):
    """Met à jour l'epoch dans le dataset de train à chaque début d'epoch."""
    def on_train_epoch_start(self, trainer, pl_module):
        # Accès robuste au dataloader de train
        if hasattr(trainer, 'train_dataloader'):
            # Récupérer le dataloader (peut être une fonction)
            dl = trainer.train_dataloader
            if callable(dl):
                dl = dl()
        elif hasattr(trainer, 'train_dataloaders'):
            dl = trainer.train_dataloaders
        else:
            return
        
        # Gérer le cas d'une liste de dataloaders
        if isinstance(dl, list):
            dl = dl[0] if len(dl) > 0 else None
        
        if dl is not None:
            ds = getattr(dl, 'dataset', None)
            if ds is not None and hasattr(ds, 'set_epoch'):
                ds.set_epoch(trainer.current_epoch)
                if trainer.is_global_zero:
                    print(f"✓ Train dataset epoch mis à jour: {trainer.current_epoch}")


class UpdateEpochInValDataset(pl.Callback):
    """Met à jour l'epoch dans le dataset de validation à chaque début d'epoch de val."""
    def on_validation_epoch_start(self, trainer, pl_module):
        # Accès robuste au(x) dataloader(s) de validation
        if hasattr(trainer, 'val_dataloaders'):
            dls = trainer.val_dataloaders
        elif hasattr(trainer, 'val_dataloader'):
            dls = trainer.val_dataloader
            if callable(dls):
                dls = dls()
        else:
            return
        
        # Normaliser en liste
        if not isinstance(dls, list):
            dls = [dls] if dls is not None else []
        
        for dl in dls:
            if dl is None:
                continue
            ds = getattr(dl, 'dataset', None)
            if ds is not None and hasattr(ds, 'set_epoch'):
                ds.set_epoch(trainer.current_epoch)
                if trainer.is_global_zero:
                    print(f"✓ Val dataset epoch mis à jour: {trainer.current_epoch}")


class AdvanceDistributedSamplerEpochAll(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dl = getattr(trainer, "train_dataloader", None)
        sampler = getattr(dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)

    def on_validation_epoch_start(self, trainer, pl_module):
        dls = getattr(trainer, "val_dataloaders", None)
        if not dls:
            return
        if not isinstance(dls, (list, tuple)):
            dls = [dls]
        for dl in dls:
            sampler = getattr(dl, "sampler", None)
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
        for p in model.out_head.parameters(): p.requires_grad_(True)
        return
    want = set(names) if names is not None else set(model.predict_params)
    for n, head in model.out_heads.items():
        req = (n in want)
        for p in head.parameters(): p.requires_grad_(req)

def freeze_module(m, req: bool):
    if m is None: return
    for p in m.parameters(): p.requires_grad_(req)

def freeze_all_refiners_except(model: PhysicallyInformedAE, keep_idx: int):
    """
    Active l'entraînement du raffineur keep_idx (0=B, 1=C, 2=D) et gèle les autres.
    """
    for i, r in enumerate(model.refiners):
        freeze_module(r, i == keep_idx)

def train_refiner_idx(
    model: PhysicallyInformedAE,
    train_loader, val_loader,
    k: int,                      
    *,
    epochs: int = 40,
    refiner_lr: float = 1e-4,
    delta_scale: float = 0.10,
    callbacks=None,
    enable_progress_bar: bool = False,
    ckpt_in: str | None = None,
    ckpt_out: str | None = None,
    use_denoiser_during_B: bool = False,
    use_exp_params_for_resid: list[str] | None = None,  # ✅ NOUVEAU
    w_pw_raw: float | None = None,
    w_spectral_angle: float | None = None,
    w_peak: float | None = None,
    w_params: float | None = None,
    use_relobralo_top: bool | None = None,
    pw_cfg: dict | None = None,
    **trainer_kwargs
):
    """
    Entraînement du raffineur B.
    
    Args:
        use_exp_params_for_resid: Liste des paramètres à utiliser en mode GT pour les résidus.
                                   Ex: ["P", "T"] ou ["P", "T", "mf_CH4"]
                                   Si None ou [], mode normal (tout prédit).
    """
    import pytorch_lightning as pl

    _load_weights_if_any(model, ckpt_in)

        # ✅ Mise à jour des poids de loss si fournis
    if w_pw_raw is not None:
        model.w_pw_raw = float(w_pw_raw)
    if w_spectral_angle is not None:
        model.w_spectral_angle = float(w_spectral_angle)
    if w_peak is not None:
        model.w_peak = float(w_peak)
    if w_params is not None:
        model.w_params = float(w_params)
    if use_relobralo_top is not None:
        model.use_relobralo_top = bool(use_relobralo_top)
    
    # ✅ Mise à jour de peak_weighted_loss si fournie
    if pw_cfg is not None and hasattr(model, 'peak_weighted_loss'):
        _pw_default = dict(
            peak_weight=4.0, baseline_weight=1.0,
            pre_smooth_sigma=1.3, salience_smooth_sigma=1.8,
            peak_kind="min", curv_scale_k=2.2,
            border_policy="taper", border_extra_margin=2,
            weight_normalize="mean", renorm_after_smoothing=True,
            salience_gamma=0.9,
            spread_kind="gaussian", spread_sigma=2.5, spread_kernel=11
        )
        _pw_default.update(pw_cfg)
        model.peak_weighted_loss = PeakWeightedMSELoss(**_pw_default)

    model.use_denoiser = bool(use_denoiser_during_B)
    model.base_lr = 1e-8
    model.refiner_lr = float(refiner_lr)
    model.set_stage_mode('B1', refine_steps=int(k+1), delta_scale=float(delta_scale))
    
    # ✅ NOUVEAU: Stocker les paramètres exp
    model._use_exp_params_for_resid = list(use_exp_params_for_resid) if use_exp_params_for_resid else []
    
    if len(model._use_exp_params_for_resid) > 0:
        print(f"✅ Mode raffineur avec paramètres expérimentaux:")
        print(f"   → Résidus calculés avec {model._use_exp_params_for_resid} GT")
        print(f"   → Raffineur ne corrige PAS {model._use_exp_params_for_resid}")
        
        # ✅ Adapter le masque de raffinement
        mask = torch.ones(len(model.predict_params), dtype=torch.float32)
        for i, name in enumerate(model.predict_params):
            if name in model._use_exp_params_for_resid:
                mask[i] = 0.0  # Bloquer la correction
        model.refine_mask_base = mask.to(model.device)
        
        correctable = [name for name in model.predict_params if name not in model._use_exp_params_for_resid]
        print(f"   → Paramètres corrigeables: {correctable}")
    else:
        print(f"✅ Mode raffineur standard:")
        print(f"   → Résidus avec tous les paramètres prédits")
        print(f"   → Raffineur corrige tous les paramètres")

    def freeze_module(m, trainable: bool):
        if m is None: return
        for p in m.parameters(): p.requires_grad_(trainable)

    def freeze_all_refiners_except(model: PhysicallyInformedAE, keep_idx: int):
        for i, r in enumerate(model.refiners):
            freeze_module(r, i == keep_idx)

    # Geler A
    freeze_module(model.backbone, False)
    freeze_module(model.shared_head, False)
    if hasattr(model, "out_head"):  freeze_module(model.out_head, False)
    if hasattr(model, "out_heads"):
        for _n, head in model.out_heads.items(): freeze_module(head, False)

    # Ne laisser entraînable que le raffineur k
    freeze_all_refiners_except(model, keep_idx=int(k))

    # Sanitization
    try:
        from pytorch_lightning.callbacks.progress import TQDMProgressBar
        if callbacks is not None and enable_progress_bar is False:
            callbacks = [cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)]
    except Exception:
        pass

    trainer_kwargs.setdefault("log_every_n_steps", 1)
    from pytorch_lightning.strategies import Strategy
    if isinstance(trainer_kwargs.get("strategy", None), Strategy):
        trainer_kwargs.pop("accelerator", None)

    trainer = pl.Trainer(
        max_epochs=int(epochs),
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)
    
    # ✅ Nettoyer
    if hasattr(model, '_use_exp_params_for_resid'):
        delattr(model, '_use_exp_params_for_resid')
    
    return model



def _apply_stage_freeze(model: PhysicallyInformedAE,
                        train_base: bool, train_heads: bool, train_refiner: bool,
                        heads_subset: Optional[List[str]]):
    _freeze_all(model)
    if train_base:
        for p in model.backbone.parameters(): p.requires_grad_(True)
        for p in model.shared_head.parameters(): p.requires_grad_(True)
    if model.head_mode == "single":
        if train_heads:
            for p in model.out_head.parameters(): p.requires_grad_(True)
    else:
        _set_trainable_heads(model, heads_subset if train_heads else [])
    if train_refiner:
        for p in model.refiners.parameters():
            p.requires_grad_(True)

def _load_weights_if_any(model: PhysicallyInformedAE, ckpt_in: Optional[str]):
    if ckpt_in:
        sd = torch.load(ckpt_in, map_location="cpu")
        state_dict = sd["state_dict"] if "state_dict" in sd else sd
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ weights chargés depuis: {ckpt_in}")

def _save_checkpoint(trainer: pl.Trainer, ckpt_out: Optional[str]):
    if ckpt_out:
        trainer.save_checkpoint(ckpt_out)
        print(f"✓ checkpoint sauvegardé: {ckpt_out}")

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
    train_refiner: bool,
    refine_steps: int,
    delta_scale: float,
    heads_subset: Optional[List[str]] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    **trainer_kwargs,
):
    """
    Stage Lightning robuste:
    - évite accelerator/strategy en double
    - pas de plugins= → aucun conflit cluster_environment
    """
    import os
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import Strategy

    print(f"\n===== Stage {stage_name} =====")

    # --- EXTRA KW non-Lightning ---
    ckpt_in  = trainer_kwargs.pop("ckpt_in",  None)
    ckpt_out = trainer_kwargs.pop("ckpt_out", None)

    _load_weights_if_any(model, ckpt_in)

    # Progress bar optionnelle
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

    _apply_stage_freeze(
        model,
        train_base=train_base, train_heads=train_heads, train_refiner=train_refiner,
        heads_subset=heads_subset,
    )

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    # 🔧 SANITIZE: si strategy est un objet, on laisse Lightning déduire l'accélérateur
    strat = trainer_kwargs.get("strategy", None)
    if isinstance(strat, Strategy):
        trainer_kwargs.pop("accelerator", None)

    # 🔧 Nettoie quelques variables d'env piégeuses
    for env_key in ("PL_TRAINER_ACCELERATOR", "PL_TRAINER_MAX_EPOCHS"):
        os.environ.pop(env_key, None)

    # Construire le Trainer (sans plugins= → pas de conflit)
    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)
    return model

def train_stage_DENOISER(
    model: PhysicallyInformedAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 30,
    denoiser_lr: float = 1e-4,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    ckpt_in: str | None = None,
    ckpt_out: str | None = None,
    **trainer_kwargs,
):
    import pytorch_lightning as pl

    if ckpt_in:
        sd = torch.load(ckpt_in, map_location="cpu")
        state_dict = sd.get("state_dict", sd)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ weights chargés depuis: {ckpt_in}")

    model.use_denoiser = True
    model.denoiser_lr = float(denoiser_lr)
    model.set_stage_mode('DEN', refine_steps=0, delta_scale=None)

    # --- sanitization kwargs pour éviter les doublons (strategy / accelerator, etc.)
    try:
        from pytorch_lightning.callbacks.progress import TQDMProgressBar
        if callbacks is not None and enable_progress_bar is False:
            callbacks = [cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)]
    except Exception:
        pass

    from pytorch_lightning.strategies import Strategy
    if isinstance(trainer_kwargs.get("strategy", None), Strategy):
        trainer_kwargs.pop("accelerator", None)

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    # Ne pas imposer une strategy si déjà fournie via **trainer_kwargs
    trainer = pl.Trainer(
        max_epochs=int(epochs),
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )
    trainer.fit(model, train_loader, val_loader)

    if ckpt_out:
        trainer.save_checkpoint(ckpt_out)
        print(f"✓ checkpoint sauvegardé: {ckpt_out}")

    return model


def train_stage_A(
    model: PhysicallyInformedAE,
    train_loader, val_loader,
    *,
    epochs: int = 70,
    base_lr: float = 1e-4,
    heads_subset=None,
    callbacks=None,
    enable_progress_bar: bool = False,
    ckpt_in: str | None = None,
    ckpt_out: str | None = None,
    use_exp_params: list[str] | None = None,
    **trainer_kwargs
):
    """
    Entraînement Stage A.
    
    Args:
        use_exp_params: Liste des paramètres à utiliser en mode GT.
                        Ces paramètres ne seront PAS entraînés (pas de loss).
    """
    import pytorch_lightning as pl
    
    _load_weights_if_any(model, ckpt_in)
    
    model.base_lr = float(base_lr)
    model.refiner_lr = 0.0
    model.set_stage_mode('A', refine_steps=0, delta_scale=None)
    
    # ✅ Stocker les paramètres exp
    model._use_exp_params_in_A = list(use_exp_params) if use_exp_params else []
    
    # ✅ SIMPLIFIÉ: Pas de réinitialisation ici, elle se fait automatiquement dans _common_step()
    n_active_params = len([p for p in model.predict_params if p not in model._use_exp_params_in_A])
    
    if len(model._use_exp_params_in_A) > 0:
        print(f"✅ Mode A avec paramètres expérimentaux:")
        print(f"   → Physique utilise {model._use_exp_params_in_A} GT")
        print(f"   → Têtes {model._use_exp_params_in_A}: PAS de loss (pas entraînées)")
        print(f"   → Focus sur {n_active_params} autres paramètres")
        print(f"   → ReLoBRaLo sera réinitialisé automatiquement au premier batch")
    else:
        print(f"✅ Mode A standard:")
        print(f"   → Tous les {len(model.predict_params)} paramètres prédits et entraînés")

    def freeze_module(m, trainable: bool):
        if m is None: return
        for p in m.parameters():
            p.requires_grad_(trainable)

    # Entraîner backbone et têtes
    freeze_module(model.backbone, True)
    freeze_module(model.shared_head, True)

    if heads_subset is None or len(heads_subset) == 0:
        if hasattr(model, "out_head"):
            freeze_module(model.out_head, True)
        if hasattr(model, "out_heads"):
            for _n, head in model.out_heads.items():
                freeze_module(head, True)
    else:
        if hasattr(model, "out_head"):
            freeze_module(model.out_head, False)
        if hasattr(model, "out_heads"):
            for name, head in model.out_heads.items():
                freeze_module(head, name in heads_subset)

    # Geler les raffineurs et denoiser
    for r in model.refiners:
        freeze_module(r, False)
    if getattr(model, "denoiser", None) is not None:
        freeze_module(model.denoiser, False)

    # Sanitization
    try:
        from pytorch_lightning.callbacks.progress import TQDMProgressBar
        if callbacks is not None and enable_progress_bar is False:
            callbacks = [cb for cb in callbacks if not isinstance(cb, TQDMProgressBar)]
    except Exception:
        pass

    trainer_kwargs.setdefault("log_every_n_steps", 1)
    from pytorch_lightning.strategies import Strategy
    if isinstance(trainer_kwargs.get("strategy", None), Strategy):
        trainer_kwargs.pop("accelerator", None)

    trainer = pl.Trainer(
        max_epochs=int(epochs),
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)
    _save_checkpoint(trainer, ckpt_out)
    
    # Nettoyer
    if hasattr(model, '_use_exp_params_in_A'):
        delattr(model, '_use_exp_params_in_A')
    
    return model


def train_stage_B1(model, train_loader, val_loader, **kw):
    # B1 = raffineur seul
    defaults = dict(
        stage_name="B1", epochs=12,
        base_lr=1e-6, refiner_lr=1e-5,
        train_base=False, train_heads=False, train_refiner=True,
        refine_steps=2, delta_scale=0.12,
        heads_subset=None,
        enable_progress_bar=False
    ); defaults.update(kw)
    return train_stage_custom(model, train_loader, val_loader, **defaults)

def train_stage_B2(model, train_loader, val_loader, **kw):
    # B2 = fine-tune global (petits LR), raffinement ON
    defaults = dict(
        stage_name="B2", epochs=15,
        base_lr=3e-5, refiner_lr=3e-6,
        train_base=True, train_heads=True,  train_refiner=True,
        refine_steps=2, delta_scale=0.08,
        heads_subset=None,
        enable_progress_bar=False
    ); defaults.update(kw)
    return train_stage_custom(model, train_loader, val_loader, **defaults)


@torch.no_grad()
def evaluate_and_plot(model: PhysicallyInformedAE, loader: DataLoader, n_show: int = 5,
                      refine: bool = True, robust_smape: bool = False, eps: float = 1e-12, seed: int = 123,
                      baseline_correction: dict | None = None,
                      save_dir: str | None = None, tag: str | None = None):
    model.eval()
    device = model.device
    rng = random.Random(seed)

    pred_names = list(getattr(model, "predict_params", []))
    if not pred_names:
        raise RuntimeError("Aucun paramètre à évaluer.")

    # ✅ AJOUT : Exclure les paramètres expérimentaux
    exp_params_A = getattr(model, '_use_exp_params_in_A', [])
    exp_params_resid = getattr(model, '_use_exp_params_for_resid', [])
    exp_params_set = set(exp_params_A) | set(exp_params_resid)
    eval_params = [p for p in pred_names if p not in exp_params_set]
    
    if len(eval_params) == 0:
        print("⚠️  Tous les paramètres sont expérimentaux, rien à évaluer.")
        return pd.DataFrame()

    per_param_err = {p: [] for p in eval_params}  # ← Seulement les params à évaluer
    show_examples = []

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

        # ✅ Calculer les erreurs UNIQUEMENT pour les paramètres non-exp
        true_cols = [p_norm[:, model.name_to_idx[n]] for n in eval_params]
        true_norm_tensor = torch.stack(true_cols, dim=1)
        true_phys = model._denorm_params_subset(true_norm_tensor, eval_params)
        pred_phys = torch.stack([y_full_pred[:, model.name_to_idx[n]] for n in eval_params], dim=1)

        if robust_smape:
            denom = pred_phys.abs() + true_phys.abs() + eps
            err_pct = 100.0 * 2.0 * (pred_phys - true_phys).abs() / denom
        else:
            denom = torch.clamp(true_phys.abs(), min=eps)
            err_pct = 100.0 * (pred_phys - true_phys).abs() / denom

        for j, name in enumerate(eval_params):
            per_param_err[name].append(err_pct[:, j].detach().cpu())

        for i in range(B):
            if len(show_examples) < n_show:
                show_examples.append({
                    "noisy":  noisy[i].detach().cpu(),
                    "clean":  clean[i].detach().cpu(),
                    "recon":  recon[i].detach().cpu(),
                    "pred":   {name: float(pred_phys[i, j]) for j, name in enumerate(eval_params)},
                    "true":   {name: float(true_phys[i, j]) for j, name in enumerate(eval_params)},
                    "errpct": {name: float(err_pct[i, j])   for j, name in enumerate(eval_params)},
                })
        if len(show_examples) >= n_show: break

    # ✅ Rapport des statistiques
    rows = []
    for name in eval_params:
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
    print("\n=== Erreurs en % (paramètres prédits uniquement) ===")
    print(df.round(4))
    
    # ✅ Afficher les paramètres exp
    if len(exp_params_set) > 0:
        print(f"\n⚠️  Paramètres expérimentaux (GT utilisé, non évalués): {sorted(exp_params_set)}")


    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        tag = tag or "eval"
        csv_path = os.path.join(save_dir, f"{tag}_metrics.csv")
        df.round(6).to_csv(csv_path)
        print(f"✓ Metrics CSV: {csv_path}")

    if len(show_examples) > 0:
        R = len(show_examples)
        fig, axes = plt.subplots(R, 2, figsize=(12, 2.8*R), sharex=False)
        if R == 1: axes = np.array([axes])
        for r, ex in enumerate(show_examples):
            x = np.arange(ex["clean"].numel()); ax1, ax2 = axes[r, 0], axes[r, 1]
            ax1.plot(x, ex["noisy"],  lw=0.9, alpha=0.7, label="Noisy")
            ax1.plot(x, ex["clean"],  lw=1.2,              label="Clean")
            ax1.plot(x, ex["recon"],  lw=1.0, ls="--",     label="Recon")
            ax1.set_ylabel("Transmission")
            err_txt = ", ".join([f"{k}: {ex['errpct'][k]:.2f}%" for k in pred_names])
            ax1.set_title(f"Exemple {r+1} — erreurs % : {err_txt}", fontsize=10)
            ax1.legend(frameon=False, fontsize=8)

            ax2.plot(x, ex["recon"] - ex["noisy"], lw=0.9, label="Recon - Noisy")
            ax2.plot(x, ex["recon"] - ex["clean"], lw=0.9, label="Recon - Clean")
            ax2.axhline(0, color="k", ls=":", lw=0.7)
            ax2.set_ylabel("Résidu"); ax2.legend(frameon=False, fontsize=8)
        axes[-1, 0].set_xlabel("Index spectral"); axes[-1, 1].set_xlabel("Index spectral")
        for axrow in axes:
            for ax in axrow: ax.grid(alpha=0.25)
        fig.tight_layout()
        if save_dir is not None:
            png_path = os.path.join(save_dir, f"{tag}_examples.png")
            save_fig(fig, png_path, dpi=160)
            print(f"✓ Examples PNG: {png_path}")
        else:
            plt.show()
    return df

import os, torch, numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class PT_PredVsExp_VisuCallback(pl.Callback):
    """
    Visualise côte-à-côte:
      - Recon (PT=pred) + résidus
      - Recon (PT=exp)  + résidus
    Les Pexp/Texp "exp" sont, par défaut, pris des GT du batch (dénormalisés).
    """
    def __init__(self, val_loader, save_dir="./figs_local", num_examples=1, tag="PT_pred_vs_exp",
                 force_Pexp: torch.Tensor | None = None,
                 force_Texp: torch.Tensor | None = None,
                 use_gt_for_provided: bool = True):
        super().__init__()
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.num_examples = int(num_examples)
        self.tag = tag
        self.force_Pexp = force_Pexp
        self.force_Texp = force_Texp
        self.use_gt_for_provided = bool(use_gt_for_provided)
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "eval"): pl_module.eval()
        device = pl_module.device
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        noisy  = batch["noisy_spectra"][:self.num_examples].to(device)
        clean  = batch["clean_spectra"][:self.num_examples].to(device)
        p_norm = batch["params"][:self.num_examples].to(device)
        B, N = noisy.shape
        x = np.arange(N)

        # ✓ FIX : Construire provided_phys avec TOUS les paramètres fournis
        provided_phys = {}
        
        # Extraire tous les paramètres "provided" (ceux qui ne sont PAS prédits)
        for name in getattr(pl_module, "provided_params", []):
            idx = pl_module.name_to_idx[name]
            v_norm = p_norm[:, idx]
            v_phys = pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]
            provided_phys[name] = v_phys
    
        # --- Pexp/Texp : soit forcés, soit pris des GT du batch
        def _gt_phys(name: str):
            idx = pl_module.name_to_idx[name]
            v_norm = p_norm[:, idx]
            return pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]

        if self.force_Pexp is not None and self.force_Texp is not None:
            Pexp = self.force_Pexp.to(device).view(B)
            Texp = self.force_Texp.to(device).view(B)
        else:
            # GT du batch (dénormalisés)
            Pexp = _gt_phys("P").to(device)
            Texp = _gt_phys("T").to(device)

        # === Recon avec PT prédits
        out_pred = pl_module.infer(
            noisy, provided_phys=provided_phys,
            refine=True,
            resid_target="input",
            recon_PT="pred",
        )
        recon_pred = out_pred["spectra_recon"]

        # === Recon avec PT expérimentaux (forcés/GT)
        out_exp = pl_module.infer(
            noisy, provided_phys=provided_phys,
            refine=True,
            resid_target="input",
            recon_PT="exp",
            Pexp=Pexp, Texp=Texp,
        )
        recon_exp = out_exp["spectra_recon"]

        # --- tracés (reste identique)
        for i in range(B):
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.25, wspace=0.25)

            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(x, noisy[i].detach().cpu(),  lw=0.9, alpha=0.7, label="Noisy", color='gray')
            ax0.plot(x, clean[i].detach().cpu(),  lw=1.2, label="Clean", color='tab:blue')
            ax0.plot(x, recon_pred[i].detach().cpu(), lw=1.0, ls="--", label="Recon (PT=pred)", color='tab:orange')
            ax0.plot(x, recon_exp[i].detach().cpu(),  lw=1.0, ls="-.", label="Recon (PT=exp)", color='tab:green')
            ax0.set_title(f"{self.tag} — epoch {trainer.current_epoch} — ex {i+1}", fontweight='bold')
            ax0.set_ylabel("Transmission")
            ax0.legend(frameon=False, fontsize=9)
            ax0.grid(alpha=0.3)

            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax1.plot(x, (recon_pred[i]-noisy[i]).detach().cpu(), lw=0.9, label="(PT=pred) - Noisy", color='tab:orange')
            ax1.plot(x, (recon_exp[i]-noisy[i]).detach().cpu(),  lw=0.9, label="(PT=exp) - Noisy", color='tab:green')
            ax1.axhline(0, ls=":", lw=0.8, color="k")
            ax1.set_xlabel("Index spectral")
            ax1.set_ylabel("Résidu")
            ax1.legend(frameon=False, fontsize=8)
            ax1.grid(alpha=0.3)

            ax2 = fig.add_subplot(gs[1, 1], sharex=ax0)
            ax2.plot(x, (recon_pred[i]-clean[i]).detach().cpu(), lw=0.9, label="(PT=pred) - Clean", color='tab:orange')
            ax2.plot(x, (recon_exp[i]-clean[i]).detach().cpu(),  lw=0.9, label="(PT=exp) - Clean", color='tab:green')
            ax2.axhline(0, ls=":", lw=0.8, color="k")
            ax2.set_xlabel("Index spectral")
            ax2.set_ylabel("Résidu")
            ax2.legend(frameon=False, fontsize=8)
            ax2.grid(alpha=0.3)

            # ⭐ TEXTE INFORMATIF (SIMPLIFIÉ)
            try:
                Pp = float(out_pred["y_phys_full"][i, pl_module.name_to_idx["P"]])
                Tp = float(out_pred["y_phys_full"][i, pl_module.name_to_idx["T"]])
            except Exception:
                Pp = Tp = float("nan")
            
            info_text = (
                f"PT_pred ≈ ({Pp:.2f} mbar, {Tp:.2f} K)\n"
                f"PT_exp  = ({float(Pexp[i]):.2f} mbar, {float(Texp[i]):.2f} K)"
            )
            
            ax0.text(
                0.02, 0.02, info_text,
                transform=ax0.transAxes,
                fontsize=9,
                ha="left", va="bottom",
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.4)
            )

            # Sauvegarde
            fname = os.path.join(self.save_dir, f"{self.tag}_epoch{trainer.current_epoch:04d}_ex{i+1}.png")
            fig.savefig(fname, dpi=160, bbox_inches="tight")
            plt.close(fig)
            
            if trainer.is_global_zero:
                print(f"✅ Figure PT sauvegardée : {fname}")

# ============================================================
# 11) Build data & modèle (exemple par défaut)
# ============================================================

def _to_serializable(x):
    if isinstance(x, tuple):
        return [_to_serializable(v) for v in x]
    if isinstance(x, list):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (np.generic,)):
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
    clean = _to_serializable(cfg)
    with open(path, "w") as f:
        yaml.dump(
            clean, f,
            Dumper=_NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False
        )
    return path

def build_data_and_model(
    *,
    seed=42, n_points=800, n_train=500000, n_val=5000, batch_size=32,
    train_ranges=None, val_ranges=None, noise_train=None, noise_val=None,
    predict_list=None, lrs=(1e-4, 1e-5),
    backbone_variant="s", refiner_variant="s",
    backbone_width_mult=1, backbone_depth_mult=0.4,
    refiner_width_mult=1.0,  refiner_depth_mult=1.0,
    backbone_stem_channels=None, refiner_stem_channels=None,
    backbone_drop_path=0.0, refiner_drop_path=0.0,
    backbone_se_ratio=0.25,  refiner_se_ratio=0.25,
    refiner_feature_pool="avg", refiner_shared_hidden_scale=0.5,
    huber_beta=0.002,
    scheduler_T_0: int = 10,
    scheduler_T_mult: float = 1.2,
    scheduler_eta_min: float = 1e-9,
    scheduler_decay_factor: float = 0.75,
    scheduler_warmup_epochs: int = 5,

    qtpy_dir: str | None = None,
    # --- LOSSES 
    w_pw_raw: float = 1.0,
    w_spectral_angle: float = 0.5,
    w_peak: float = 0.0,
    w_params: float = 1.0,              # ← AJOUTÉ
    use_relobralo_top: bool = True,     # ← AJOUTÉ
    pw_cfg: dict | None = None,
):
    pl.seed_everything(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_windows = sys.platform == "win32"
    num_workers = 0 if is_windows else 4

    # ---- QTpy (TIPS_2021) ----
    if qtpy_dir is None:
        qtpy_dir = os.environ.get("QTPY_DIR", "./QTpy")
    tipspy = Tips2021QTpy(qtpy_dir, device='cpu')

    # ---- Polynom de fréquence & transitions de démonstration ----
    poly_freq_CH4 = [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]
    transitions_ch4_str = """6;1;3085.861015;1.013E-19;0.06;0.078;219.9411;0.73;-0.00712;0.0;0.0221;0.96;0.584;1.12
6;1;3085.832038;1.693E-19;0.0597;0.078;219.9451;0.73;-0.00712;0.0;0.0222;0.91;0.173;1.11
6;1;3085.893769;1.011E-19;0.0602;0.078;219.9366;0.73;-0.00711;0.0;0.0184;1.14;-0.516;1.37
6;1;3086.030985;1.659E-19;0.0595;0.078;219.9197;0.73;-0.00711;0.0;0.0193;1.17;-0.204;0.97
6;1;3086.071879;1.000E-19;0.0585;0.078;219.9149;0.73;-0.00703;0.0;0.0232;1.09;-0.0689;0.82
6;1;3086.085994;6.671E-20;0.055;0.078;219.9133;0.70;-0.00610;0.0;0.0300;0.54;0.00;0.0"""

    transitions_h2o_str = """1;2;3083.831748;2.874e-24;0.0971;0.460;78.9886;0.87;-0.00653
1;1;3085.357520;9.562e-25;0.0452;0.282;2254.2838;0.51;0.001433
1;1;3085.506609;1.396e-25;0.0662;0.344;2927.9412;0.63;0.00324
1;1;3085.558839;3.186e-25;0.0491;0.293;2254.2844;0.82;-0.00464
1;1;3085.689600;3.912e-25;0.0508;0.333;2612.7999;0.64;-0.00649
1;1;3086.133208;2.369e-25;0.0457;0.272;2414.7234;0.44;-0.00591
1;1;3087.192118;2.070e-22;0.0768;0.413;648.9787;0.60;-0.00803"""

    transitions_dict = {'CH4': parse_csv_transitions(transitions_ch4_str)}

    # ---- Ranges par défaut (val puis train étendu) ----
    default_val = {
        'sig0': (3085.43, 3085.46),
        'dsig': (0.001521, 0.00154),
        'mf_CH4': (2e-6, 20e-6),
        'baseline0': (0.999999, 1.00001),
        'baseline1': (-0.0004, -0.0003),
        'baseline2': (-4.0565E-08, -3.07117E-08),
        'P': (450, 550), 
        'T': (273.15 + 32, 273.15 + 37), #
    }

    expand_factors = {"_default": 1.0, 'sig0': 5.0, 'dsig': 3.0, 'mf_CH4': 2,
                      "baseline0": 1, "baseline1": 3, "baseline2": 8.0, "P": 2, "T":2}
    
    default_train = map_ranges(default_val, expand_interval, per_param=expand_factors)

    # Plancher log
    lo, hi = default_train['mf_CH4']; default_train['mf_CH4'] = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR*10))
    lo, hi = default_val['mf_CH4'];   default_val['mf_CH4']   = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR*10))

    # ---- NORM_PARAMS ----
    global NORM_PARAMS
    VAL_RANGES = val_ranges or default_val
    TRAIN_RANGES = train_ranges or default_train
    assert_subset(VAL_RANGES, TRAIN_RANGES, "VAL", "TRAIN")
    NORM_PARAMS = TRAIN_RANGES

    # ---- Profils de bruit ----
    NOISE_TRAIN = noise_train or dict(
        std_add_range=(0, 5e-3), std_mult_range=(0, 5e-3),
        p_drift=0.2, drift_sigma_range=(1.0, 100.0), drift_amp_range=(0.00001, 0.005),
        p_fringes=0.2, n_fringes_range=(1, 2), fringe_freq_range=(0.3, 50.0), fringe_amp_range=(0.001, 0.0015),
        p_spikes=0.2, spikes_count_range=(1, 2), spike_amp_range=(0.001, 1), spike_width_range=(1.0, 200.0),
        clip=(0.0, 1.5),
    )

    NOISE_VAL = noise_val or dict(
        std_add_range=(0, 1e-3), std_mult_range=(0, 1e-3),
        p_drift=0, drift_sigma_range=(1.0, 100.0), drift_amp_range=(0.00001, 0.005),
        p_fringes=0, n_fringes_range=(1, 2), fringe_freq_range=(0.3, 50.0), fringe_amp_range=(0.00001, 0.0015),
        p_spikes=0, spikes_count_range=(1, 2), spike_amp_range=(0.0001, 1), spike_width_range=(1.0, 200.0),
        clip=(0.0, 1.5),
    )

    dataset_train = SpectraDataset(n_samples=n_train, num_points=n_points,
                                   poly_freq_CH4=poly_freq_CH4, transitions_dict=transitions_dict,
                                   sample_ranges=TRAIN_RANGES, strict_check=True,
                                   with_noise=True, noise_profile=NOISE_TRAIN, freeze_noise=False, tipspy=tipspy)
    
    dataset_val = SpectraDataset(n_samples=n_val, num_points=n_points,
                                 poly_freq_CH4=poly_freq_CH4, transitions_dict=transitions_dict,
                                 sample_ranges=VAL_RANGES, strict_check=True,
                                 with_noise=True, noise_profile=NOISE_VAL, freeze_noise=False, tipspy=tipspy)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=(device == 'cuda'))

    # baseline0 n'est PAS prédit (normalisation via max LOWESS)
    predict_list = predict_list or ["sig0", "dsig","P", "T", "mf_CH4", "baseline1", "baseline2"]

    model = PhysicallyInformedAE(
            n_points=n_points, param_names=PARAMS,
            poly_freq_CH4=poly_freq_CH4, transitions_dict=transitions_dict,

            # --- optims & pondérations globales ---
            lr=lrs[0], head_mode="multi",

            # --- params à prédire 
            predict_params=predict_list,

            # --- raffinement ---
            refine_steps=1, refine_delta_scale=0.1, refine_target="noisy",
            base_lr=lrs[0], refiner_lr=lrs[1],

            # --- ⭐ POIDS DE PERTES (avec ReLoBRaLo) ⭐ ---
            w_pw_raw=w_pw_raw,
            w_spectral_angle=w_spectral_angle,
            w_peak=w_peak,
            w_params=w_params,                    # ← AJOUTÉ
            use_relobralo_top=use_relobralo_top,  # ← AJOUTÉ
            pw_cfg=pw_cfg,

            huber_beta=huber_beta,
            weight_mf=2.0,

            mlp_dropout=0.30,
            refiner_mlp_dropout=0.20,

            # --- backbones ---
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

            # --- data & TIPS ---
            tipspy=tipspy,
            # SCHEDULER PARAMETERS
            scheduler_T_0=scheduler_T_0,
            scheduler_T_mult=scheduler_T_mult,
            scheduler_eta_min=scheduler_eta_min,
            scheduler_decay_factor=scheduler_decay_factor,
            scheduler_warmup_epochs=scheduler_warmup_epochs,
        )

    model.hparams.optimizer = "lion"
    model.hparams.betas = (0.9, 0.99)
    model.weight_decay = 1e-4
    
    # ✅ VÉRIFICATION
    print("\n" + "="*70)
    print("🔍 VÉRIFICATION DES POIDS DE LOSS APRÈS CRÉATION")
    print("="*70)
    print(f"w_pw_raw          = {model.w_pw_raw}")
    print(f"w_spectral_angle  = {model.w_spectral_angle}")
    print(f"w_peak            = {model.w_peak}")
    print(f"w_params          = {model.w_params}")
    print(f"use_relobralo_top = {model.use_relobralo_top}")
    print(f"peak_weighted_loss = {type(model.peak_weighted_loss).__name__}")
    if model.use_relobralo_top:
        print(f"relo_top          = {type(model.relo_top).__name__} (actif)")
    print("="*70 + "\n")

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


# ------------- Utilitaires HPC & DDP -------------
def get_master_addr_and_port(default_port=12910):
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
        if nodelist:
            try:
                import subprocess, shlex
                cmd = f"scontrol show hostnames {shlex.quote(nodelist)} | head -n 1"
                master_addr = subprocess.check_output(cmd, shell=True).decode().strip()
            except Exception:
                master_addr = socket.gethostname()
        else:
            master_addr = socket.gethostname()
    master_port = int(os.environ.get("MASTER_PORT", default_port))
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    return master_addr, master_port

def choose_precision():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        if torch.cuda.get_device_capability(0)[0] >= 7:
            return "16-mixed"
    return "32-true"

from torch.utils.data import DataLoader, DistributedSampler as _DistributedSampler

def _with_sampler(loader, shuffle_default):
    if loader is None: return None
    if isinstance(getattr(loader, "sampler", None), _DistributedSampler):
        return loader
    ds = loader.dataset
    sampler = _DistributedSampler(ds, shuffle=shuffle_default)
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return train_loader, val_loader
    return _with_sampler(train_loader, True), _with_sampler(val_loader, False)


def on_rank_zero():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def make_run_dir(base="runs"):
    job = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = os.path.join(base, f"{job}")
    os.makedirs(run_dir, exist_ok=True)
    for d in ("checkpoints", "figs", "eval", "logs"):
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    return run_dir

import torch

def trainer_common_kwargs():
    return dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,                     # 1 seul GPU local
        # PAS de 'strategy' ici
        precision="32-true",
        default_root_dir="./lightning_logs_notebook",
        enable_progress_bar=True,
        log_every_n_steps=5,
        deterministic=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

tkw = trainer_common_kwargs()

# ============================================================
# CONFIGURATION
# ============================================================

# ✅ Paramètres expérimentaux (pas prédits, pas de loss)
USE_EXP_PARAMS_IN_A = ["P", "T"]           # ou [] pour mode pred
USE_EXP_PARAMS_FOR_RESID = ["P", "T"]      # ou [] pour mode pred

print("\n" + "="*70)
print("🎯 CONFIGURATION SIMPLIFIÉE")
print("="*70)
if len(USE_EXP_PARAMS_IN_A) > 0:
    print(f"Stage A:")
    print(f"  → Utilise {USE_EXP_PARAMS_IN_A} expérimentaux pour physique")
    print(f"  → PAS de loss sur {USE_EXP_PARAMS_IN_A} (pas entraînés)")
    print(f"  → Entraînement uniquement sur autres paramètres")
else:
    print(f"Stage A: Mode PRED (tous paramètres prédits et entraînés)")

if len(USE_EXP_PARAMS_FOR_RESID) > 0:
    print(f"\nStage B:")
    print(f"  → Résidus calculés avec {USE_EXP_PARAMS_FOR_RESID} expérimentaux")
    print(f"  → Raffineur ne corrige PAS {USE_EXP_PARAMS_FOR_RESID}")
else:
    print(f"\nStage B: Mode PRED (résidus avec tous params prédits)")
print("="*70 + "\n")


# ============================================================
# BUILD
# ============================================================

model, train_loader, val_loader = build_data_and_model(
    seed=42,
    n_points=800,
    n_train=50000,
    n_val=5000,
    batch_size=32,
    backbone_variant="s",
    backbone_width_mult=1,
    backbone_depth_mult=1,
    refiner_variant="s",
    refiner_width_mult=1,
    refiner_depth_mult=1,
    w_pw_raw=0.10,
    w_spectral_angle=0.70,
    w_peak=0.20,              
    w_params=0.0,
    use_relobralo_top=True, 

    
    # ✅ Configuration de la peak loss
    pw_cfg=dict(
        peak_weight=4.0,
        baseline_weight=1.0,
        pre_smooth_sigma=1.3,
        salience_smooth_sigma=2.0,
        spread_kind="gaussian",
        spread_sigma=2.8,
        weight_normalize="mean",
        border_policy="taper",
        salience_gamma=0.9
    ),

    qtpy_dir="C:/Users/goff0007/Documents/Aerolab/Python_code/PINN/QTpy",
    predict_list=["sig0", "dsig", "mf_CH4", "baseline1", "baseline2", "P", "T"],
    
    # ✅ Configuration du scheduler
    scheduler_T_0=10,
    scheduler_T_mult=1.2,
    scheduler_eta_min=1e-6,
    scheduler_decay_factor=0.8,
    scheduler_warmup_epochs=5,
)

# ============================================================
# ✅ VISUALISER LE SCHEDULER AVANT L'ENTRAÎNEMENT
# ============================================================

suffix_A = "_".join(USE_EXP_PARAMS_IN_A) if USE_EXP_PARAMS_IN_A else "pred"

print("\n" + "="*70)
print("📊 GÉNÉRATION DU PREVIEW DU LEARNING RATE SCHEDULE")
print("="*70)

# Créer un optimizer temporaire pour preview
temp_opt_config = model.configure_optimizers()
temp_scheduler = temp_opt_config['lr_scheduler']['scheduler']

# Générer et sauvegarder la figure dans le même dossier que les loss curves
save_dir_scheduler = f"./figs_local/stageA_{suffix_A}"
os.makedirs(save_dir_scheduler, exist_ok=True)

cycle_stats = plot_lr_scheduler_preview(
    temp_scheduler,
    max_epochs=166,  # Nombre d'epochs prévus pour les stages A+B
    save_dir=save_dir_scheduler,
    filename="lr_schedule_preview.png"
)

# Afficher les détails des cycles
print(f"\n📋 DÉTAIL DES CYCLES PRÉVUS:")
print(f"{'Cycle':<8} {'Durée':<8} {'Max LR':<12} {'Période théorique':<20}")
print("-"*60)

for i, stat in enumerate(cycle_stats):
    expected_period = temp_scheduler.T_0 * (temp_scheduler.T_mult ** i)
    print(f"{stat['cycle']:<8} {stat['duration']:<8} {stat['max_lr']:<12.2e} {expected_period:.1f} epochs")

print("="*70 + "\n")


# ============================================================
# STAGE A
# ============================================================

callbacks = [
    StageAwarePlotCallback(
        val_loader=val_loader,
        param_names=model.param_names,
        num_examples=1,
        save_dir=f"./figs_local/stageA_{suffix_A}",
        stage_tag=f"StageA_{suffix_A.upper()}",
        refine=False,
        use_gt_for_provided=True,
        recon_PT="pred",
        max_val_batches=10
    ),
    PT_PredVsExp_VisuCallback(
        val_loader,
        save_dir=f"./figs_local/PT_A_{suffix_A}",
        num_examples=1,
        tag=f"PT_A_{suffix_A}",
        use_gt_for_provided=True
    ),
    LossCurvePlotCallback(
        save_path=f"./figs_local/stageA_{suffix_A}/loss_curves.png"
    ),
    UpdateEpochInDataset(),
    UpdateEpochInValDataset()
]

print("\n" + "="*70)
print("🚀 ENTRAÎNEMENT STAGE A")
print("="*70)

model = train_stage_A(
    model, train_loader, val_loader,
    epochs=10,
    base_lr=1e-4,
    heads_subset=["sig0","dsig","P","T","mf_CH4", "baseline1","baseline2"],
    use_exp_params=USE_EXP_PARAMS_IN_A,
    callbacks=callbacks,
    **tkw,
)

preview_peak_weights_on_val(
    model, 
    val_loader, 
    save_dir="./figs_local/weights_preview"
)

# Sauvegarde
os.makedirs("./checkpoints", exist_ok=True)
checkpoint_A = {
    'epoch': 170,
    'stage': f'A_{suffix_A}',
    'model_state_dict': model.state_dict(),
    'use_exp_params': USE_EXP_PARAMS_IN_A,
}
checkpoint_path_A = f"./checkpoints/stageA_{suffix_A}.ckpt"
torch.save(checkpoint_A, checkpoint_path_A)
print(f"✅ Stage A sauvegardé: {checkpoint_path_A}\n")

