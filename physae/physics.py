"""Spectroscopic forward model utilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch

MOLECULE_PARAMS = {
    "CH4": {"M": 16.04, "PL": 15.12},
    "H2O": {"M": 18.02, "PL": 15.12},
}

_b = torch.tensor(
    [-0.0173 - 0.0463j, -0.7399 + 0.8395j, 5.8406 + 0.9536j, -5.5834 - 11.2086j],
    dtype=torch.cdouble,
)
_b = torch.cat((_b, _b.conj()))
_c = torch.tensor(
    [2.2377 - 1.626j, 1.4652 - 1.7896j, 0.8393 - 1.892j, 0.2739 - 1.9418j],
    dtype=torch.cdouble,
)
_c = torch.cat((_c, -_c.conj()))


def parse_csv_transitions(csv_str: str) -> List[Dict[str, float]]:
    """Parse a custom semi-colon separated transitions table."""

    transitions = []
    for line in csv_str.strip().splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        tokens = [token.strip() for token in line.split(";")]
        while len(tokens) < 14:
            tokens.append("0")
        transitions.append(
            {
                "mid": int(tokens[0]),
                "lid": int(float(tokens[1])),
                "center": float(tokens[2]),
                "amplitude": float(tokens[3]),
                "gamma_air": float(tokens[4]),
                "gamma_self": float(tokens[5]),
                "e0": float(tokens[6]),
                "n_air": float(tokens[7]),
                "shift_air": float(tokens[8]),
                "abundance": float(tokens[9]),
                "gDicke": float(tokens[10]),
                "nDicke": float(tokens[11]),
                "lmf": float(tokens[12]),
                "nlmf": float(tokens[13]),
            }
        )
    return transitions


def transitions_to_tensors(transitions, device) -> List[torch.Tensor]:
    """Convert a list of transition dictionaries to tensors."""

    keys = [
        "amplitude",
        "center",
        "gamma_air",
        "gamma_self",
        "n_air",
        "shift_air",
        "gDicke",
        "nDicke",
        "lmf",
        "nlmf",
    ]
    return [
        torch.tensor([transition[key] for transition in transitions], dtype=torch.float32, device=device)
        for key in keys
    ]


def wofz_torch(z: torch.Tensor) -> torch.Tensor:
    """Numerically stable Faddeeva function implementation."""

    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
    b_local = _b.to(device=z.device, dtype=z.dtype)
    c_local = _c.to(device=z.device, dtype=z.dtype)
    w = (b_local / (z.unsqueeze(-1) - c_local)).sum(dim=-1)
    w = w * (1j * inv_sqrt_pi)
    mask = z.imag < 0
    reflected = torch.exp(-(z**2)) * 2.0 - w.conj()
    return torch.where(mask, reflected, w)


def pine_profile_torch_complex(x, sigma_hwhm, gamma, gDicke, *, device="cpu"):
    """Pine line profile returning real and imaginary parts."""

    sigma = sigma_hwhm / math.sqrt(2 * math.log(2.0))
    xh = math.sqrt(math.log(2.0)) * x / sigma_hwhm
    yh = math.sqrt(math.log(2.0)) * gamma / sigma_hwhm
    zD = math.sqrt(math.log(2.0)) * gDicke / sigma_hwhm
    z = xh + 1j * (yh + zD)
    k = -wofz_torch(z)
    k_r, k_i = k.real, k.imag
    denom = (1 - zD * math.sqrt(math.pi) * k_r) ** 2 + (zD * math.sqrt(math.pi) * k_i) ** 2
    real = (k_r - zD * math.sqrt(math.pi) * (k_r**2 + k_i**2)) / denom
    imag = k_i / denom
    factor = math.sqrt(math.log(2.0) / math.pi) / sigma
    return real * factor, imag * factor


def apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T, TREF=296.0):
    """Apply line mixing following the original empirical rule."""

    flm = lmf * ((T / TREF) ** nlmf)
    return real_prof + imag_prof * flm


def polyval_torch(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a polynomial using broadcasting-friendly tensors."""

    powers = torch.arange(coeffs.shape[1], device=coeffs.device, dtype=coeffs.dtype)
    return torch.sum(coeffs.unsqueeze(2) * x.unsqueeze(0).pow(powers.view(1, -1, 1)), dim=1)


def batch_physics_forward_multimol_vgrid(
    sig0,
    dsig,
    poly_freq,
    v_grid_idx,
    baseline_coeffs,
    transitions_dict,
    P,
    T,
    mf_dict,
    device="cpu",
):
    """Forward-model spectra for a batch of parameters."""

    batch, n_points = sig0.shape[0], v_grid_idx.shape[0]
    v_grid_idx = v_grid_idx.to(device=device, dtype=torch.float64)
    sig0, dsig, P, T = (
        sig0.to(dtype=torch.float64).unsqueeze(1),
        dsig.to(dtype=torch.float64).unsqueeze(1),
        P.to(dtype=torch.float64).unsqueeze(1),
        T.to(dtype=torch.float64).unsqueeze(1),
    )
    if baseline_coeffs.dim() == 1:
        baseline_coeffs = baseline_coeffs.unsqueeze(0)
    baseline_coeffs = baseline_coeffs.to(dtype=torch.float64)

    poly_freq_torch = torch.tensor(poly_freq, dtype=torch.float64, device=device).unsqueeze(0).expand(batch, -1)
    coeffs = torch.cat([sig0, dsig, poly_freq_torch], dim=1)
    v_grid_batch = polyval_torch(coeffs, v_grid_idx)

    total_profile = torch.zeros((batch, n_points), device=device, dtype=torch.float64)

    C = torch.tensor(2.99792458e10, dtype=torch.float64, device=device)
    NA = torch.tensor(6.02214129e23, dtype=torch.float64, device=device)
    KB = torch.tensor(1.380649e-16, dtype=torch.float64, device=device)
    P0 = torch.tensor(1013.25, dtype=torch.float64, device=device)
    T0 = torch.tensor(273.15, dtype=torch.float64, device=device)
    L0 = torch.tensor(2.6867773e19, dtype=torch.float64, device=device)
    TREF = torch.tensor(296.0, dtype=torch.float64, device=device)

    for mol, transitions in transitions_dict.items():
        tensors = transitions_to_tensors(transitions, device)
        amp, center, ga, gs, na, sa, gd, nd, lmf, nlmf = [t.to(dtype=torch.float64).view(1, -1, 1) for t in tensors]
        mf = mf_dict[mol].to(dtype=torch.float64).view(batch, 1, 1)
        Mmol = torch.tensor(MOLECULE_PARAMS[mol]["M"], dtype=torch.float64, device=device)
        PL = torch.tensor(MOLECULE_PARAMS[mol]["PL"], dtype=torch.float64, device=device)
        T_exp, P_exp, v_grid_exp = T.view(batch, 1, 1), P.view(batch, 1, 1), v_grid_batch.view(batch, 1, n_points)
        x = v_grid_exp - center
        sigma_hwhm = (center / C) * torch.sqrt(2 * NA * KB * T_exp * math.log(2.0) / Mmol)
        gamma = P_exp / P0 * (TREF / T_exp) ** na * (ga * (1 - mf) + gs * mf)
        real_prof, imag_prof = pine_profile_torch_complex(x, sigma_hwhm, gamma, gd, device=device)
        profile = apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T_exp)
        band = profile * amp * PL * 100 * mf * L0 * P_exp / P0 * T0 / T_exp
        total_profile += band.sum(dim=1)

    transmission = torch.exp(-total_profile)

    x_bl = torch.arange(n_points, device=device, dtype=torch.float64)
    powers_bl = torch.arange(baseline_coeffs.shape[1], device=device, dtype=torch.float64)
    baseline = torch.sum(
        baseline_coeffs.unsqueeze(2) * x_bl.unsqueeze(0).pow(powers_bl.view(1, -1, 1)), dim=1
    )

    return transmission * baseline, v_grid_batch
