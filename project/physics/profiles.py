"""
Spectral line profile functions (Voigt, Pine with Dicke narrowing, line mixing).
"""
import math
import torch
from .constants import SQRT_LN2, INV_SQRT_PI, P0, TREF

# Humlíček coefficients for Faddeeva function (wofz)
_b = torch.tensor(
    [-0.0173 - 0.0463j, -0.7399 + 0.8395j, 5.8406 + 0.9536j, -5.5834 - 11.2086j],
    dtype=torch.cdouble
)
_b = torch.cat((_b, _b.conj()))

_c = torch.tensor(
    [2.2377 - 1.626j, 1.4652 - 1.7896j, 0.8393 - 1.892j, 0.2739 - 1.9418j],
    dtype=torch.cdouble
)
_c = torch.cat((_c, -_c.conj()))


def wofz_torch(z: torch.Tensor) -> torch.Tensor:
    """
    Faddeeva/complex error function (vectorized PyTorch implementation).

    Args:
        z: Complex input tensor.

    Returns:
        Faddeeva function w(z).
    """
    b_loc = _b.to(device=z.device, dtype=z.dtype)
    c_loc = _c.to(device=z.device, dtype=z.dtype)
    w_pos = (b_loc / (z.unsqueeze(-1) - c_loc)).sum(dim=-1) * (1j * INV_SQRT_PI)
    w_neg = (b_loc / ((-z).unsqueeze(-1) - c_loc)).sum(dim=-1) * (1j * INV_SQRT_PI)
    return torch.where(z.imag < 0, 2.0 * torch.exp(-(z ** 2)) - w_neg, w_pos)


def pine_profile_torch_complex(x, sigma_hwhm, gamma, g_dicke):
    """
    Pine profile with Dicke broadening (Hartmann-Tran model).

    Args:
        x: Frequency detuning.
        sigma_hwhm: Half-width at half-maximum for Doppler broadening.
        gamma: Collisional broadening parameter.
        g_dicke: Dicke narrowing parameter.

    Returns:
        Tuple of (real_profile, imag_profile).
    """
    xh = SQRT_LN2 * x / sigma_hwhm
    yh = SQRT_LN2 * gamma / sigma_hwhm
    zD = SQRT_LN2 * g_dicke / sigma_hwhm
    z = xh + 1j * (yh + zD)
    k = -wofz_torch(z)
    k_r, k_i = k.real, k.imag
    pi_sqrt = math.sqrt(math.pi)
    denom = (1 - zD * pi_sqrt * k_r) ** 2 + (zD * pi_sqrt * k_i) ** 2
    real = (k_r - zD * pi_sqrt * (k_r ** 2 + k_i ** 2)) / denom
    imag = k_i / denom
    factor = math.sqrt(math.log(2.0) / math.pi) / sigma_hwhm
    return real * factor, imag * factor


def apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T, P, *, PREF=P0, TREF_=TREF):
    """
    Apply line mixing correction with temperature/pressure dependence.

    Args:
        real_prof: Real part of line profile.
        imag_prof: Imaginary part of line profile.
        lmf: Line mixing factor.
        nlmf: Temperature exponent for line mixing.
        T: Temperature (K).
        P: Pressure (mbar).
        PREF: Reference pressure (default: P0).
        TREF_: Reference temperature (default: TREF).

    Returns:
        Line profile with line mixing correction (absorption sign).
    """
    flm = lmf * ((TREF_ / T) ** nlmf) * (P / PREF)
    return -(real_prof + imag_prof * flm)


def polyval_torch(coeffs, x):
    """
    Polynomial evaluation (Horner's method in PyTorch).

    Args:
        coeffs: Polynomial coefficients [B, n_coeffs].
        x: Evaluation points [N].

    Returns:
        Polynomial values [B, N].
    """
    powers = torch.arange(coeffs.shape[1], device=coeffs.device, dtype=coeffs.dtype)
    return torch.sum(coeffs.unsqueeze(2) * x.unsqueeze(0).pow(powers.view(1, -1, 1)), dim=1)
