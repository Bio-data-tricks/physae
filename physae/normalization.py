"""Normalisation helpers for PhysAE parameters."""

from __future__ import annotations

import math

import torch

from .config import LOG_FLOOR, LOG_SCALE_PARAMS, NORM_PARAMS


def norm_param_value(name: str, value: float) -> float:
    """Normalise a scalar parameter value into ``[0, 1]``.

    The logic mirrors the original implementation but is now expressed in a
    dedicated function that can be reused both by the dataset and by external
    tooling.
    """

    vmin, vmax = NORM_PARAMS[name]
    if name in LOG_SCALE_PARAMS:
        vmin = max(vmin, LOG_FLOOR)
        vmax = max(vmax, vmin * (1.0 + 1e-12))
        value = max(value, LOG_FLOOR)
        lval, lmin, lmax = math.log10(value), math.log10(vmin), math.log10(vmax)
        return (lval - lmin) / (lmax - lmin)
    return (value - vmin) / (vmax - vmin)


def norm_param_tensor(name: str, values: torch.Tensor) -> torch.Tensor:
    """Vectorised version of :func:`norm_param_value`."""

    vmin, vmax = NORM_PARAMS[name]
    vmin_t = torch.as_tensor(vmin, dtype=values.dtype, device=values.device)
    vmax_t = torch.as_tensor(vmax, dtype=values.dtype, device=values.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        vmax_t = torch.maximum(vmax_t, vmin_t * (1 + torch.finfo(values.dtype).eps))
        values = torch.clamp(values, min=LOG_FLOOR)
        lval = torch.log10(values)
        lmin = torch.log10(vmin_t)
        lmax = torch.log10(vmax_t)
        return (lval - lmin) / (lmax - lmin)
    return (values - vmin_t) / (vmax_t - vmin_t)


def unnorm_param_tensor(name: str, values: torch.Tensor) -> torch.Tensor:
    """Convert normalised values back to their physical range."""

    vmin, vmax = NORM_PARAMS[name]
    vmin_t = torch.as_tensor(vmin, dtype=values.dtype, device=values.device)
    vmax_t = torch.as_tensor(vmax, dtype=values.dtype, device=values.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        vmax_t = torch.maximum(vmax_t, vmin_t * (1 + torch.finfo(values.dtype).eps))
        lmin = torch.log10(vmin_t)
        lmax = torch.log10(vmax_t)
        lval = values * (lmax - lmin) + lmin
        return torch.pow(10.0, lval)
    return values * (vmax_t - vmin_t) + vmin_t


# Backwards compatibility with the original monolithic script -----------------
norm_param_torch = norm_param_tensor
unnorm_param_torch = unnorm_param_tensor
