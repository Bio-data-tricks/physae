"""
Parameter normalization functions.
"""
import math
import torch
from config.params import NORM_PARAMS, LOG_SCALE_PARAMS, LOG_FLOOR


def norm_param_value(name: str, val: float) -> float:
    """
    Normalize a scalar parameter value.

    Args:
        name: Parameter name.
        val: Parameter value to normalize.

    Returns:
        Normalized value in [0, 1].
    """
    vmin, vmax = NORM_PARAMS[name]
    if name in LOG_SCALE_PARAMS:
        vmin = max(vmin, LOG_FLOOR)
        vmax = max(vmax, vmin * (1 + 1e-12))
        val = max(val, LOG_FLOOR)
        lv, lmin, lmax = math.log10(val), math.log10(vmin), math.log10(vmax)
        return (lv - lmin) / (lmax - lmin)
    else:
        return (val - vmin) / (vmax - vmin)


def norm_param_torch(name: str, val_t: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor parameter value.

    Args:
        name: Parameter name.
        val_t: Parameter tensor to normalize.

    Returns:
        Normalized tensor in [0, 1].
    """
    vmin, vmax = NORM_PARAMS[name]
    vmin_t = torch.as_tensor(vmin, dtype=val_t.dtype, device=val_t.device)
    vmax_t = torch.as_tensor(vmax, dtype=val_t.dtype, device=val_t.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        vmax_t = torch.maximum(vmax_t, vmin_t * (1 + torch.finfo(val_t.dtype).eps))
        val_t = torch.clamp(val_t, min=LOG_FLOOR)
        lval = torch.log10(val_t)
        lmin = torch.log10(vmin_t)
        lmax = torch.log10(vmax_t)
        return (lval - lmin) / (lmax - lmin)
    else:
        return (val_t - vmin_t) / (vmax_t - vmin_t)


def unnorm_param_torch(name: str, val_norm_t: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor parameter value.

    Args:
        name: Parameter name.
        val_norm_t: Normalized parameter tensor in [0, 1].

    Returns:
        Denormalized tensor in original range.
    """
    vmin, vmax = NORM_PARAMS[name]
    vmin_t = torch.as_tensor(vmin, dtype=val_norm_t.dtype, device=val_norm_t.device)
    vmax_t = torch.as_tensor(vmax, dtype=val_norm_t.dtype, device=val_norm_t.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        vmax_t = torch.maximum(vmax_t, vmin_t * (1 + torch.finfo(val_norm_t.dtype).eps))
        lmin = torch.log10(vmin_t)
        lmax = torch.log10(vmax_t)
        lval = val_norm_t * (lmax - lmin) + lmin
        return torch.pow(10.0, lval)
    else:
        return val_norm_t * (vmax_t - vmin_t) + vmin_t
