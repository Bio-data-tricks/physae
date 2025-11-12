"""Utility helpers shared by plotting callbacks and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt

__all__ = [
    "ensure_nature_methods_style",
    "save_fig",
]


_STYLE_APPLIED = False


def ensure_nature_methods_style() -> None:
    """Apply a lightweight plotting style inspired by Nature Methods figures."""

    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    try:
        plt.style.use("seaborn-colorblind")
    except (OSError, ValueError):
        # ``plt.style.use`` raises if the style is missing. The default style is
        # perfectly acceptable in that case.
        pass

    rc_updates: Mapping[str, float | str] = {
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    }
    plt.rcParams.update(rc_updates)

    _STYLE_APPLIED = True


def save_fig(fig, path: str | Path, dpi: int = 150) -> None:
    """Save a Matplotlib figure and close it."""

    ensure_nature_methods_style()
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_obj, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
