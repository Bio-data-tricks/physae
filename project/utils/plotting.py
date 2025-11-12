"""Plotting utilities and global matplotlib styling helpers."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

from cycler import cycler
import matplotlib.pyplot as plt

__all__ = [
    "PALETTE",
    "tint",
    "shade",
    "use_nature_methods_style",
    "ensure_nature_methods_style",
    "figsize_scaled",
    "save_fig",
]


PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "olive": "#7A8F33",
    "purple": "#CC79A7",
    "red": "#D55E00",
    "orange": "#E69F00",
    "yellow": "#F0E442",
    "brown": "#8C564B",
    "grey0": "#f7f7f7",
    "grey1": "#e6e6e6",
    "grey2": "#cccccc",
    "grey3": "#b3b3b3",
    "grey4": "#999999",
    "grey5": "#7f7f7f",
    "grey6": "#595959",
    "grey7": "#404040",
    "grey8": "#262626",
    "black": "#000000",
}


def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb01_to_hex(rgb: Iterable[float]) -> str:
    r, g, b = (int(round(c * 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def tint(color: str, f: float = 0.2) -> str:
    base = PALETTE.get(color, color)
    r, g, b = _hex_to_rgb01(base)
    r = r + (1 - r) * f
    g = g + (1 - g) * f
    b = b + (1 - b) * f
    return _rgb01_to_hex((r, g, b))


def shade(color: str, f: float = 0.2) -> str:
    base = PALETTE.get(color, color)
    r, g, b = _hex_to_rgb01(base)
    r = r * (1 - f)
    g = g * (1 - f)
    b = b * (1 - f)
    return _rgb01_to_hex((r, g, b))


_STYLE_APPLIED = False


def use_nature_methods_style(
    *, font_size: int = 9, tick_size: int | None = None, line_width: float = 0.6
) -> None:
    """Apply the shared matplotlib style across callbacks and notebooks."""

    global _STYLE_APPLIED
    if tick_size is None:
        tick_size = font_size - 1

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": font_size,
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "axes.linewidth": line_width,
            "xtick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.major.size": 3,
            "ytick.minor.size": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.grid": False,
            "figure.constrained_layout.use": True,
            "savefig.transparent": True,
            "axes.prop_cycle": cycler(
                color=[
                    PALETTE["green"],
                    PALETTE["blue"],
                    PALETTE["orange"],
                    PALETTE["red"],
                    PALETTE["purple"],
                    PALETTE["sky"],
                    PALETTE["olive"],
                    PALETTE["yellow"],
                ]
            ),
        }
    )

    _STYLE_APPLIED = True


def ensure_nature_methods_style() -> None:
    """Idempotently apply the shared matplotlib style."""

    if not _STYLE_APPLIED:
        use_nature_methods_style()


def figsize_scaled(base_w: float, base_h: float, scale: float) -> tuple[float, float]:
    return base_w * scale, base_h * scale


def save_fig(fig, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk using the shared defaults."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


ensure_nature_methods_style()
