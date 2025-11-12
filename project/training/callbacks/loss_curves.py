"""Training callbacks for plotting loss curves during experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

_ROOT_PACKAGE = __name__.partition(".")[0]


def _resolve_plotting_utils():
    if _ROOT_PACKAGE == "project":
        from project.utils.plotting import (  # type: ignore[import]
            ensure_nature_methods_style,
            save_fig,
        )
        return ensure_nature_methods_style, save_fig

    try:
        from utils.plotting import (  # type: ignore[import]
            ensure_nature_methods_style,
            save_fig,
        )
        return ensure_nature_methods_style, save_fig
    except ImportError:
        from project.utils.plotting import (  # type: ignore[import]
            ensure_nature_methods_style,
            save_fig,
        )
        return ensure_nature_methods_style, save_fig


ensure_nature_methods_style, save_fig = _resolve_plotting_utils()
ensure_nature_methods_style()

__all__ = ["LossCurvePlotCallback"]


class LossCurvePlotCallback(pl.Callback):
    """Persist loss curves mirroring the original ``physae.py`` helper.

    The original notebook maintained a rolling history of the aggregated
    Lightning metrics at both train/validation frequency and rendered five
    dedicated panels (total loss + per-loss contributions) together with a
    textual summary.  This callback preserves that behaviour so downstream
    scripts relying on the side-effects from ``physae.py`` continue to work
    when importing the modular package.
    """

    #: metrics tracked both for train/val and displayed in the panel grid
    _METRIC_NAMES: tuple[str, ...] = (
        "train_loss",
        "val_loss",
        "train_loss_pointwise",
        "val_loss_pointwise",
        "train_loss_spectral",
        "val_loss_spectral",
        "train_loss_peak",
        "val_loss_peak",
        "train_loss_params",
        "val_loss_params",
    )

    def __init__(self, save_path: str | Path = "./figs_loss/loss_curves.png") -> None:
        super().__init__()
        self.save_path = Path(save_path)
        self._history: Dict[int, Dict[str, float]] = {}

    def _record(self, epoch: int, metrics: Dict[str, torch.Tensor | float]) -> None:
        if epoch not in self._history:
            self._history[epoch] = {}

        store = self._history[epoch]
        for name in self._METRIC_NAMES:
            if name not in metrics:
                continue
            value = metrics[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            try:
                store[name] = float(value)
            except Exception:
                # Lightning can surface tensors without concrete values (NaN/infs);
                # keep the previous value in that scenario to mimic the script.
                continue

        self._plot()

    def _series(self, names: Iterable[str]) -> dict[str, list[float | None]]:
        epochs = sorted(self._history)
        return {
            name: [self._history[e].get(name) for e in epochs]
            for name in names
        }

    def _plot(self) -> None:
        if not self._history:
            return

        epochs = sorted(self._history)
        epoch_axis = [e + 1 for e in epochs]
        series = self._series(self._METRIC_NAMES)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Courbes d'entraînement", fontsize=14, fontweight="bold")
        ax = axes.flatten()

        def _plot_pair(index: int, train_key: str, val_key: str, title: str) -> None:
            train_values = series[train_key]
            val_values = series[val_key]
            if any(v is not None for v in train_values):
                ax[index].plot(epoch_axis, train_values, marker="o", label="Train")
            if any(v is not None for v in val_values):
                ax[index].plot(epoch_axis, val_values, marker="s", label="Val")
            ax[index].set_title(title)
            ax[index].legend()
            ax[index].grid(alpha=0.3, ls="--")

        _plot_pair(0, "train_loss", "val_loss", "Loss Totale")
        _plot_pair(1, "train_loss_pointwise", "val_loss_pointwise", "Pointwise (MSE)")
        _plot_pair(2, "train_loss_spectral", "val_loss_spectral", "Spectral (Angle)")
        _plot_pair(3, "train_loss_peak", "val_loss_peak", "Peak (Weighted)")
        _plot_pair(4, "train_loss_params", "val_loss_params", "Params (phys.)")

        ax[5].axis("off")
        last_epoch = epochs[-1]
        metrics = self._history[last_epoch]
        summary_lines = [
            f"Epoch {last_epoch + 1}",
            f"Loss Totale     | T: {metrics.get('train_loss', float('nan')):.6g}"
            f" | V: {metrics.get('val_loss', float('nan')):.6g}",
            f"Pointwise (MSE) | T: {metrics.get('train_loss_pointwise', float('nan')):.6g}"
            f" | V: {metrics.get('val_loss_pointwise', float('nan')):.6g}",
            f"Spectral Angle  | T: {metrics.get('train_loss_spectral', float('nan')):.6g}"
            f" | V: {metrics.get('val_loss_spectral', float('nan')):.6g}",
            f"Peak (weighted) | T: {metrics.get('train_loss_peak', float('nan')):.6g}"
            f" | V: {metrics.get('val_loss_peak', float('nan')):.6g}",
            f"Params (phys.)  | T: {metrics.get('train_loss_params', float('nan')):.6g}"
            f" | V: {metrics.get('val_loss_params', float('nan')):.6g}",
        ]
        ax[5].text(
            0.05,
            0.95,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.3),
        )

        plt.tight_layout()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, self.save_path, dpi=150)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning callback signature
        self._record(trainer.current_epoch, trainer.callback_metrics)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning callback signature
        self._record(trainer.current_epoch, trainer.callback_metrics)
