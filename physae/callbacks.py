"""Lightning callbacks used for monitoring PhysAE training."""

from __future__ import annotations

from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from IPython.display import display, update_display
from pytorch_lightning.callbacks import Callback

try:  # pragma: no cover - optional cosmetic configuration
    mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI Emoji", "Segoe UI Symbol"]
    mpl.rcParams["axes.unicode_minus"] = False
except Exception:  # pragma: no cover
    pass


class PlotAndMetricsCallback(Callback):
    """Display spectra, residuals and key metrics during training."""

    def __init__(self, val_loader, param_names: Iterable[str], num_examples: int = 1, display_id: str = "physae-monitor"):
        super().__init__()
        self.val_loader = val_loader
        self.param_names = list(param_names)
        self.num_examples = int(num_examples)
        self.display_id = display_id
        self._first = True

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        pl_module.eval()
        batch = next(iter(self.val_loader))
        noisy = batch["noisy_spectra"].to(pl_module.device)
        clean = batch["clean_spectra"].to(pl_module.device)
        params_true = batch["params"].to(pl_module.device)
        scale = batch["scale"].to(pl_module.device)

        with torch.no_grad():
            outputs = pl_module(noisy, params_true=params_true, provided_phys=batch.get("provided_phys"))
            spectra_recon = outputs["recon"]

        train_loss = trainer.callback_metrics.get("train_loss", torch.tensor(float("nan")))
        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("nan")))
        val_phys = trainer.callback_metrics.get("val_loss_phys", torch.tensor(float("nan")))
        val_phys_corr = trainer.callback_metrics.get("val_loss_phys_corr", torch.tensor(float("nan")))
        val_param = trainer.callback_metrics.get("val_loss_param", torch.tensor(float("nan")))

        def fmt(value):
            return f"{float(value):.4f}" if torch.is_tensor(value) else str(value)

        metrics_text = {
            "train_loss": fmt(train_loss),
            "val_loss": fmt(val_loss),
            "val_phys": fmt(val_phys),
            "val_corr": fmt(val_phys_corr),
            "val_param": fmt(val_param),
        }

        def get_metric(name):
            value = trainer.callback_metrics.get(name)
            return fmt(value) if value is not None else "--"

        per_param_lines = [
            f"{name:10s} : {get_metric('val_loss_param_' + name)}" for name in self.param_names
        ]
        per_param_text = "\n".join(per_param_lines)

        fig = plt.figure(figsize=(11, 6), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.8], height_ratios=[3, 1], hspace=0.25, wspace=0.3)
        ax_spec = fig.add_subplot(grid[0, 0])
        ax_res = fig.add_subplot(grid[1, 0], sharex=ax_spec)
        ax_tbl = fig.add_subplot(grid[:, 1])
        ax_tbl.axis("off")

        x = torch.arange(noisy.size(1), device=noisy.device)
        ax_spec.plot(x.cpu(), noisy[0].cpu(), label="Noisy", lw=1, alpha=0.7)
        ax_spec.plot(x.cpu(), clean[0].cpu(), label="Clean (réel)", lw=1.5)
        ax_spec.plot(x.cpu(), spectra_recon[0].cpu(), label="Reconstruit", lw=1.2, ls="--")
        ax_spec.set_ylabel("Transmission")
        ax_spec.set_title(f"Epoch {trainer.current_epoch}")
        ax_spec.legend(frameon=False, fontsize=9)

        resid_clean = spectra_recon - clean
        resid_noisy = spectra_recon - noisy
        ax_res.plot(x.cpu(), resid_noisy[0].cpu(), lw=1, label="Reconstruit - Noisy")
        ax_res.plot(x.cpu(), resid_clean[0].cpu(), lw=1.2, label="Reconstruit - Clean")
        ax_res.axhline(0, ls=":", lw=0.8)
        ax_res.set_xlabel("Points spectraux")
        ax_res.set_ylabel("Résidu")
        ax_res.legend(frameon=False, fontsize=9)

        header = f"Metriques (epoch {trainer.current_epoch})"
        lines = [
            f"train_loss : {metrics_text['train_loss']}",
            f"val_loss   : {metrics_text['val_loss']}",
            f"val_phys   : {metrics_text['val_phys']}",
            f"val_corr   : {metrics_text['val_corr']}",
            f"val_param  : {metrics_text['val_param']}",
            "",
            "Pertes par paramètre (val) :",
            per_param_text,
        ]
        ax_tbl.text(0.02, 0.98, header, va="top", ha="left", fontsize=12, fontweight="bold")
        ax_tbl.text(0.02, 0.90, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

        for ax in (ax_spec, ax_res):
            ax.grid(alpha=0.25)

        if getattr(self, "_first", True):
            display(fig, display_id=self.display_id)
            self._first = False
        else:
            update_display(fig, display_id=self.display_id)
        plt.close(fig)


class UpdateEpochInDataset(Callback):
    def on_train_epoch_start(self, trainer, pl_module):  # type: ignore[override]
        dataset = (
            trainer.train_dataloaders.dataset
            if hasattr(trainer, "train_dataloaders")
            else trainer.train_dataloader.dataset
        )
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(trainer.current_epoch)
