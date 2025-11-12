"""PyTorch Lightning callbacks providing rich visual diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

_ROOT_PACKAGE = __name__.partition(".")[0]


def _resolve_utils():
    if _ROOT_PACKAGE == "project":
        from project.utils.distributed import is_rank0  # type: ignore[import]
        from project.utils.plotting import save_fig  # type: ignore[import]

        return is_rank0, save_fig

    try:
        from utils.distributed import is_rank0  # type: ignore[import]
    except ImportError:
        from project.utils.distributed import is_rank0  # type: ignore[import]

    try:
        from utils.plotting import save_fig  # type: ignore[import]
    except ImportError:
        from project.utils.plotting import save_fig  # type: ignore[import]

    return is_rank0, save_fig


is_rank0, save_fig = _resolve_utils()

__all__ = [
    "StageAwarePlotCallback",
    "PT_PredVsExp_VisuCallback",
]


class StageAwarePlotCallback(pl.Callback):
    """Visualise spectra, residuals and key metrics after each val epoch.

    The implementation mirrors the rich diagnostics implemented inside
    ``physae.py``: a two-panel figure is generated for the first validation
    example while the right panel aggregates textual metrics (global losses,
    per-parameter error, adaptive weights, etc.).  This callback is stage-aware
    and respects the experimental-parameter masking that was added during the
    refactor.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        param_names: Iterable[str],
        *,
        num_examples: int = 1,
        example_mode: str = "cycle",
        save_dir: str | Path | None = None,
        stage_tag: str = "stage",
        refine: bool = True,
        cascade_stages_override: int | None = None,
        use_gt_for_provided: bool = True,
        recon_PT: str = "pred",
        Pexp: torch.Tensor | None = None,
        Texp: torch.Tensor | None = None,
        max_val_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.val_loader = val_loader
        self.param_names = list(param_names)
        self.num_examples = int(num_examples)
        self.example_mode = example_mode.lower().strip()
        job = os.environ.get("SLURM_JOB_ID", "local")
        root = Path(save_dir) if save_dir is not None else Path(f"./figs_{job}")
        self.stage_tag = stage_tag
        self.save_dir = root / self.stage_tag
        self.refine = bool(refine)
        self.cascade_stages_override = cascade_stages_override
        self.use_gt_for_provided = bool(use_gt_for_provided)
        self.recon_PT = recon_PT
        self.Pexp = Pexp
        self.Texp = Texp
        self.max_val_batches = None if max_val_batches is None else int(max_val_batches)

        valid_modes = {"first", "cycle"}
        if self.example_mode not in valid_modes:
            raise ValueError(
                f"example_mode must be one of {sorted(valid_modes)}, got {example_mode!r}"
            )

        self._preview_iterator = None

    def _next_preview_batch(self):
        """Return a batch used for qualitative visualisation."""

        if self.example_mode == "first":
            return next(iter(self.val_loader))

        if self._preview_iterator is None:
            self._preview_iterator = iter(self.val_loader)

        try:
            batch = next(self._preview_iterator)
        except StopIteration:
            self._preview_iterator = iter(self.val_loader)
            batch = next(self._preview_iterator)

        return batch

    def _denorm_subset(self, pl_module, y_norm: torch.Tensor, names: List[str]) -> torch.Tensor:
        return pl_module._denorm_params_subset(y_norm, names)

    @torch.no_grad()
    def _compute_val_stats(self, pl_module) -> tuple[float, Dict[str, float]]:
        device = pl_module.device
        pred_names = list(getattr(pl_module, "predict_params", []))
        if len(pred_names) == 0:
            return float("nan"), {}

        exp_params_A = getattr(pl_module, "_use_exp_params_in_A", [])
        exp_params_resid = getattr(pl_module, "_use_exp_params_for_resid", [])
        exp_params = set(exp_params_A) | set(exp_params_resid)
        eval_params = [p for p in pred_names if p not in exp_params]
        if len(eval_params) == 0:
            return float("nan"), {}

        mse_sum = 0.0
        n_points_total = 0
        err_sum: Dict[str, float] = {p: 0.0 for p in eval_params}
        err_cnt = 0
        eps = 1e-12

        for b_idx, batch in enumerate(self.val_loader):
            noisy = batch["noisy_spectra"].to(device)
            clean = batch["clean_spectra"].to(device)
            p_norm = batch["params"].to(device)
            B, N = noisy.shape

            provided_phys: Dict[str, torch.Tensor] = {}
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
                Pexp=self.Pexp,
                Texp=self.Texp,
                cascade_stages_override=self.cascade_stages_override,
            )
            recon = out["spectra_recon"]
            y_full_pred = out["y_phys_full"]

            diff = (recon - clean).float()
            mse_sum += float((diff * diff).sum().item())
            n_points_total += B * N

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
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning signature
        if not is_rank0():
            return

        pl_module.eval()
        device = pl_module.device

        try:
            batch = self._next_preview_batch()
        except StopIteration:
            return

        noisy = batch["noisy_spectra"][: self.num_examples].to(device)
        clean = batch["clean_spectra"][: self.num_examples].to(device)
        params_true_norm = batch["params"][: self.num_examples].to(device)

        provided_phys: Dict[str, torch.Tensor] = {}
        if self.use_gt_for_provided:
            for name in getattr(pl_module, "provided_params", []):
                idx = pl_module.name_to_idx[name]
                v_norm = params_true_norm[:, idx]
                v_phys = pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]
                provided_phys[name] = v_phys

        out = pl_module.infer(
            noisy,
            provided_phys=provided_phys,
            refine=self.refine,
            resid_target="input",
            recon_PT=self.recon_PT,
            Pexp=self.Pexp,
            Texp=self.Texp,
            cascade_stages_override=self.cascade_stages_override,
        )

        spectra_recon = out["spectra_recon"].detach().cpu()
        noisy_cpu, clean_cpu = noisy.detach().cpu(), clean.detach().cpu()
        x = np.arange(clean_cpu.shape[1])

        val_mse, mean_pct = self._compute_val_stats(pl_module)

        metrics = trainer.callback_metrics

        def get_metric(name: str, default: str = "-") -> str:
            value = metrics.get(name, None)
            try:
                return f"{float(value):.6g}"
            except Exception:
                return default

        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[3, 1], hspace=0.30, wspace=0.35)
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0], sharex=ax_spec)
        ax_tbl = fig.add_subplot(gs[:, 1])
        ax_tbl.axis("off")

        idx = 0
        ax_spec.plot(x, noisy_cpu[idx], label="Noisy", lw=1, alpha=0.7, color="gray")
        ax_spec.plot(x, clean_cpu[idx], label="Clean (GT)", lw=1.5, color="tab:blue")
        ax_spec.plot(x, spectra_recon[idx], label="Reconstruit", lw=1.2, ls="--", color="tab:orange")
        ax_spec.set_ylabel("Transmission", fontsize=10)
        ax_spec.set_title(f"{self.stage_tag} — Epoch {trainer.current_epoch}", fontsize=11, fontweight="bold")
        ax_spec.legend(frameon=False, fontsize=9, loc="upper right")
        ax_spec.grid(alpha=0.25, ls="--", lw=0.5)

        resid_clean = spectra_recon[idx] - clean_cpu[idx]
        resid_noisy = spectra_recon[idx] - noisy_cpu[idx]
        ax_res.plot(x, resid_noisy, lw=1, label="Recon - Noisy", color="tab:gray")
        ax_res.plot(x, resid_clean, lw=1.2, label="Recon - Clean", color="tab:red")
        ax_res.axhline(0, ls=":", lw=0.8, color="black")
        ax_res.set_xlabel("Index spectral", fontsize=10)
        ax_res.set_ylabel("Résidu", fontsize=10)
        ax_res.legend(frameon=False, fontsize=9)
        ax_res.grid(alpha=0.25, ls="--", lw=0.5)

        exp_params_A = getattr(pl_module, "_use_exp_params_in_A", [])
        exp_params_resid = getattr(pl_module, "_use_exp_params_for_resid", [])
        exp_params_set = set(exp_params_A) | set(exp_params_resid)

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
            f"peak (weighted)  : {get_metric('val_loss_peak')}",
            f"params (physique): {get_metric('val_loss_params')}",
            "",
        ]

        if getattr(pl_module, "use_relobralo_top", False):
            lines.extend([
                "─────────────────────────────────────",
                "  Poids ReLoBRaLo (adaptatifs)",
                "────────────────────────────────────",
                f"w_pointwise : {get_metric('relo_weight_phys_pointwise')}",
                f"w_spectral  : {get_metric('relo_weight_phys_spectral')}",
                f"w_peak      : {get_metric('relo_weight_phys_peak')}",
                f"w_params    : {get_metric('relo_weight_param_group')}",
                "",
            ])

        lines.extend([
            "────────────────────────────────────",
            "  Statistiques Val Globales",
            "─────────────────────────────────────",
            f"MSE(clean,recon) : {val_mse:.6g}",
        ])

        if len(mean_pct) > 0:
            lines.append("")
            lines.append("Erreur % moyenne par paramètre:")
            for key in sorted(mean_pct.keys()):
                lines.append(f"  {key:12s} : {mean_pct[key]:6.3f} %")

            if len(exp_params_set) > 0:
                lines.append("")
                lines.append("Paramètres expérimentaux (0%):")
                for key in sorted(exp_params_set):
                    if key in getattr(pl_module, "predict_params", []):
                        lines.append(f"  {key:12s} : EXP (GT utilisé)")

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

        if len(exp_params_A) > 0:
            lines.append(f"exp_in_A={','.join(exp_params_A)}")
        if len(exp_params_resid) > 0:
            lines.append(f"exp_in_resid={','.join(exp_params_resid)}")

        ax_tbl.text(
            0.05,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
            transform=ax_tbl.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="wheat",
                alpha=0.3,
                edgecolor="gray",
                linewidth=1,
            ),
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_png = self.save_dir / f"{self.stage_tag}_val_epoch{trainer.current_epoch:04d}.png"
        save_fig(fig, out_png, dpi=150)

        if trainer.is_global_zero:
            print(f"✅ Figure sauvegardée: {out_png}")


class PT_PredVsExp_VisuCallback(pl.Callback):
    """Compare reconstructions using predicted vs experimental PT values."""

    def __init__(
        self,
        val_loader: DataLoader,
        save_dir: str | Path = "./figs_local",
        num_examples: int = 1,
        tag: str = "PT_pred_vs_exp",
        *,
        force_Pexp: torch.Tensor | None = None,
        force_Texp: torch.Tensor | None = None,
        use_gt_for_provided: bool = True,
    ) -> None:
        super().__init__()
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.num_examples = int(num_examples)
        self.tag = tag
        self.force_Pexp = force_Pexp
        self.force_Texp = force_Texp
        self.use_gt_for_provided = bool(use_gt_for_provided)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning signature
        if hasattr(pl_module, "eval"):
            pl_module.eval()
        device = pl_module.device
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        noisy = batch["noisy_spectra"][ : self.num_examples].to(device)
        clean = batch["clean_spectra"][ : self.num_examples].to(device)
        p_norm = batch["params"][ : self.num_examples].to(device)
        B, N = noisy.shape
        x = np.arange(N)

        provided_phys: Dict[str, torch.Tensor] = {}
        for name in getattr(pl_module, "provided_params", []):
            idx = pl_module.name_to_idx[name]
            v_norm = p_norm[:, idx]
            v_phys = pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]
            provided_phys[name] = v_phys

        def _gt_phys(name: str) -> torch.Tensor:
            idx = pl_module.name_to_idx[name]
            v_norm = p_norm[:, idx]
            return pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]

        if self.force_Pexp is not None and self.force_Texp is not None:
            Pexp = self.force_Pexp.to(device).view(B)
            Texp = self.force_Texp.to(device).view(B)
        else:
            Pexp = _gt_phys("P").to(device)
            Texp = _gt_phys("T").to(device)

        out_pred = pl_module.infer(
            noisy,
            provided_phys=provided_phys,
            refine=True,
            resid_target="input",
            recon_PT="pred",
        )
        recon_pred = out_pred["spectra_recon"]

        out_exp = pl_module.infer(
            noisy,
            provided_phys=provided_phys,
            refine=True,
            resid_target="input",
            recon_PT="exp",
            Pexp=Pexp,
            Texp=Texp,
        )
        recon_exp = out_exp["spectra_recon"]

        for i in range(B):
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.25, wspace=0.25)

            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(x, noisy[i].detach().cpu(), lw=0.9, alpha=0.7, label="Noisy", color="gray")
            ax0.plot(x, clean[i].detach().cpu(), lw=1.2, label="Clean", color="tab:blue")
            ax0.plot(x, recon_pred[i].detach().cpu(), lw=1.0, ls="--", label="Recon (PT=pred)", color="tab:orange")
            ax0.plot(x, recon_exp[i].detach().cpu(), lw=1.0, ls="-.", label="Recon (PT=exp)", color="tab:green")
            ax0.set_title(f"{self.tag} — epoch {trainer.current_epoch} — ex {i + 1}", fontweight="bold")
            ax0.set_ylabel("Transmission")
            ax0.legend(frameon=False, fontsize=9)
            ax0.grid(alpha=0.3)

            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax1.plot(x, (recon_pred[i] - noisy[i]).detach().cpu(), lw=0.9, label="(PT=pred) - Noisy", color="tab:orange")
            ax1.plot(x, (recon_exp[i] - noisy[i]).detach().cpu(), lw=0.9, label="(PT=exp) - Noisy", color="tab:green")
            ax1.axhline(0, ls=":", lw=0.8, color="k")
            ax1.set_xlabel("Index spectral")
            ax1.set_ylabel("Résidu")
            ax1.legend(frameon=False, fontsize=8)
            ax1.grid(alpha=0.3)

            ax2 = fig.add_subplot(gs[1, 1], sharex=ax0)
            ax2.plot(x, (recon_pred[i] - clean[i]).detach().cpu(), lw=0.9, label="(PT=pred) - Clean", color="tab:orange")
            ax2.plot(x, (recon_exp[i] - clean[i]).detach().cpu(), lw=0.9, label="(PT=exp) - Clean", color="tab:green")
            ax2.axhline(0, ls=":", lw=0.8, color="k")
            ax2.set_xlabel("Index spectral")
            ax2.set_ylabel("Résidu")
            ax2.legend(frameon=False, fontsize=8)
            ax2.grid(alpha=0.3)

            try:
                idx_P = pl_module.name_to_idx["P"]
                idx_T = pl_module.name_to_idx["T"]
                Pp = float(out_pred["y_phys_full"][i, idx_P])
                Tp = float(out_pred["y_phys_full"][i, idx_T])
            except Exception:
                Pp = float("nan")
                Tp = float("nan")

            info_text = (
                f"PT_pred ≈ ({Pp:.2f} mbar, {Tp:.2f} K)\n"
                f"PT_exp  = ({float(Pexp[i]):.2f} mbar, {float(Texp[i]):.2f} K)"
            )

            ax0.text(
                0.02,
                0.02,
                info_text,
                transform=ax0.transAxes,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.4),
            )

            fname = self.save_dir / f"{self.tag}_epoch{trainer.current_epoch:04d}_ex{i + 1}.png"
            fig.savefig(fname, dpi=160, bbox_inches="tight")
            plt.close(fig)

            if trainer.is_global_zero:
                print(f"✅ Figure PT sauvegardée : {fname}")
