"""Evaluation helpers for PhysAE models."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baseline import lowess_max_value
from .normalization import unnorm_param_torch
from .model import PhysicallyInformedAE


@torch.no_grad()
def _baseline_refit_edges(
    model: PhysicallyInformedAE,
    y_full_pred: torch.Tensor,
    recon: torch.Tensor,
    target: torch.Tensor,
    edge_pts: int = 50,
    deg: int = 2,
    iters: int = 1,
):
    assert edge_pts > 0
    batch, length = recon.shape
    device = recon.device
    dtype = torch.float64
    scale_est = lowess_max_value(target, frac=0.08, iters=2, n_eval=64).to(recon.dtype)
    scale_est = torch.clamp(scale_est, min=torch.tensor(1e-6, dtype=scale_est.dtype, device=scale_est.device))
    deg = int(max(0, min(2, deg)))
    used_cols = deg + 1
    assert 2 * edge_pts < length, "edge_pts trop grand par rapport à N"
    x = torch.arange(length, device=device, dtype=dtype)
    A = torch.stack([x ** k for k in range(used_cols)], dim=1).T
    idx_edge = torch.cat([torch.arange(edge_pts, device=device), torch.arange(length - edge_pts, length, device=device)])
    A_edge = A[:, idx_edge].T
    b0 = model.name_to_idx["baseline0"]
    for _ in range(max(1, iters)):
        resid = (target - recon).to(dtype)
        r_edge = resid[:, idx_edge]
        for b in range(batch):
            sol = torch.linalg.lstsq(A_edge, r_edge[b].unsqueeze(-1)).solution
            delta = sol.view(-1)
            y_full_pred[b, b0 : b0 + used_cols] = (y_full_pred[b, b0 : b0 + used_cols].to(dtype) + delta).to(
                y_full_pred.dtype
            )
        recon = model._physics_reconstruction(y_full_pred, device, scale=scale_est)
    return y_full_pred, recon


@torch.no_grad()
def evaluate_and_plot(
    model: PhysicallyInformedAE,
    loader: DataLoader,
    n_show: int = 5,
    refine: bool = True,
    robust_smape: bool = False,
    eps: float = 1e-12,
    seed: int = 123,
    baseline_correction: Dict | None = None,
):
    model.eval()
    device = model.device
    pred_names = list(getattr(model, "predict_params", []))
    if not pred_names:
        raise RuntimeError("Aucun paramètre à évaluer.")
    bc = baseline_correction or {}
    bc_enabled = bool(bc.get("enabled", False))
    bc_edge = int(bc.get("edge_pts", 50))
    bc_deg = int(bc.get("deg", 2))
    bc_iters = int(bc.get("iters", 1))
    per_param_err = {name: [] for name in pred_names}
    show_examples = []
    for batch in loader:
        noisy = batch["noisy_spectra"].to(device)
        clean = batch["clean_spectra"].to(device)
        params_true = batch["params"].to(device)
        provided_phys = {name: params_true[:, model.name_to_idx[name]] for name in model.provided_params}
        outputs = model.infer(noisy, provided_phys, refine=refine)
        pred_phys = outputs["y_phys_full"]
        recon = outputs["spectra_recon"]
        if bc_enabled:
            pred_phys, recon = _baseline_refit_edges(model, pred_phys, recon, clean, edge_pts=bc_edge, deg=bc_deg, iters=bc_iters)
        true_phys = torch.stack(
            [unnorm_param_torch(name, params_true[:, model.name_to_idx[name]]) for name in pred_names], dim=1
        )
        if robust_smape:
            denom = pred_phys.abs() + true_phys.abs() + eps
            err_pct = 100.0 * 2.0 * (pred_phys - true_phys).abs() / denom
        else:
            denom = torch.clamp(true_phys.abs(), min=eps)
            err_pct = 100.0 * (pred_phys - true_phys).abs() / denom
        for j, name in enumerate(pred_names):
            per_param_err[name].append(err_pct[:, j].detach().cpu())
        for i in range(noisy.size(0)):
            if len(show_examples) < n_show:
                show_examples.append(
                    {
                        "noisy": noisy[i].detach().cpu(),
                        "clean": clean[i].detach().cpu(),
                        "recon": recon[i].detach().cpu(),
                        "pred": {name: float(pred_phys[i, j]) for j, name in enumerate(pred_names)},
                        "true": {name: float(true_phys[i, j]) for j, name in enumerate(pred_names)},
                        "errpct": {name: float(err_pct[i, j]) for j, name in enumerate(pred_names)},
                    }
                )
        if len(show_examples) >= n_show:
            break
    rows = []
    for name in pred_names:
        if len(per_param_err[name]) == 0:
            continue
        values = torch.cat(per_param_err[name])
        mean = values.mean().item()
        median = values.median().item()
        try:
            p90 = torch.quantile(values, torch.tensor(0.90)).item()
            p95 = torch.quantile(values, torch.tensor(0.95)).item()
        except Exception:
            arr = values.numpy()
            p90 = float(np.quantile(arr, 0.90))
            p95 = float(np.quantile(arr, 0.95))
        rows.append({"param": name, "mean_%": mean, "median_%": median, "p90_%": p90, "p95_%": p95})
    df = pd.DataFrame(rows).set_index("param").sort_index()
    print("\n=== Erreurs en % (globales, sur le loader) ===")
    print(df.round(4))
    if len(show_examples) > 0:
        rows = len(show_examples)
        fig, axes = plt.subplots(rows, 2, figsize=(12, 2.8 * rows), sharex=False)
        if rows == 1:
            axes = np.array([axes])
        for r, example in enumerate(show_examples):
            x = np.arange(example["clean"].numel())
            ax1, ax2 = axes[r, 0], axes[r, 1]
            ax1.plot(x, example["noisy"], lw=0.9, alpha=0.7, label="Noisy")
            ax1.plot(x, example["clean"], lw=1.2, label="Clean")
            ax1.plot(x, example["recon"], lw=1.0, ls="--", label="Recon")
            ax1.set_ylabel("Transmission")
            lbl_bc = " (+baseline refit)" if bc_enabled else ""
            err_txt = ", ".join([f"{k}: {example['errpct'][k]:.2f}%" for k in pred_names])
            ax1.set_title(f"Exemple {r + 1}{lbl_bc} — erreurs % : {err_txt}", fontsize=10)
            ax1.legend(frameon=False, fontsize=8)
            ax2.plot(x, example["recon"] - example["noisy"], lw=0.9, label="Recon - Noisy")
            ax2.plot(x, example["recon"] - example["clean"], lw=0.9, label="Recon - Clean")
            ax2.axhline(0, color="k", ls=":", lw=0.7)
            ax2.set_ylabel("Résidu")
            ax2.legend(frameon=False, fontsize=8)
        axes[-1, 0].set_xlabel("Index spectral")
        axes[-1, 1].set_xlabel("Index spectral")
        for axrow in axes:
            for ax in axrow:
                ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
    return df
