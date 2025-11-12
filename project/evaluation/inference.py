"""Evaluation helpers mirroring the original ``physae.py`` workflow."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from project.utils.plotting import save_fig

__all__ = ["evaluate_and_plot"]


@torch.no_grad()
def evaluate_and_plot(
    model,
    loader: DataLoader,
    *,
    n_show: int = 5,
    refine: bool = True,
    robust_smape: bool = False,
    eps: float = 1e-12,
    seed: int = 123,
    baseline_correction: dict | None = None,
    save_dir: str | Path | None = None,
    tag: str | None = None,
):
    """Evaluate parameter errors and plot qualitative examples.

    Parameters match the long-form helper that lived in ``physae.py`` to ease
    migration of existing experiment scripts.  The evaluation excludes
    experimental parameters – i.e. those overridden with ground-truth values
    during training – so that reported statistics only concern genuinely
    predicted quantities.
    """

    model.eval()
    device = model.device
    rng = random.Random(seed)
    _ = (rng, baseline_correction)

    pred_names = list(getattr(model, "predict_params", []))
    if not pred_names:
        raise RuntimeError("Aucun paramètre à évaluer.")

    exp_params_A = getattr(model, "_use_exp_params_in_A", [])
    exp_params_resid = getattr(model, "_use_exp_params_for_resid", [])
    exp_params_set = set(exp_params_A) | set(exp_params_resid)
    eval_params = [p for p in pred_names if p not in exp_params_set]

    if len(eval_params) == 0:
        print("⚠️  Tous les paramètres sont expérimentaux, rien à évaluer.")
        return pd.DataFrame()

    per_param_err = {p: [] for p in eval_params}
    show_examples: list[dict] = []

    for batch in loader:
        noisy = batch["noisy_spectra"].to(device)
        clean = batch["clean_spectra"].to(device)
        p_norm = batch["params"].to(device)
        B = noisy.size(0)

        provided_phys: dict[str, torch.Tensor] = {}
        if len(model.provided_params) > 0:
            cols = [p_norm[:, model.name_to_idx[n]] for n in model.provided_params]
            provided_norm_tensor = torch.stack(cols, dim=1)
            provided_phys_tensor = model._denorm_params_subset(
                provided_norm_tensor, model.provided_params
            )
            for j, name in enumerate(model.provided_params):
                provided_phys[name] = provided_phys_tensor[:, j]

        out = model.infer(
            noisy,
            provided_phys=provided_phys,
            refine=refine,
            resid_target="input",
        )
        recon = out["spectra_recon"]
        y_full_pred = out["y_phys_full"].clone()

        true_cols = [p_norm[:, model.name_to_idx[n]] for n in eval_params]
        true_norm_tensor = torch.stack(true_cols, dim=1)
        true_phys = model._denorm_params_subset(true_norm_tensor, eval_params)
        pred_phys = torch.stack(
            [y_full_pred[:, model.name_to_idx[n]] for n in eval_params], dim=1
        )

        if robust_smape:
            denom = pred_phys.abs() + true_phys.abs() + eps
            err_pct = 100.0 * 2.0 * (pred_phys - true_phys).abs() / denom
        else:
            denom = torch.clamp(true_phys.abs(), min=eps)
            err_pct = 100.0 * (pred_phys - true_phys).abs() / denom

        for j, name in enumerate(eval_params):
            per_param_err[name].append(err_pct[:, j].detach().cpu())

        for i in range(B):
            if len(show_examples) < n_show:
                show_examples.append(
                    {
                        "noisy": noisy[i].detach().cpu(),
                        "clean": clean[i].detach().cpu(),
                        "recon": recon[i].detach().cpu(),
                        "pred": {
                            name: float(pred_phys[i, j])
                            for j, name in enumerate(eval_params)
                        },
                        "true": {
                            name: float(true_phys[i, j])
                            for j, name in enumerate(eval_params)
                        },
                        "errpct": {
                            name: float(err_pct[i, j])
                            for j, name in enumerate(eval_params)
                        },
                    }
                )
        if len(show_examples) >= n_show:
            break

    rows = []
    for name in eval_params:
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
        rows.append(
            {"param": name, "mean_%": mean, "median_%": median, "p90_%": p90, "p95_%": p95}
        )

    df = pd.DataFrame(rows).set_index("param").sort_index()
    print("\n=== Erreurs en % (paramètres prédits uniquement) ===")
    print(df.round(4))

    if len(exp_params_set) > 0:
        print(
            f"\n⚠️  Paramètres expérimentaux (GT utilisé, non évalués): {sorted(exp_params_set)}"
        )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = tag or "eval"
        csv_path = save_dir / f"{tag}_metrics.csv"
        df.round(6).to_csv(csv_path)
        print(f"✓ Metrics CSV: {csv_path}")

    if len(show_examples) > 0:
        rows_to_plot = len(show_examples)
        fig, axes = plt.subplots(rows_to_plot, 2, figsize=(12, 2.8 * rows_to_plot), sharex=False)
        if rows_to_plot == 1:
            axes = np.array([axes])
        for r, example in enumerate(show_examples):
            x = np.arange(example["clean"].numel())
            ax1, ax2 = axes[r, 0], axes[r, 1]
            ax1.plot(x, example["noisy"], lw=0.9, alpha=0.7, label="Noisy")
            ax1.plot(x, example["clean"], lw=1.2, label="Clean")
            ax1.plot(x, example["recon"], lw=1.0, ls="--", label="Recon")
            ax1.set_ylabel("Transmission")
            err_txt = ", ".join(
                [f"{k}: {example['errpct'][k]:.2f}%" for k in eval_params]
            )
            ax1.set_title(f"Exemple {r + 1} — erreurs % : {err_txt}", fontsize=10)
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

        if save_dir is not None:
            tag = tag or "eval"
            png_path = Path(save_dir) / f"{tag}_examples.png"
            save_fig(fig, png_path, dpi=160)
            print(f"✓ Examples PNG: {png_path}")
        else:
            plt.show()

    return df
