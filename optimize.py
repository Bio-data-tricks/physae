"""Module dédié à l'optimisation d'hyperparamètres.

Le module s'appuie sur la boucle d'entraînement basique définie dans
``train_basic.py`` et effectue une simple recherche en grille.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import torch

from model_data import DataConfig
from train_basic import TrainingConfig, train


@dataclass(slots=True)
class GridSearchSpace:
    learning_rates: Sequence[float]
    batch_sizes: Sequence[int]
    hidden_layer_sets: Sequence[Iterable[int]]
    dropouts: Sequence[float]


def grid_search(
    data_cfg: DataConfig,
    search_space: GridSearchSpace,
    *,
    max_epochs: int = 50,
    weight_decay: float = 1e-4,
    validation_split: float = 0.2,
    device: str | torch.device = "cpu",
) -> list[tuple[dict, float]]:
    """Retourne les couples (config, val_loss) triés par performance."""

    results: list[tuple[dict, float]] = []
    for lr, batch, hidden, dropout in product(
        search_space.learning_rates,
        search_space.batch_sizes,
        search_space.hidden_layer_sets,
        search_space.dropouts,
    ):
        cfg = TrainingConfig(
            data=data_cfg,
            batch_size=batch,
            max_epochs=max_epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            validation_split=validation_split,
            hidden_layers=hidden,
            dropout=dropout,
        )
        metrics = train(cfg, device=device)
        results.append((cfg.to_dict(), metrics["best_val"]))

    results.sort(key=lambda item: item[1])
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Recherche en grille sur les hyperparamètres")
    parser.add_argument("csv", type=Path, help="Chemin du fichier CSV")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", nargs="+", type=float, default=[1e-3, 5e-4])
    parser.add_argument("--batch", nargs="+", type=int, default=[32, 64])
    parser.add_argument(
        "--hidden",
        nargs="+",
        default=["256,128,64", "512,256,128"],
        help="Listes de tailles séparées par des virgules",
    )
    parser.add_argument("--dropout", nargs="+", type=float, default=[0.1, 0.2])

    args = parser.parse_args()

    hidden_sets = [tuple(int(x) for x in item.split(",")) for item in args.hidden]
    data_cfg = DataConfig(csv_path=args.csv, feature_columns=args.features, target_columns=args.targets)
    space = GridSearchSpace(
        learning_rates=args.lr,
        batch_sizes=args.batch,
        hidden_layer_sets=hidden_sets,
        dropouts=args.dropout,
    )

    results = grid_search(data_cfg, space, max_epochs=args.epochs)
    for cfg, score in results:
        print(f"val_loss={score:.4f} | config={cfg}")


if __name__ == "__main__":
    main()

