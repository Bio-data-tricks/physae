"""Boucle d'entraînement basique pour le modèle PhysAE.

Ce module suppose que toutes les définitions de modèle et de données se
trouvent dans :mod:`model_data` et fournit un point d'entrée minimaliste pour
entraîner un modèle sur un fichier CSV.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from model_data import DataConfig, SpectraDataset, SpectraRegressor


@dataclass(slots=True)
class TrainingConfig:
    data: DataConfig
    batch_size: int = 64
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.2
    hidden_layers: Iterable[int] = (256, 128, 64)
    dropout: float = 0.1

    def to_dict(self) -> dict:
        out = asdict(self)
        out["data"] = asdict(self.data)
        return out


def _build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SpectraDataset(config.data)
    total_len = len(dataset)
    if total_len < 2:
        raise ValueError("Le dataset doit contenir au moins 2 échantillons pour un split train/val")
    val_len = int(total_len * config.validation_split)
    val_len = max(1, val_len)
    train_len = max(total_len - val_len, 1)
    if train_len + val_len > total_len:
        val_len = total_len - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    return train_loader, val_loader


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    return running_loss / len(dataloader.dataset)


def _evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss += criterion(predictions, targets).item() * features.size(0)
    return loss / len(dataloader.dataset)


def train(config: TrainingConfig, *, device: str | torch.device = "cpu") -> dict[str, float]:
    """Entraîne un modèle et renvoie les métriques finales."""

    device = torch.device(device)
    train_loader, val_loader = _build_dataloaders(config)

    model = SpectraRegressor(
        input_dim=len(config.data.feature_columns),
        hidden_layers=config.hidden_layers,
        output_dim=len(config.data.target_columns),
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        best_val = min(best_val, val_loss)
        tqdm.write(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    return {"best_val": best_val, "history": history, "state_dict": model.state_dict()}


def main(config: TrainingConfig) -> None:
    metrics = train(config)
    print("Meilleure perte de validation:", metrics["best_val"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement basique du modèle PhysAE")
    parser.add_argument("csv", type=Path, help="Chemin du fichier CSV")
    parser.add_argument("--features", nargs="+", required=True, help="Colonnes caractéristiques")
    parser.add_argument("--targets", nargs="+", required=True, help="Colonnes cibles")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    data_cfg = DataConfig(csv_path=args.csv, feature_columns=args.features, target_columns=args.targets)
    train_cfg = TrainingConfig(
        data=data_cfg,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )
    main(train_cfg)

