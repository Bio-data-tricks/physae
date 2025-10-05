"""Module dédié aux composants *modèle* et *données* pour le projet PhysAE.

Ce module se concentre uniquement sur la définition :

* d'une configuration de données (`DataConfig`)
* d'un `Dataset` PyTorch prêt à l'emploi
* d'un modèle de régression simple basé sur un MLP

Tout le code lié à l'entraînement ou à l'optimisation est volontairement
laissé dans d'autres modules afin de respecter une séparation claire des
responsabilités.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(slots=True)
class DataConfig:
    """Configuration décrivant les chemins et colonnes à charger.

    Attributes
    ----------
    csv_path: str | Path
        Chemin du fichier CSV contenant les échantillons.
    feature_columns: Sequence[str]
        Noms des colonnes à utiliser comme caractéristiques.
    target_columns: Sequence[str]
        Noms des colonnes à utiliser comme cibles.
    """

    csv_path: str | Path
    feature_columns: Sequence[str]
    target_columns: Sequence[str]

    def validate(self) -> None:
        """Vérifie la cohérence basique de la configuration."""

        if not self.feature_columns:
            raise ValueError("feature_columns ne peut pas être vide")
        if not self.target_columns:
            raise ValueError("target_columns ne peut pas être vide")
        if Path(self.csv_path).suffix.lower() != ".csv":
            raise ValueError("csv_path doit pointer vers un fichier .csv")


class SpectraDataset(Dataset):
    """Dataset tabulaire léger pour entraîner un modèle de régression.

    Le dataset charge toutes les données en mémoire sous forme de tenseurs
    `torch.float32` pour simplifier les boucles d'entraînement. Pour des jeux
    de données massifs il conviendrait d'adapter cette implémentation, mais
    cette version couvre la majorité des cas d'usage dans le projet.
    """

    def __init__(self, config: DataConfig, *, cache: bool = True) -> None:
        config.validate()
        self.config = config
        self.cache = cache

        if cache:
            self._features, self._targets = self._load_into_memory()
        else:
            self._features = self._targets = None

    def _load_into_memory(self) -> tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(self.config.csv_path)
        missing_features = set(self.config.feature_columns) - set(df.columns)
        missing_targets = set(self.config.target_columns) - set(df.columns)
        if missing_features:
            raise KeyError(f"Colonnes caractéristiques manquantes: {missing_features}")
        if missing_targets:
            raise KeyError(f"Colonnes cibles manquantes: {missing_targets}")

        features = torch.tensor(df[self.config.feature_columns].values, dtype=torch.float32)
        targets = torch.tensor(df[self.config.target_columns].values, dtype=torch.float32)
        return features, targets

    def __len__(self) -> int:  # type: ignore[override]
        if self.cache and self._features is not None:
            return self._features.shape[0]

        df = pd.read_csv(self.config.csv_path, usecols=self.config.feature_columns[:1])
        return len(df.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if self.cache and self._features is not None and self._targets is not None:
            return self._features[index], self._targets[index]

        df = pd.read_csv(
            self.config.csv_path,
            usecols=list(self.config.feature_columns) + list(self.config.target_columns),
        )
        feature_values = torch.tensor(df.loc[index, self.config.feature_columns], dtype=torch.float32)
        target_values = torch.tensor(df.loc[index, self.config.target_columns], dtype=torch.float32)
        return feature_values, target_values


class SpectraRegressor(nn.Module):
    """Réseau de neurones multi-couches entièrement connecté."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int] = (256, 128, 64),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


def count_parameters(model: nn.Module) -> int:
    """Renvoie le nombre de paramètres entraînables d'un modèle."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "DataConfig",
    "SpectraDataset",
    "SpectraRegressor",
    "count_parameters",
]

