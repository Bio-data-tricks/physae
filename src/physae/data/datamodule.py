"""Lightning-like datamodule for PhysAE."""
from __future__ import annotations

from typing import Dict, Iterator

import torch
from torch.utils.data import DataLoader

from physae.config import DataConfig

from .dataset import SpectraDataset


class PhysaeDataModule:
    """Simple data module exposing train/val/test loaders."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._train: SpectraDataset | None = None
        self._val: SpectraDataset | None = None
        self._test: SpectraDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", "train", None):
            self._train = SpectraDataset(self.config, stage="train")
        if stage in ("fit", "validate", None):
            self._val = SpectraDataset(self.config, stage="val")
        if stage in ("test", None):
            self._test = SpectraDataset(self.config, stage="test")

    def train_dataloader(self) -> DataLoader[Dict[str, torch.Tensor]]:
        if self._train is None:
            raise RuntimeError("setup() must be called before requesting dataloaders")
        return DataLoader(
            self._train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Dict[str, torch.Tensor]]:
        if self._val is None:
            raise RuntimeError("setup() must be called before requesting dataloaders")
        return DataLoader(
            self._val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Dict[str, torch.Tensor]]:
        if self._test is None:
            raise RuntimeError("setup() must be called before requesting dataloaders")
        return DataLoader(
            self._test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self._test is None:
            self.setup("test")
        assert self._test is not None
        yield from (
            {
                key: value.to(torch.float32)
                for key, value in batch.items()
            }
            for batch in self.test_dataloader()
        )
