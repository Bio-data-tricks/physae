"""Utilities that keep dataset and sampler epochs in sync with Lightning."""

from __future__ import annotations

from collections.abc import Iterable

import pytorch_lightning as pl

__all__ = [
    "UpdateEpochInDataset",
    "UpdateEpochInValDataset",
    "AdvanceDistributedSamplerEpochAll",
]


class UpdateEpochInDataset(pl.Callback):
    """Propagate the trainer epoch to the training dataset if supported."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning signature
        if hasattr(trainer, "train_dataloader"):
            dataloaders = trainer.train_dataloader
            if callable(dataloaders):
                dataloaders = dataloaders()
        elif hasattr(trainer, "train_dataloaders"):
            dataloaders = trainer.train_dataloaders
        else:
            return

        if isinstance(dataloaders, (list, tuple)):
            dataloaders = dataloaders[0] if len(dataloaders) > 0 else None

        if dataloaders is None:
            return

        dataset = getattr(dataloaders, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(trainer.current_epoch)
            if getattr(trainer, "is_global_zero", True):
                print(f"✓ Train dataset epoch mis à jour: {trainer.current_epoch}")


class UpdateEpochInValDataset(pl.Callback):
    """Propagate the trainer epoch to every validation dataset."""

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning signature
        if hasattr(trainer, "val_dataloaders"):
            dataloaders = trainer.val_dataloaders
        elif hasattr(trainer, "val_dataloader"):
            dataloaders = trainer.val_dataloader
            if callable(dataloaders):
                dataloaders = dataloaders()
        else:
            return

        if not isinstance(dataloaders, Iterable):
            dataloaders = [dataloaders] if dataloaders is not None else []

        for dl in dataloaders:
            if dl is None:
                continue
            dataset = getattr(dl, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(trainer.current_epoch)
                if getattr(trainer, "is_global_zero", True):
                    print(f"✓ Val dataset epoch mis à jour: {trainer.current_epoch}")


class AdvanceDistributedSamplerEpochAll(pl.Callback):
    """Advance the epoch counter of distributed samplers for train/val."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning signature
        dataloader = getattr(trainer, "train_dataloader", None)
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401 - Lightning signature
        dataloaders = getattr(trainer, "val_dataloaders", None)
        if not dataloaders:
            return
        if not isinstance(dataloaders, (list, tuple)):
            dataloaders = [dataloaders]

        for dl in dataloaders:
            sampler = getattr(dl, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(trainer.current_epoch)
