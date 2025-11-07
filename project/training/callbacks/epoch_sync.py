"""
Callbacks for epoch synchronization.
"""
import pytorch_lightning as pl


class UpdateEpochInDataset(pl.Callback):
    """Callback to update epoch counter in dataset."""

    def on_train_epoch_start(self, trainer, pl_module):
        """Update epoch in dataset at the start of each training epoch."""
        if hasattr(trainer.train_dataloader.dataset, 'set_epoch'):
            trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch)
