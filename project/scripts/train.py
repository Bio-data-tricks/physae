"""
Main training script.

This is a template for the training script. Adapt it to your specific needs based on
the original training setup in physae.py.
"""
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.params import NORM_PARAMS
from data.dataset import SpectraDataset
from models.autoencoder import PhysicallyInformedAE
from physics.tips import Tips2021QTpy, find_qtpy_dir
from training.callbacks.epoch_sync import UpdateEpochInDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PhysicallyInformedAE model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--qtpy_dir', type=str, default='./QTpy', help='QTpy directory')
    return parser.parse_args()


def create_datasets(args):
    """
    Create train and validation datasets.

    Args:
        args: Command line arguments.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # Initialize TIPS partition functions
    qtpy_dir = find_qtpy_dir(args.qtpy_dir)
    tipspy = Tips2021QTpy(qtpy_dir, device='cpu')

    # TODO: Load transitions data
    # This should be loaded from your specific data files
    transitions_dict = {
        # 'CH4': transitions_CH4,
        # 'H2O': transitions_H2O,
    }

    # TODO: Define polynomial frequency coefficients. Provide the exact values
    # from your spectrometer calibration; pass ``None`` to rely on a linear grid
    # when no correction is available.
    poly_freq_CH4 = None

    # Create datasets
    train_dataset = SpectraDataset(
        n_samples=10000,
        num_points=1024,  # Adjust to your data
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=NORM_PARAMS,
        with_noise=True,
        tipspy=tipspy,
    )

    val_dataset = SpectraDataset(
        n_samples=1000,
        num_points=1024,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=NORM_PARAMS,
        with_noise=True,
        freeze_noise=True,  # Fixed noise for validation
        tipspy=tipspy,
    )

    return train_dataset, val_dataset


def main():
    """Main training function."""
    args = parse_args()

    # Set seed for reproducibility
    pl.seed_everything(42)

    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(args)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # TODO: Load transitions data
    transitions_dict = {}
    poly_freq_CH4 = None

    # Create model
    print("Creating model...")
    model = PhysicallyInformedAE(
        n_points=1024,
        param_names=['sig0', 'dsig', 'mf_CH4', 'baseline0', 'baseline1', 'baseline2', 'P', 'T'],
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        lr=args.lr,
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='physae-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
        ),
        UpdateEpochInDataset(),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Training complete!")


if __name__ == '__main__':
    main()
