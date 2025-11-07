"""
Example training script for Stage A: Backbone training.

This script demonstrates how to train the PhysicallyInformedAE model in Stage A,
which trains the encoder backbone and initial parameter prediction head while
keeping refiners disabled.

Dataset Configuration:
- Synthetic spectra generated from physics simulation
- Parameter ranges chosen for typical atmospheric CH4/H2O measurements
- Noise augmentation enabled for robustness

Model Configuration:
- Stage A: Train backbone encoder only
- EfficientNet-S variant for balance of speed and performance
- 3 cascade refiners (disabled in Stage A, will be trained in Stage B)
- ReLoBRaLo adaptive loss balancing
"""

import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader
import numpy as np
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.params import PARAMS, LOG_SCALE_PARAMS, NORM_PARAMS
from config.data_config import load_parameter_ranges, load_noise_profile, load_transitions
from data.dataset import SpectraDataset
from models.autoencoder import PhysicallyInformedAE
from physics.tips import Tips2021QTpy, find_qtpy_dir
from training.callbacks import StageAwarePlotCallback, UpdateEpochInDataset


def setup_parameter_ranges(parameters_config: str | None):
    """Load parameter ranges from YAML and update global normalisation.

    Args:
        parameters_config: Optional custom YAML path provided via CLI. When
            ``None``, the bundled ``parameters_default.yaml`` file is used.
    """

    default_path = Path(__file__).parent.parent / "config" / "data" / "parameters_default.yaml"
    config_path = Path(parameters_config) if parameters_config else default_path

    print(f"Loading parameter ranges from: {config_path}")
    try:
        ranges = load_parameter_ranges(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Failed to load parameter configuration '{config_path}': {exc}") from exc

    print("Parameter ranges configured:")
    for param, (min_val, max_val) in ranges.items():
        log_note = " (log scale)" if param in LOG_SCALE_PARAMS else ""
        print(f"  {param:12s}: [{min_val:12.6e}, {max_val:12.6e}]{log_note}")

    return ranges


def load_transitions_data(transitions_config: str | None):
    """Load spectroscopic transitions and optional frequency polynomials.

    Args:
        transitions_config: Optional custom YAML path. When ``None``, a small
            synthetic example bundled with the repository is used.

    Returns:
        Tuple ``(transitions, poly_freq_map)`` where ``poly_freq_map`` contains
        per-molecule polynomial coefficients (if any) declared in the YAML
        ``poly_frequency`` section.
    """

    default_path = Path(__file__).parent.parent / "config" / "data" / "transitions_sample.yaml"
    config_path = Path(transitions_config) if transitions_config else default_path

    try:
        transitions_dict, poly_freq_map = load_transitions(
            config_path, include_poly_freq=True
        )
    except FileNotFoundError:
        if transitions_config:
            raise
        print("\nWARNING: No transitions configuration found; using empty dictionary.")
        return {}, {}
    except ValueError as exc:
        raise RuntimeError(f"Invalid transitions configuration '{config_path}': {exc}") from exc

    if not transitions_dict:
        print("\nWARNING: Loaded transitions configuration is empty.")
        print("Provide real HITRAN data for physics-accurate training.")
    else:
        print("Transitions loaded:")
        for mol, entries in transitions_dict.items():
            print(f"  {mol}: {len(entries)} lines")

    if poly_freq_map:
        print("Polynomial frequency coefficients available for:")
        for mol, coeffs in poly_freq_map.items():
            print(f"  {mol}: degree {len(coeffs) - 1}")

    return transitions_dict, poly_freq_map


def load_noise_profile_config(noise_config: str | None):
    """Load the noise profile configuration used for data augmentation."""

    default_path = Path(__file__).parent.parent / "config" / "data" / "noise_default.yaml"
    config_path = Path(noise_config) if noise_config else default_path

    try:
        noise_profile = load_noise_profile(config_path)
    except FileNotFoundError:
        if noise_config:
            raise
        print("\nWARNING: No noise configuration found; disabling noise augmentations.")
        return {}
    except ValueError as exc:
        raise RuntimeError(f"Invalid noise configuration '{config_path}': {exc}") from exc

    print(f"Noise profile loaded from: {config_path}")
    return noise_profile


def create_datasets(args, poly_freq_CH4, transitions_dict, noise_profile, tipspy):
    """
    Create training and validation datasets.

    Args:
        args: Command line arguments.
        poly_freq_CH4: Polynomial frequency grid coefficients. ``None`` falls
            back to a linear grid defined by ``sig0`` and ``dsig``.
        transitions_dict: Dictionary of transition data.
        noise_profile: Noise augmentation configuration dictionary.
        tipspy: TIPS partition function calculator.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    print(f"\nCreating datasets:")
    print(f"  Training samples: {args.train_samples}")
    print(f"  Validation samples: {args.val_samples}")
    print(f"  Spectral points: {args.num_points}")
    print(
        "  Training dataset mode: "
        + ("statique" if args.static_training_set else "inline (résample chaque epoch)")
    )
    print(
        "  Validation dataset mode: "
        + ("statique" if args.static_validation_set else "inline (résample chaque epoch)")
    )

    train_dataset = SpectraDataset(
        n_samples=args.train_samples,
        num_points=args.num_points,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=NORM_PARAMS,
        with_noise=True,
        noise_profile=noise_profile,
        freeze_parameters=args.static_training_set,
        freeze_noise=False,  # Random noise each epoch
        tipspy=tipspy,
    )

    val_dataset = SpectraDataset(
        n_samples=args.val_samples,
        num_points=args.num_points,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=NORM_PARAMS,
        with_noise=True,
        noise_profile=noise_profile,
        freeze_parameters=args.static_validation_set,
        freeze_noise=True,  # Fixed noise for consistent validation
        tipspy=tipspy,
    )

    return train_dataset, val_dataset


def create_model(args, poly_freq_CH4, transitions_dict):
    """
    Create PhysicallyInformedAE model configured for Stage A training.

    Stage A Configuration:
    - Training stage: 'A' (backbone only)
    - Encoder: EfficientNet-S (balanced performance)
    - Number of refiners: 3 (will be trained in Stage B)
    - Optimizer: AdamW with cosine annealing
    - Loss: ReLoBRaLo adaptive balancing of MSE, Spectral Angle, Peak-Weighted MSE

    Args:
        args: Command line arguments.
        poly_freq_CH4: Polynomial frequency grid coefficients. ``None`` uses a
            purely linear grid during physics reconstruction.
        transitions_dict: Dictionary of transition data.

    Returns:
        PhysicallyInformedAE model instance.
    """
    print(f"\nCreating model for Stage A:")
    print(f"  Encoder variant: {args.encoder_variant}")
    print(f"  Number of refiners: {args.num_refiners}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Dropout: {args.mlp_dropout}")

    model = PhysicallyInformedAE(
        # Data configuration
        n_points=args.num_points,
        param_names=PARAMS,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,

        # Model architecture
        encoder_variant=args.encoder_variant,  # 's', 'm', or 'l'
        num_cascade_refiners=args.num_refiners,
        mlp_dropout=args.mlp_dropout,
        refiner_dropout=args.refiner_dropout,

        # Training configuration
        training_stage='A',  # STAGE A: Train backbone only
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,

        # Loss configuration
        use_relobralo=True,  # Adaptive loss balancing
        relobralo_alpha=0.95,  # EMA decay for loss history
        relobralo_beta=0.99,   # Relative weight tuning

        # Individual loss weights (will be balanced by ReLoBRaLo)
        lambda_params_mse=1.0,
        lambda_spectral_angle=0.1,
        lambda_peak_weighted_mse=0.5,

        # Scheduler configuration
        lr_scheduler='cosine_warmup',
        warmup_epochs=5,
        t_0=args.epochs // 2,  # First restart at half training
        t_mult=2,              # Double period after each restart
        eta_min_ratio=0.01,    # Minimum LR = 1% of initial

        # Physical simulation settings
        use_linemixing=True,   # Include line mixing corrections
        device='cuda' if args.gpus > 0 else 'cpu',
    )

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PhysicallyInformedAE - Stage A (Backbone)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument('--train_samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000,
                        help='Number of validation samples')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of spectral points')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--static_training_set', action='store_true',
                        help="Pré-génère les paramètres une fois pour l'entraînement (dataset fixe)")
    parser.add_argument('--static_validation_set', action='store_true',
                        help="Pré-génère les paramètres une fois pour la validation")

    # Model configuration
    parser.add_argument('--encoder_variant', type=str, default='s',
                        choices=['s', 'm', 'l'],
                        help='EfficientNet encoder variant (s=small, m=medium, l=large)')
    parser.add_argument('--num_refiners', type=int, default=3,
                        help='Number of cascade refiners')
    parser.add_argument('--mlp_dropout', type=float, default=0.10,
                        help='Dropout rate in MLP heads')
    parser.add_argument('--refiner_dropout', type=float, default=0.05,
                        help='Dropout rate in refiners')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'lion'],
                        help='Optimizer (AdamW or Lion)')

    # Hardware configuration
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs (0 for CPU)')
    parser.add_argument('--precision', type=str, default='32',
                        choices=['16', '32', 'bf16'],
                        help='Training precision')

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/stage_a',
                        help='Checkpoint directory')
    parser.add_argument('--qtpy_dir', type=str, default='./QTpy',
                        help='TIPS QTpy directory')
    parser.add_argument('--parameters_config', type=str, default=None,
                        help='Chemin vers le YAML des plages de paramètres')
    parser.add_argument('--noise_config', type=str, default=None,
                        help='Chemin vers le YAML du profil de bruit')
    parser.add_argument('--transitions_config', type=str, default=None,
                        help='Chemin vers le YAML des transitions spectroscopiques')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs/stage_a',
                        help='TensorBoard log directory')

    return parser.parse_args()


def main():
    """Main training function for Stage A."""
    args = parse_args()

    print("=" * 80)
    print("PhysicallyInformedAE - Stage A Training")
    print("=" * 80)

    # Set random seed
    pl.seed_everything(args.seed)
    print(f"\nRandom seed: {args.seed}")

    # Setup parameter ranges
    setup_parameter_ranges(args.parameters_config)

    # Load noise profile
    noise_profile = load_noise_profile_config(args.noise_config)

    # Load transitions data (or use placeholder)
    transitions_dict, poly_freq_map = load_transitions_data(args.transitions_config)
    poly_freq_CH4 = poly_freq_map.get("CH4")
    if poly_freq_CH4 is None:
        print("\nNo polynomial frequency provided for CH4 — defaulting to linear grid.")

    # Initialize TIPS partition functions
    print(f"\nInitializing TIPS partition functions...")
    try:
        qtpy_dir = find_qtpy_dir(args.qtpy_dir)
        tipspy = Tips2021QTpy(qtpy_dir, device='cpu')
        print(f"  TIPS loaded from: {qtpy_dir}")
    except Exception as e:
        print(f"  WARNING: Could not load TIPS: {e}")
        print(f"  Continuing with None (physics simulation will be disabled)")
        tipspy = None

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        args, poly_freq_CH4, transitions_dict, noise_profile, tipspy
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    model = create_model(args, poly_freq_CH4, transitions_dict)

    # Setup callbacks
    stage_fig_dir = Path(args.log_dir) / "figures"

    callbacks = [
        # Stage-aware visual diagnostics mirroring physae.py
        StageAwarePlotCallback(
            val_loader=val_loader,
            param_names=model.param_names,
            num_examples=1,
            save_dir=stage_fig_dir,
            stage_tag="StageA",
            refine=False,
            use_gt_for_provided=True,
            recon_PT="pred",
            max_val_batches=10,
        ),

        # Model checkpointing
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='physae-stage-a-{epoch:03d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            verbose=True,
        ),

        # Learning rate monitoring
        LearningRateMonitor(logging_interval='epoch'),

        # Update epoch in dataset for curriculum learning
        UpdateEpochInDataset(),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=50,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        default_root_dir=args.log_dir,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Print training summary
    print(f"\n{'=' * 80}")
    print("Training Configuration Summary")
    print(f"{'=' * 80}")
    print(f"Stage: A (Backbone training)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Device: {'GPU' if args.gpus > 0 else 'CPU'}")
    print(f"Precision: {args.precision}")
    print(f"{'=' * 80}\n")

    # Train
    print("Starting training...\n")
    trainer.fit(model, train_loader, val_loader)

    # Print completion message
    print(f"\n{'=' * 80}")
    print("Stage A Training Complete!")
    print(f"{'=' * 80}")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"All checkpoints saved to: {args.checkpoint_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review training logs in TensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"  2. Evaluate model performance on test set")
    print(f"  3. Proceed to Stage B1 training (refiners with frozen backbone)")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
