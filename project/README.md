# Physae - Physically Informed Autoencoder

This is a refactored version of the physae project with a clean modular structure.

## Project Structure

```
project/
├── config/              # Configuration parameters
│   ├── params.py        # Global parameters (PARAMS, LOG_SCALE_PARAMS, NORM_PARAMS)
│   └── training_config.py  # Training configurations
├── data/                # Data handling
│   ├── normalization.py # Parameter normalization functions
│   ├── noise.py         # Noise augmentation (add_noise_variety)
│   ├── dataset.py       # SpectraDataset
│   └── loaders.py       # DataLoader creation utilities
├── physics/             # Physics simulation
│   ├── constants.py     # Physical constants (CGS units)
│   ├── tips.py          # HITRAN TIPS_2021 (Tips2021QTpy)
│   ├── transitions.py   # Transition data parsing
│   ├── profiles.py      # Spectral profiles (wofz, Pine, line mixing)
│   └── forward.py       # Forward spectral model
├── models/              # Neural network models
│   ├── blocks/          # Building blocks
│   │   ├── convolutions.py  # ConvBNAct1d, BlurPool1D, etc.
│   │   ├── mbconv.py    # MBConv1d, FusedMBConv1d
│   │   └── attention.py # SE1d, ChannelAttention1d, etc.
│   ├── backbone.py      # EfficientNetEncoder
│   ├── refiner.py       # EfficientNetRefiner
│   ├── denoiser.py      # Denoiser1D
│   └── autoencoder.py   # PhysicallyInformedAE (Lightning Module)
├── losses/              # Loss functions
│   ├── spectral.py      # SpectralAngleLoss
│   ├── peak_weighted.py # PeakWeightedMSELoss
│   └── relobralo.py     # ReLoBRaLoLoss
├── training/            # Training utilities
│   ├── stages.py        # Training stages (A, B, C, etc.)
│   ├── scheduler.py     # Custom LR schedulers
│   └── callbacks/       # Training callbacks
│       ├── visualization.py
│       ├── epoch_sync.py
│       └── loss_curves.py
├── evaluation/          # Evaluation utilities
│   ├── metrics.py       # Metric calculation
│   └── inference.py     # Inference functions
├── utils/               # Utility functions
│   ├── lowess.py        # LOWESS algorithm
│   ├── plotting.py      # Plotting utilities
│   ├── distributed.py   # Distributed training utilities
│   └── io.py            # I/O utilities
└── scripts/             # Executable scripts
    ├── train.py         # Main training script
    └── evaluate.py      # Evaluation script
```

## Installation

```bash
# Install dependencies
pip install torch pytorch-lightning numpy pandas matplotlib scipy
pip install lion-pytorch  # Lion optimizer
```

## Usage

### Training

```bash
python project/scripts/train.py --batch_size 32 --epochs 100 --lr 1e-4 --gpus 1
```

### Evaluation

```bash
python project/scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt
```

## Notes

- The `autoencoder.py` file contains a placeholder implementation. The full implementation
  from the original `physae.py` (starting at line 1670) needs to be extracted and adapted.
- Some training stage functions in `training/stages.py` are placeholders and need to be
  implemented based on your specific training protocol.
- The physics module requires HITRAN TIPS_2021 data files (QTpy format) for partition functions.

## Migration from Original Code

This refactored version separates concerns into logical modules:

1. **Physics** is isolated in the `physics/` module
2. **Neural networks** are in `models/` with clear separation of concerns
3. **Data handling** is centralized in `data/`
4. **Training logic** is in `training/` with PyTorch Lightning integration
5. **Loss functions** are modular in `losses/`

To complete the migration, you may need to:

1. Extract the full `PhysicallyInformedAE` implementation from the original `physae.py`
2. Adapt the training scripts to your specific data and configuration
3. Implement any missing training stages or callbacks
4. Add visualization and logging as needed
