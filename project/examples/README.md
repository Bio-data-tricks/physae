# Training Examples for PhysicallyInformedAE

This directory contains example scripts and configuration guides for training the PhysicallyInformedAE model.

## Quick Start

### Stage A: Train Backbone Encoder

```bash
# Navigate to project root
cd /path/to/physae

# Run Stage A training with default configuration
python project/examples/train_stage_a.py \
    --train_samples 10000 \
    --val_samples 1000 \
    --epochs 100 \
    --batch_size 64 \
    --encoder_variant s \
    --gpus 1
```

**What is Stage A?**
Stage A trains the encoder backbone and initial parameter prediction head. The cascade refiners are present in the architecture but remain inactive during this stage.

### Quick Test Run (5 minutes)

```bash
python project/examples/train_stage_a.py \
    --train_samples 1000 \
    --val_samples 100 \
    --epochs 10 \
    --batch_size 32 \
    --gpus 1
```

## Files in This Directory

### `train_stage_a.py`
Complete training script for Stage A with:
- Synthetic dataset generation
- Model configuration
- Training loop with PyTorch Lightning
- Checkpointing and early stopping
- Comprehensive command-line interface

**Key Features:**
- ✅ Ready to run (no external data required for synthetic training)
- ✅ Configurable via command-line arguments
- ✅ Supports GPU and CPU training
- ✅ Automatic mixed precision
- ✅ TensorBoard logging
- ✅ Progress bars and training summaries

### `CONFIGURATION_GUIDE.md`
Comprehensive guide explaining:
- All hyperparameters and their effects
- Dataset parameter ranges and their physical meanings
- Model architecture choices
- Training configurations for different use cases
- Alternative configurations (atmospheric, industrial, laboratory)
- Troubleshooting guide

**Topics Covered:**
- Parameter ranges (NORM_PARAMS)
- Model architecture (encoder variants, refiners, dropout)
- Training hyperparameters (learning rate, optimizer, batch size)
- Loss configuration (ReLoBRaLo, multi-objective)
- Hardware configuration (GPUs, precision, workers)
- Data requirements (HITRAN, TIPS)
- Troubleshooting (OOM, instability, slow convergence)

## Training Stages

PhysicallyInformedAE uses a multi-stage training protocol:

| Stage | What is Trained | Learning Rate | Typical Epochs |
|-------|-----------------|---------------|----------------|
| **A** | Backbone encoder + initial head | 1e-4 | 100 |
| B1 | Refiners (backbone frozen) | 1e-5 | 30-50 |
| B2 | Full model fine-tuning | 1e-6 | 20-30 |
| DEN | Denoiser only | 1e-4 | 50 |

**This directory currently provides:**

- Stage A training script (`train_stage_a.py`)
- A YAML-driven Stage A walkthrough notebook (`notebooks/stage_a_from_yaml.ipynb`) showing how to load the bundled configuration files and kick off a short training run.

**Coming soon:** DEN example notebook.

## Requirements

### Python Dependencies

```bash
pip install torch pytorch-lightning numpy pandas matplotlib scipy
pip install lion-pytorch  # Optional: for Lion optimizer
```

### Data Requirements (Optional)

For physics-based simulation with real molecular data:

1. **HITRAN Transition Data:**
   - Download from: https://hitran.org/lbl/
   - Required molecules: CH₄ (06), H₂O (01)
   - Save to: `./data/hitran/`

2. **TIPS Partition Functions:**
   - Download from: https://hitran.org/docs/iso-meta/
   - Save QTpy files to: `./QTpy/`

**Note:** The example script works without external data (uses placeholder physics).

## Command-Line Arguments

### Data Configuration

```bash
--train_samples 10000      # Number of training samples
--val_samples 1000         # Number of validation samples
--num_points 1024          # Spectral resolution (pixels)
--batch_size 64            # Batch size
--num_workers 4            # Data loading workers
```

### Model Configuration

```bash
--encoder_variant s        # Encoder size: 's' (small), 'm' (medium), 'l' (large)
--num_refiners 3           # Number of cascade refiners
--mlp_dropout 0.10         # Dropout in MLP heads
--refiner_dropout 0.05     # Dropout in refiners
```

### Training Configuration

```bash
--epochs 100               # Training epochs
--lr 1e-4                  # Initial learning rate
--weight_decay 0.01        # L2 regularization
--optimizer adamw          # Optimizer: 'adamw' or 'lion'
```

### Hardware Configuration

```bash
--gpus 1                   # Number of GPUs (0 for CPU)
--precision 32             # Precision: '32', '16', 'bf16'
```

### Paths

```bash
--checkpoint_dir ./checkpoints/stage_a    # Checkpoint save directory
--log_dir ./logs/stage_a                  # TensorBoard log directory
--qtpy_dir ./QTpy                          # TIPS data directory
--transitions_dir ./data/hitran            # HITRAN data directory
```

## Example Configurations

### Fast Prototyping
```bash
python project/examples/train_stage_a.py \
    --train_samples 5000 \
    --val_samples 500 \
    --epochs 50 \
    --batch_size 32 \
    --encoder_variant s \
    --lr 1e-4
```

### High Performance
```bash
python project/examples/train_stage_a.py \
    --train_samples 50000 \
    --val_samples 5000 \
    --epochs 150 \
    --batch_size 128 \
    --encoder_variant l \
    --lr 1e-4 \
    --precision bf16 \
    --num_workers 8
```

### Multi-GPU
```bash
python project/examples/train_stage_a.py \
    --train_samples 50000 \
    --val_samples 5000 \
    --epochs 100 \
    --batch_size 256 \
    --encoder_variant m \
    --gpus 4 \
    --precision bf16
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./logs/stage_a

# Open in browser
# http://localhost:6006
```

**Available metrics:**
- Training loss (total, per-component)
- Validation loss
- Learning rate schedule
- Parameter prediction errors
- Spectral reconstruction quality

### Checkpoints

Checkpoints are saved to `--checkpoint_dir` (default: `./checkpoints/stage_a`):

```
checkpoints/stage_a/
├── physae-stage-a-epoch=050-val_loss=0.0123.ckpt  # Top-3 best models
├── physae-stage-a-epoch=075-val_loss=0.0089.ckpt
├── physae-stage-a-epoch=099-val_loss=0.0067.ckpt
└── last.ckpt                                       # Most recent checkpoint
```

## After Training

### Load Trained Model

```python
from models.autoencoder import PhysicallyInformedAE

# Load from checkpoint
model = PhysicallyInformedAE.load_from_checkpoint(
    'checkpoints/stage_a/physae-stage-a-epoch=099-val_loss=0.0067.ckpt'
)

# Inference
model.eval()
predictions = model(noisy_spectra)
```

### Evaluate Performance

```bash
python project/scripts/evaluate.py \
    --checkpoint checkpoints/stage_a/last.ckpt \
    --batch_size 64
```

### Continue to Stage B1

After Stage A completes, train the refiners:

```python
# Load Stage A checkpoint
model = PhysicallyInformedAE.load_from_checkpoint('checkpoints/stage_a/best.ckpt')

# Switch to Stage B1 (refiners, frozen backbone)
model.training_stage = 'B1'

# Train with lower learning rate
# Stage B walkthroughs will be added alongside the YAML examples
```

## Jupyter Notebook Tutorials

Interactive tutorials are available under `project/examples/notebooks/`:

| Notebook | Focus |
|----------|-------|
| `stage_a_from_yaml.ipynb` | Stage A backbone training driven entirely by the bundled YAML configs |
| `noise_levels_sweep.ipynb` | Visualisation des spectres synthétiques avec balayage des profils de bruit (20 niveaux × 5 spectres) |

The notebook includes:

- Dependency checks and project-path bootstrapping
- YAML loading blocks that feed directly into `TrainingConfig`
- Dataset preparation using the modular `SpectraDataset`
- Model initialisation with `PhysicallyInformedAE`
- PyTorch Lightning trainer configuration, callbacks, and execution commands

> **Tip:** When customising the transitions file to include additional molecules (e.g. H₂O), make sure the corresponding mole fraction parameters are added to `config/params.py` so the dataset sampler can generate consistent examples.

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch_size 32` → `--batch_size 16`
2. Use mixed precision: `--precision 16`
3. Smaller encoder: `--encoder_variant l` → `--encoder_variant s`

### Training Unstable

**Symptoms:** Loss spikes, NaN values

**Solutions:**
1. Reduce learning rate: `--lr 1e-4` → `--lr 3e-5`
2. Use FP32 precision: `--precision 32`
3. Check data validity (parameter ranges, spectra)

### Slow Training

**Symptoms:** Low GPU utilization, long epoch times

**Solutions:**
1. Increase batch size: `--batch_size 64` → `--batch_size 128`
2. More data workers: `--num_workers 4` → `--num_workers 8`
3. Use mixed precision: `--precision bf16`
4. Use `persistent_workers=True` (already enabled)

### Slow Convergence

**Symptoms:** Validation loss not improving

**Solutions:**
1. Increase learning rate: `--lr 1e-4` → `--lr 3e-4`
2. More training data: `--train_samples 10000` → `--train_samples 50000`
3. Longer training: `--epochs 100` → `--epochs 150`
4. Check dataset parameter ranges (may be too narrow/wide)

## Customization

### Modify Parameter Ranges

Edit `setup_parameter_ranges()` in `train_stage_a.py`:

```python
NORM_PARAMS.update({
    'sig0': (5990.0, 6010.0),    # Your custom range
    'mf_CH4': (1e-7, 1e-2),       # Wider concentration range
    # ... other parameters
})
```

### Use Custom Loss Weights

Edit `create_model()` in `train_stage_a.py`:

```python
model = PhysicallyInformedAE(
    # ...
    lambda_params_mse=2.0,           # Emphasize parameter accuracy
    lambda_spectral_angle=0.05,
    lambda_peak_weighted_mse=0.3,
    # ...
)
```

### Add Custom Callbacks

```python
from pytorch_lightning.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Your custom logic
        pass

callbacks = [
    # ... existing callbacks
    MyCallback(),
]

trainer = pl.Trainer(callbacks=callbacks)
```

## Additional Resources

- **Main README:** `../README.md` - Project overview and structure
- **Configuration Guide:** `CONFIGURATION_GUIDE.md` - Detailed parameter documentation
- **Model Documentation:** `../models/autoencoder.py` - Model architecture details
- **Dataset Documentation:** `../data/dataset.py` - Dataset implementation

## Support

For issues, questions, or contributions:
1. Check `CONFIGURATION_GUIDE.md` for detailed explanations
2. Review troubleshooting section above
3. Check PyTorch Lightning documentation: https://lightning.ai/docs/pytorch/
4. Open an issue on GitHub

## License

[Your license information here]
