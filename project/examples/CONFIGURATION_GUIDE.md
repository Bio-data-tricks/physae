# Stage A Training Configuration Guide

This guide explains the parameter choices for training Stage A of the PhysicallyInformedAE model.

## Overview

**Stage A** focuses on training the encoder backbone and initial parameter prediction head. The cascade refiners are present in the architecture but remain inactive during this stage. This allows the backbone to learn robust feature extraction from spectra before fine-tuning with refinement.

## Dataset Parameters

### Parameter Ranges (`NORM_PARAMS`)

The dataset generates synthetic spectra by sampling parameters from specified ranges. These ranges should match your target application domain.

```python
NORM_PARAMS = {
    'sig0': (5995.0, 6005.0),       # Center wavenumber [cm^-1]
    'dsig': (8.0, 12.0),             # Spectral width [cm^-1]
    'mf_CH4': (1e-6, 5e-3),          # CH4 molar fraction (log scale)
    'baseline0': (0.8, 1.2),         # Baseline constant
    'baseline1': (-0.02, 0.02),      # Baseline linear term
    'baseline2': (-0.001, 0.001),    # Baseline quadratic term
    'P': (0.5, 1.2),                 # Pressure [atm]
    'T': (250.0, 320.0),             # Temperature [K]
}
```

#### Parameter Explanations

1. **sig0** (Center Wavenumber)
   - Physical meaning: Center of the spectral window in wavenumber units
   - Example range: 5995-6005 cm⁻¹ (near-IR CH₄ absorption band)
   - Typical values: Depends on your spectrometer and target molecules
   - Notes: Should match your HITRAN data range

2. **dsig** (Spectral Width)
   - Physical meaning: Width of the spectral window
   - Example range: 8-12 cm⁻¹
   - Typical values: 5-20 cm⁻¹ depending on application
   - Notes: Larger windows capture more spectral features but require more computation

3. **mf_CH4** (CH₄ Molar Fraction)
   - Physical meaning: Concentration of methane in the gas mixture
   - Example range: 1e-6 to 5e-3 (1 ppm to 0.5%)
   - Typical values:
     - Atmospheric: ~1.8 ppm (1.8e-6)
     - Industrial monitoring: 100 ppm to 1% (1e-4 to 1e-2)
     - Calibration gas: 1-5%
   - Notes: Uses **log scale** during normalization (see `LOG_SCALE_PARAMS`)

4. **baseline0, baseline1, baseline2** (Polynomial Baseline)
   - Physical meaning: Coefficients for polynomial baseline correction
     - baseline0: constant offset (typical instrument drift)
     - baseline1: linear slope (gradual intensity change)
     - baseline2: quadratic curvature (optical fringing)
   - Example ranges:
     - baseline0: 0.8-1.2 (±20% intensity variation)
     - baseline1: -0.02 to 0.02 (gentle slopes)
     - baseline2: -0.001 to 0.001 (subtle curvature)
   - Notes: These compensate for instrumental artifacts

5. **P** (Pressure)
   - Physical meaning: Total gas pressure
   - Example range: 0.5-1.2 atm
   - Typical values:
     - Sea level: ~1.0 atm
     - Mountain/aircraft: 0.5-0.8 atm
     - Pressurized cell: 1-5 atm
   - Notes: Affects line broadening (collisional effects)

6. **T** (Temperature)
   - Physical meaning: Gas temperature
   - Example range: 250-320 K (-23°C to 47°C)
   - Typical values:
     - Room temperature: ~296 K (23°C)
     - Atmospheric: 220-300 K (stratosphere to surface)
     - Industrial: 273-373 K (0-100°C)
   - Notes: Affects line intensities via partition functions and Boltzmann distribution

### Reference Ranges from physae.py

The original `physae.py` script trains the autoencoder on the following
parameter intervals:

```python
NORM_PARAMS = {
    'sig0': (3085.37, 3085.52),
    'dsig': (0.001502, 0.001559),
    'mf_CH4': (1e-7, 2.9e-5),
    'mf_H2O': (1e-7, 5e-4),
    'baseline0': (0.999999, 1.00001),
    'baseline1': (-5.0e-4, -2.0e-4),
    'baseline2': (-7.505155e-8, 3.77485e-9),
    'P': (400.0, 600.0),
    'T': (302.65, 312.65),
}
```

These values—augmented with a log-scaled H₂O mole-fraction range to support the
sample HITRAN catalogue—are bundled in
`project/config/data/parameters_default.yaml` and match the
`build_data_and_model` helper shipped with the monolithic script.

### Alternative Configuration Examples

#### Example 1: Atmospheric Monitoring
```python
NORM_PARAMS = {
    'sig0': (6000.0, 6010.0),
    'dsig': (10.0, 15.0),
    'mf_CH4': (1e-6, 1e-4),      # 1-100 ppm
    'baseline0': (0.9, 1.1),
    'baseline1': (-0.01, 0.01),
    'baseline2': (-0.0005, 0.0005),
    'P': (0.6, 1.0),              # High altitude to sea level
    'T': (240.0, 300.0),          # Cold stratosphere to warm surface
}
```

#### Example 2: Industrial Leak Detection
```python
NORM_PARAMS = {
    'sig0': (5995.0, 6005.0),
    'dsig': (8.0, 12.0),
    'mf_CH4': (1e-5, 1e-2),      # 10 ppm to 1%
    'baseline0': (0.7, 1.3),      # Higher baseline variation
    'baseline1': (-0.03, 0.03),
    'baseline2': (-0.002, 0.002),
    'P': (0.9, 1.1),              # Near atmospheric
    'T': (273.0, 323.0),          # 0-50°C
}
```

#### Example 3: Laboratory Calibration
```python
NORM_PARAMS = {
    'sig0': (5998.0, 6002.0),    # Narrow range, high precision
    'dsig': (4.0, 6.0),           # Smaller window
    'mf_CH4': (1e-4, 5e-2),      # 100 ppm to 5%
    'baseline0': (0.95, 1.05),    # Stable baseline
    'baseline1': (-0.005, 0.005),
    'baseline2': (-0.0001, 0.0001),
    'P': (0.95, 1.05),            # Controlled pressure
    'T': (293.0, 298.0),          # Room temperature ±2.5K
}
```

### Dataset Size

```python
train_samples = 500_000  # Training spectra (Stage A reference setup)
val_samples = 5_000      # Validation spectra
num_points = 800         # Spectral resolution (pixels)
```

**Recommendations:**
- Fast iteration: 10,000 train / 1,000 val
- Balanced: 50,000 train / 5,000 val
- Full reference: 500,000 train / 5,000 val (physae.py default)

### Frequency Grid

The frequency grid maps pixel indices to wavenumber positions. Provide the
calibration polynomial alongside your transitions YAML using the
``poly_frequency`` section:

```yaml
poly_frequency:
  CH4: [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]
```

When omitted, the training pipeline falls back to the linear grid defined by
the predicted ``sig0`` and ``dsig`` parameters, matching the original
``physae.py`` behaviour when no calibration polynomial is available.

## Model Configuration

### Architecture Parameters

```python
encoder_variant = 's'      # EfficientNet variant: 's' (small), 'm' (medium), 'l' (large)
num_cascade_refiners = 3   # Number of refinement stages
mlp_dropout = 0.10         # Dropout in parameter prediction heads
refiner_dropout = 0.05     # Dropout in refiners
```

#### Encoder Variants

| Variant | Parameters | Speed | Performance | Recommended Use |
|---------|-----------|-------|-------------|-----------------|
| 's' (small) | ~3M | Fast | Good | Development, fast iteration |
| 'm' (medium) | ~6M | Medium | Better | Balanced, general purpose |
| 'l' (large) | ~12M | Slow | Best | High accuracy requirements |

**Stage A Recommendation:** Start with 's' for fast prototyping, then scale to 'm' or 'l' if needed.

#### Number of Refiners

- **1 refiner:** Minimal refinement, faster training
- **3 refiners:** Default, good balance (recommended)
- **5 refiners:** Maximum refinement, slower, diminishing returns

**Stage A Note:** Refiners are present but inactive. They will be trained in Stage B.

### Training Hyperparameters

```python
epochs = 100           # Training epochs for Stage A
lr = 1e-4              # Initial learning rate
weight_decay = 0.01    # L2 regularization
optimizer = 'adamw'    # Optimizer: 'adamw' or 'lion'
batch_size = 64        # Batch size (adjust for GPU memory)
```

#### Learning Rate

- **1e-3:** Aggressive, fast convergence, risk of instability
- **1e-4:** Default, good balance (recommended)
- **3e-5:** Conservative, stable, slower convergence

**Recommendations:**
- Start with 1e-4
- If training is unstable (loss spikes), reduce to 3e-5
- If convergence is too slow, try 3e-4

#### Optimizer

**AdamW (default):**
- Stable, well-tested
- Good for most applications
- Weight decay properly decoupled

**Lion:**
- Memory efficient (useful for large models)
- Can converge faster
- May require learning rate tuning (try 3e-5 with Lion)

#### Batch Size

Depends on GPU memory:

| GPU Memory | Recommended Batch Size |
|-----------|------------------------|
| 8 GB | 16-32 |
| 12 GB | 32-64 |
| 16 GB | 64-128 |
| 24 GB+ | 128-256 |

**Notes:**
- Larger batches → more stable gradients, faster training
- Smaller batches → more noise, better generalization
- If OOM (out of memory), reduce batch size by half

### Loss Configuration

```python
use_relobralo = True       # Adaptive loss balancing
lambda_params_mse = 1.0    # MSE loss weight
lambda_spectral_angle = 0.1    # Spectral angle loss weight
lambda_peak_weighted_mse = 0.5  # Peak-weighted MSE weight
```

**ReLoBRaLo (Recommended: Enabled)**

ReLoBRaLo automatically balances multiple loss functions during training, eliminating the need to manually tune loss weights.

- **Enabled (True):** Loss weights adapt automatically. Initial weights are starting points.
- **Disabled (False):** Use fixed loss weights (requires manual tuning).

**Manual Weight Tuning (if ReLoBRaLo disabled):**

```python
# Balanced (default)
lambda_params_mse = 1.0
lambda_spectral_angle = 0.1
lambda_peak_weighted_mse = 0.5

# Emphasize parameter accuracy
lambda_params_mse = 2.0
lambda_spectral_angle = 0.05
lambda_peak_weighted_mse = 0.3

# Emphasize spectral reconstruction
lambda_params_mse = 0.5
lambda_spectral_angle = 0.2
lambda_peak_weighted_mse = 1.0
```

### Learning Rate Schedule

```python
lr_scheduler = 'cosine_warmup'
warmup_epochs = 5      # Linear warmup period
t_0 = 50               # Cosine annealing period
t_mult = 2             # Period multiplier after restart
eta_min_ratio = 0.01   # Min LR = 1% of initial
```

**Cosine Annealing with Warm Restarts:**
- Starts with linear warmup (5 epochs)
- Follows cosine decay from max to min LR
- Periodic restarts help escape local minima
- LR range: [lr × eta_min_ratio, lr]

**Alternative: No Scheduler**
```python
lr_scheduler = None  # Constant learning rate
```

## Hardware Configuration

```python
gpus = 1              # Number of GPUs (0 for CPU)
precision = '32'      # Training precision: '32', '16', 'bf16'
num_workers = 4       # Data loading workers
```

### Precision

| Precision | Speed | Memory | Accuracy | Notes |
|-----------|-------|--------|----------|-------|
| '32' | Baseline | Baseline | Best | Default, most stable |
| '16' | ~2x faster | ~0.5x | Good | Requires gradient scaling |
| 'bf16' | ~2x faster | ~0.5x | Good | Better than FP16, A100/H100 only |

**Recommendations:**
- Start with '32' for stability
- Try '16' or 'bf16' for 2x speedup (if GPU supports it)
- Monitor for NaN/Inf if using mixed precision

## Complete Configuration Examples

### Quick Test (Fast Iteration)

```bash
python project/examples/train_stage_a.py \
    --train_samples 1000 \
    --val_samples 100 \
    --epochs 20 \
    --batch_size 32 \
    --encoder_variant s \
    --lr 1e-4 \
    --gpus 1
```

**Use case:** Quick sanity check, ~5-10 minutes

### Default Configuration

```bash
python project/examples/train_stage_a.py \
    --train_samples 10000 \
    --val_samples 1000 \
    --epochs 100 \
    --batch_size 64 \
    --encoder_variant s \
    --num_refiners 3 \
    --lr 1e-4 \
    --optimizer adamw \
    --gpus 1 \
    --precision 32
```

**Use case:** Standard training, ~2-4 hours on modern GPU

### High Performance

```bash
python project/examples/train_stage_a.py \
    --train_samples 50000 \
    --val_samples 5000 \
    --epochs 150 \
    --batch_size 128 \
    --encoder_variant l \
    --num_refiners 3 \
    --lr 1e-4 \
    --optimizer adamw \
    --weight_decay 0.01 \
    --gpus 1 \
    --precision bf16 \
    --num_workers 8
```

**Use case:** Best accuracy, requires A100/H100, ~6-12 hours

### Multi-GPU Training

```bash
python project/examples/train_stage_a.py \
    --train_samples 50000 \
    --val_samples 5000 \
    --epochs 100 \
    --batch_size 256 \
    --encoder_variant m \
    --lr 1e-4 \
    --gpus 4 \
    --precision bf16 \
    --num_workers 16
```

**Use case:** Large-scale training, 4 GPUs, ~2-4 hours

## Data Requirements

### HITRAN Transition Data

To enable physics-based simulation, download HITRAN line data:

1. Visit: https://hitran.org/lbl/
2. Select molecules:
   - CH₄ (Methane): Molecule ID 06
   - H₂O (Water): Molecule ID 01
   - Add others as needed
3. Set wavenumber range (e.g., 5995-6005 cm⁻¹)
4. Download in HITRAN format
5. Convert to CSV (or adapt the loader)
6. Save to `./data/hitran/`

Example files:
```
data/hitran/
├── 06_hit20_5995_6005.csv  # CH4 transitions
└── 01_hit20_5995_6005.csv  # H2O transitions
```

### TIPS Partition Functions

Download HITRAN TIPS_2021 data:

1. Visit: https://hitran.org/docs/iso-meta/
2. Download QTpy files for your molecules
3. Save to `./QTpy/` directory

Example structure:
```
QTpy/
├── q1.QTpy   # H2O
├── q6.QTpy   # CH4
└── ...
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--batch_size 32` → `--batch_size 16`
2. Use mixed precision: `--precision 16` or `--precision bf16`
3. Use smaller encoder: `--encoder_variant l` → `--encoder_variant s`

### Training Unstable (Loss Spikes)

1. Reduce learning rate: `--lr 1e-4` → `--lr 3e-5`
2. Enable gradient clipping (already enabled: `gradient_clip_val=1.0`)
3. Increase warmup: Edit script, set `warmup_epochs=10`

### Slow Convergence

1. Increase learning rate: `--lr 1e-4` → `--lr 3e-4`
2. Larger batch size: `--batch_size 64` → `--batch_size 128`
3. More training samples: `--train_samples 10000` → `--train_samples 50000`

### NaN/Inf in Loss

1. Use FP32 precision: `--precision 32`
2. Check parameter ranges (ensure no division by zero)
3. Reduce learning rate
4. Check for data issues (corrupted spectra, invalid parameters)

## Next Steps After Stage A

Once Stage A training completes:

1. **Evaluate Performance:**
   ```bash
   python project/scripts/evaluate.py \
       --checkpoint checkpoints/stage_a/physae-stage-a-best.ckpt
   ```

2. **Stage B1 (Train Refiners, Frozen Backbone):**
   - Load Stage A checkpoint
   - Set `training_stage='B1'`
   - Use lower learning rate (1e-5)
   - Train for fewer epochs (~30-50)

3. **Stage B2 (Full Fine-Tuning):**
   - Load Stage B1 checkpoint
   - Set `training_stage='B2'`
   - Use very low learning rate (1e-6)
   - Train for fewer epochs (~20-30)

4. **Stage DEN (Train Denoiser):**
   - Load Stage B2 checkpoint
   - Set `training_stage='DEN'`
   - Train denoiser only
   - Use moderate learning rate (1e-4)

## References

- **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- **ReLoBRaLo:** Birkenes et al., "ReLoBRaLo: Adaptive Loss Balancing", 2020
- **HITRAN:** Gordon et al., "The HITRAN2020 molecular spectroscopic database", JQSRT 2022
- **TIPS:** Gamache et al., "TIPS-2021", JQSRT 2021
