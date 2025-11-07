# Migration guide

This guide outlines the recommended steps to migrate workloads from the legacy `physae.py` monolith to the modular PhysAE package.

1. **Freeze legacy artefacts**
   - Tag the current repository state or archive `legacy/physae_monolith.py`.
   - Export any existing checkpoints and note the hyper-parameters used to train them.

2. **Map modules**
   - Legacy dataset generation → `src/physae/data/` (`SpectraDataset`, `PhysaeDataModule`).
   - Legacy neural network + Lightning module → `src/physae/models/autoencoder.py` and `src/physae/training/trainer.py`.
   - Ad-hoc scripts → CLI entrypoints in `src/physae/cli/` and FastAPI service in `serve.py`.

3. **Convert checkpoints**
   - For models trained with the monolith, export state dictionaries as PyTorch checkpoints.
   - Implement a conversion script that maps legacy parameter names to the new module layout (e.g. rename linear layers to `encoder.*` / `decoder.*`).
   - Validate converted checkpoints by running `tests/test_inference.py` with the converted artefacts.

4. **Validate behaviour**
   - Run `make test` to execute unit and integration suites.
   - Compare reconstruction metrics on a golden validation set by plugging custom datasets into `SpectraDataset`.

5. **Decommission legacy path**
   - Once parity is confirmed, delete or archive the `legacy/` directory.
   - Update deployment manifests (Docker, CI) to point at the new CLI and `serve.py` entrypoints.

## Additional verification

- Reproduce a training run with `python -m physae.cli.train --config configs/train.yaml` and confirm loss trends.
- Smoke-test the FastAPI server locally (`python serve.py`) before shipping Docker images.
- Ensure W&B (if enabled) logs match the expected experiment naming conventions.
