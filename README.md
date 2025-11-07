# PhysAE

PhysAE is a modular template for building physically informed auto-encoders on synthetic spectroscopy data. The project demonstrates modern MLOps practices including configuration management, reproducible training, testing, and deployment assets.

## Quickstart

```bash
make setup
make train
make test
```

## Project layout

- `configs/`: YAML configuration files for training, evaluation, inference, and deployment.
- `data/`: Input data artefacts (read-only at runtime).
- `outputs/`: Auto-generated run artefacts such as checkpoints and logs.
- `src/physae/`: Python package containing data pipelines, models, training loop, inference service, and utilities.
- `tests/`: Pytest-based unit and integration coverage.
- `serve.py`: FastAPI entrypoint for serving checkpoints.

## CLI usage

```bash
python -m physae.cli.train --config configs/train.yaml
python -m physae.cli.eval --config configs/eval.yaml --checkpoint outputs/train/model.ckpt
python -m physae.cli.infer --config configs/infer.yaml --checkpoint outputs/train/model.ckpt --input data/sample_input.json --output outputs/prediction.json
```

## Deployment

Build the Docker image and run the FastAPI service:

```bash
docker build -t physae .
docker run -p 8000:8000 physae
```

Then call the API:

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"spectra": [0.0, 0.1, 0.2]}'
```

## Legacy

The original monolithic implementation is stored under `legacy/physae_monolith.py` for reference during migration.
