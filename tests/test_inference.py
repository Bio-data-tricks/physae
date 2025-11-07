from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from physae.config import config_to_dict, load_config
from physae.data import PhysaeDataModule, SpectraDataset
from physae.inference.service import InferenceService, create_app
from physae.models import PhysaeModel
from physae.training import Trainer


def _train_checkpoint(tmp_path: Path):
    config = load_config("configs/train.yaml")
    config.trainer.max_epochs = 1
    config.data.train_samples = config.data.batch_size
    config.data.val_samples = config.data.batch_size
    config.logging.output_dir = tmp_path / "outputs"
    config.logging.run_name = "pytest-infer"

    dm = PhysaeDataModule(config.data)
    model = PhysaeModel(config.model, config.optimizer)
    trainer = Trainer(config)
    ckpt_path = trainer.fit(model, dm)
    return ckpt_path, config


def test_inference_cli(tmp_path):
    ckpt_path, config = _train_checkpoint(tmp_path)
    config_path = tmp_path / "infer.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle)

    dataset = SpectraDataset(config.data)
    sample = dataset[0]
    input_path = tmp_path / "input.json"
    with input_path.open("w", encoding="utf-8") as handle:
        json.dump(sample["noisy"].tolist(), handle)
    output_path = tmp_path / "output.json"

    cmd = [
        sys.executable,
        "-m",
        "physae.cli.infer",
        "--config",
        str(config_path),
        "--checkpoint",
        str(ckpt_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    subprocess.run(cmd, check=True, env=env)
    assert output_path.exists()


def test_inference_api(tmp_path):
    ckpt_path, config = _train_checkpoint(tmp_path)
    service = InferenceService.from_checkpoint(ckpt_path, config)
    app = create_app(service)
    client = TestClient(app)

    dataset = SpectraDataset(config.data)
    sample = dataset[0]
    response = client.post("/predict", json={"spectra": sample["noisy"].tolist()})
    assert response.status_code == 200
    payload = response.json()
    assert "reconstruction" in payload
    assert "params" in payload
