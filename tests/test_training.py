from __future__ import annotations

import yaml

from physae.config import config_to_dict, load_config
from physae.data import PhysaeDataModule
from physae.models import PhysaeModel
from physae.training import Trainer


def test_trainer_runs_one_epoch(tmp_path):
    config = load_config("configs/train.yaml")
    config.trainer.max_epochs = 1
    config.data.train_samples = config.data.batch_size
    config.data.val_samples = config.data.batch_size
    config.logging.output_dir = tmp_path / "outputs"
    config.logging.run_name = "pytest-train"

    dm = PhysaeDataModule(config.data)
    model = PhysaeModel(config.model, config.optimizer)
    trainer = Trainer(config)
    ckpt_path = trainer.fit(model, dm)
    assert ckpt_path.exists()

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle)
    assert config_path.exists()
