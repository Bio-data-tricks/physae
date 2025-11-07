"""Command-line interface for training."""
from __future__ import annotations

import argparse
from pathlib import Path

from physae.config import ExperimentConfig, load_config
from physae.data import datamodule
from physae.models import PhysaeModel
from physae.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhysAE model")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: ExperimentConfig = load_config(args.config)
    data_module = datamodule.PhysaeDataModule(config.data)
    model = PhysaeModel(config.model, config.optimizer)
    trainer = Trainer(config)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
