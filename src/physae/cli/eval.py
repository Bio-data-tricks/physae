"""Command-line interface for evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from physae.config import ExperimentConfig, load_config
from physae.data import datamodule
from physae.models import PhysaeModel
from physae.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhysAE model")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: ExperimentConfig = load_config(args.config)
    data_module = datamodule.PhysaeDataModule(config.data)
    model = PhysaeModel(config.model, config.optimizer)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state(state)
    trainer = Trainer(config)
    trainer.evaluate(model, data_module)


if __name__ == "__main__":
    main()
