"""Command-line interface for inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from physae.config import ExperimentConfig, load_config
from physae.inference.service import InferenceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with PhysAE")
    parser.add_argument("--config", type=Path, default=Path("configs/infer.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: ExperimentConfig = load_config(args.config)
    with args.input.open("r", encoding="utf-8") as handle:
        spectra = json.load(handle)
    tensor = torch.tensor(spectra, dtype=torch.float32).unsqueeze(0)
    service = InferenceService.from_checkpoint(args.checkpoint, config)
    result = service.predict_batch(tensor)
    output = {
        "reconstruction": result["reconstruction"].squeeze(0).cpu().tolist(),
        "params": result["params"].squeeze(0).cpu().tolist(),
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
