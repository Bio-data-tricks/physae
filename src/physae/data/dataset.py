"""Dataset definitions for PhysAE."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from torch.utils.data import Dataset

from physae.config import DataConfig, NoiseConfig


@dataclass
class SpectraSample:
    noisy: torch.Tensor
    clean: torch.Tensor
    params: torch.Tensor
    scale: torch.Tensor


class SpectraDataset(Dataset[Dict[str, torch.Tensor]]):
    """Synthetic dataset producing noisy spectra from simple peak models."""

    def __init__(self, config: DataConfig, *, stage: str = "train") -> None:
        super().__init__()
        self.config = config
        self.stage = stage
        self.num_samples = {
            "train": config.train_samples,
            "val": config.val_samples,
            "test": config.test_samples,
        }[stage]
        self.generator = torch.Generator().manual_seed(config.seed + hash(stage) % 997)
        self.param_names = list(config.parameter_bounds.keys())

    def __len__(self) -> int:
        return self.num_samples

    def _sample_parameters(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        params = []
        named = {}
        for name in self.param_names:
            low, high = self.config.parameter_bounds[name]
            value = torch.empty(1, generator=self.generator).uniform_(float(low), float(high)).item()
            params.append(value)
            named[name] = value
        return torch.tensor(params, dtype=torch.float32), named

    def _generate_clean(self, bounds: Dict[str, float]) -> torch.Tensor:
        n = self.config.num_points
        x = torch.linspace(0.0, 1.0, n)
        baseline = bounds["baseline0"] + bounds["baseline1"] * (x - 0.5) + bounds["baseline2"] * (x - 0.5) ** 2
        center = bounds["peak_position"]
        width = bounds["peak_width"] / n
        height = bounds["peak_height"]
        peak = height * torch.exp(-0.5 * ((x - center) / (width + 1e-6)) ** 2)
        return (baseline + peak).to(torch.float32)

    def _apply_noise(self, clean: torch.Tensor) -> torch.Tensor:
        noise_cfg: NoiseConfig = self.config.noise
        g = self.generator
        mult = (torch.randn_like(clean, generator=g) * noise_cfg.multiplicative_std).exp()
        add = torch.randn_like(clean, generator=g) * noise_cfg.additive_std
        noisy = clean * mult + add
        if noise_cfg.drift_strength > 0:
            kernel_size = max(3, int(math.sqrt(self.config.num_points) // 2 * 2 + 1))
            kernel = torch.ones(kernel_size, dtype=torch.float32)
            kernel = kernel / kernel.sum()
            padded = torch.nn.functional.pad(
                torch.randn_like(clean, generator=g),
                (kernel_size // 2, kernel_size // 2),
                mode="reflect",
            )
            drift = torch.nn.functional.conv1d(
                padded.unsqueeze(0).unsqueeze(0),
                kernel.view(1, 1, -1),
            ).squeeze()
            noisy = noisy + drift * noise_cfg.drift_strength
        if noise_cfg.spike_probability > 0:
            spike_mask = torch.rand_like(clean, generator=g) < noise_cfg.spike_probability
            spikes = torch.randn_like(clean, generator=g)
            noisy = torch.where(spike_mask, noisy + spikes * 0.5, noisy)
        return noisy

    def _scale(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.clamp(signal.abs().max(), min=1e-6)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        params_tensor, params_named = self._sample_parameters()
        clean = self._generate_clean(params_named)
        noisy = self._apply_noise(clean)
        scale = self._scale(noisy)
        return {
            "noisy": (noisy / scale).to(torch.float32),
            "clean": (clean / scale).to(torch.float32),
            "params": params_tensor,
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


def iter_batches(dataset: Dataset[Dict[str, torch.Tensor]], batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
    """Utility for deterministic batching in tests."""

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        merged = {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}
        yield merged
