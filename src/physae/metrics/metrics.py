"""Metric utilities."""
from __future__ import annotations

from typing import Dict

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def summarize_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in metrics.items()}
