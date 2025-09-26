"""Loss weighting utilities."""

from __future__ import annotations

import random
from typing import Iterable, List

import torch


class ReLoBRaLoLoss:
    """Relative loss balancing as proposed by Braumann & Lochner (ReLoBRaLo)."""

    def __init__(self, loss_names: Iterable[str], alpha: float = 0.9, tau: float = 1.0, history_len: int = 10):
        self.loss_names = list(loss_names)
        self.alpha = alpha
        self.tau = tau
        self.history_len = history_len
        self.loss_history = {name: [] for name in loss_names}
        self.weights = torch.ones(len(loss_names), dtype=torch.float32)

    def compute_weights(self, current_losses: torch.Tensor) -> torch.Tensor:
        for idx, name in enumerate(self.loss_names):
            self.loss_history[name].append(float(current_losses[idx].detach().cpu()))
            if len(self.loss_history[name]) > self.history_len:
                self.loss_history[name].pop(0)
        if len(self.loss_history[self.loss_names[0]]) < 2:
            return self.weights.to(current_losses.device)
        relative = []
        for name in self.loss_names:
            history = self.loss_history[name]
            j = random.randint(0, len(history) - 2)
            ratio = history[-1] / (history[j] + 1e-8)
            relative.append(ratio)
        relative_tensor = torch.tensor(relative, dtype=torch.float32, device=current_losses.device)
        mean_relative = relative_tensor.mean()
        balancing = mean_relative / (relative_tensor + 1e-8)
        new_weights = len(self.loss_names) * torch.softmax(balancing / self.tau, dim=0)
        self.weights = (self.alpha * self.weights.to(current_losses.device) + (1 - self.alpha) * new_weights).detach()
        return self.weights.to(current_losses.device)
