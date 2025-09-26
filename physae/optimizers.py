"""Custom optimizers used by PhysAE."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Implementation of the Lion optimizer.

    The algorithm is described in https://arxiv.org/abs/2302.06675 and mirrors the
    reference PyTorch implementation released by the authors. The optimizer uses a
    sign-based update of an exponential moving average of gradients.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group.get("weight_decay", 0.0)
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                param.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


__all__ = ["Lion"]
