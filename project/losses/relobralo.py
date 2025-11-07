"""
ReLoBRaLo (Relative Loss Balancing with Random Lookback) loss.
"""
import torch


class ReLoBRaLoLoss:
    """
    ReLoBRaLo with deterministic PyTorch sampling (DDP friendly).
    Supports both num_losses (int) and loss_names (list) for compatibility.
    """

    def __init__(self, num_losses=None, loss_names=None, alpha=0.9, tau=1.0, history_len=10, seed=12345):
        """
        Args:
            num_losses: Number of losses (int) - recommended.
            loss_names: List of loss names (list) - for old API compatibility.
            alpha: EMA smoothing factor.
            tau: Temperature for softmax.
            history_len: History length.
            seed: Seed for reproducibility.
        """
        # Compatibility with both signatures
        if num_losses is not None:
            self.num_losses = int(num_losses)
            self.loss_names = [f"loss_{i}" for i in range(self.num_losses)]
        elif loss_names is not None:
            self.loss_names = list(loss_names)
            self.num_losses = len(self.loss_names)
        else:
            raise ValueError("Provide either num_losses (int) or loss_names (list)")

        self.alpha = float(alpha)
        self.tau = float(tau)
        self.history_len = int(history_len)
        self.loss_history = {name: [] for name in self.loss_names}
        self.weights = torch.ones(self.num_losses, dtype=torch.float32)
        self.last_weights = torch.ones(self.num_losses, dtype=torch.float32) / self.num_losses
        self._g = torch.Generator(device="cpu")
        if seed is not None:
            self._g.manual_seed(int(seed))

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._g.manual_seed(int(seed))

    def to(self, device=None, dtype=None):
        """Move weights to device/dtype."""
        if device is not None or dtype is not None:
            self.weights = self.weights.to(
                device=device or self.weights.device,
                dtype=dtype or self.weights.dtype
            )
            self.last_weights = self.last_weights.to(
                device=device or self.last_weights.device,
                dtype=dtype or self.last_weights.dtype
            )
        return self

    def _append_history(self, current_losses):
        """Append current losses to history."""
        for i, name in enumerate(self.loss_names):
            if i < len(current_losses):
                self.loss_history[name].append(float(current_losses[i].detach().cpu()))
                if len(self.loss_history[name]) > self.history_len:
                    self.loss_history[name].pop(0)

    @torch.no_grad()
    def compute_weights(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weights for losses.

        Args:
            current_losses: [N] tensor of individual losses.

        Returns:
            weights: [N] weighted losses.
        """
        device = current_losses.device
        dtype = current_losses.dtype

        # Check consistency
        if current_losses.shape[0] != self.num_losses:
            raise ValueError(
                f"Inconsistent number of losses: expected {self.num_losses}, "
                f"received {current_losses.shape[0]}"
            )

        self._append_history(current_losses)

        # Need at least 2 values
        if len(self.loss_history[self.loss_names[0]]) < 2:
            w = self.weights.to(device=device, dtype=dtype)
            self.last_weights = w.detach().cpu()
            return w

        ratios = []
        for name in self.loss_names:
            hist = self.loss_history[name]
            j = int(torch.randint(low=0, high=len(hist) - 1, size=(), generator=self._g).item())
            num = float(hist[-1])
            den = float(hist[j]) + 1e-8
            ratios.append(num / den)

        ratios_t = torch.tensor(ratios, device=device, dtype=dtype)
        mean_rel = ratios_t.mean()
        balancing = mean_rel / (ratios_t + 1e-8)

        K = self.num_losses
        new_w = K * torch.softmax(balancing / self.tau, dim=0)

        w_old = self.weights.to(device=device, dtype=dtype)
        w_new = self.alpha * w_old + (1.0 - self.alpha) * new_w

        self.weights = w_new.detach().cpu()
        self.last_weights = w_new.detach().cpu()

        return w_new

    def __call__(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute total weighted loss with ReLoBRaLo.

        Args:
            current_losses: [N] tensor of individual losses.

        Returns:
            total_loss: Scalar tensor of total weighted loss.
        """
        weights = self.compute_weights(current_losses)
        total_loss = (weights * current_losses).sum()
        return total_loss

    def forward(self, current_losses: torch.Tensor) -> torch.Tensor:
        """Alias for __call__ (nn.Module style compatibility)."""
        return self.__call__(current_losses)
