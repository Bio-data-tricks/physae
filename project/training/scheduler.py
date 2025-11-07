"""
Custom learning rate schedulers.
"""
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts, progressive decay, and linear warmup.
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: float = 1,
        eta_min: float = 0,
        decay_factor: float = 1.0,
        warmup_epochs: int = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Args:
            optimizer: Optimizer.
            T_0: Number of iterations for the first restart.
            T_mult: Factor to increase T_0 after each restart.
            eta_min: Minimum learning rate.
            decay_factor: Decay factor for max learning rate after each restart.
            warmup_epochs: Number of warmup epochs.
            last_epoch: The index of last epoch.
            verbose: If True, prints a message to stdout for each update.
        """
        self.T_0 = T_0
        self.T_i = float(T_0)
        self.T_mult = float(T_mult)
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_max_lrs = self.base_lrs.copy()
        self.T_cur = 0
        self.restart_count = 0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute learning rate."""
        current_epoch = self.last_epoch

        # Warmup phase
        if current_epoch < self.warmup_epochs:
            warmup_factor = (current_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine phase
        adjusted_epoch = current_epoch - self.warmup_epochs

        if adjusted_epoch == 0:
            self.T_cur = 0
            self.T_i = float(self.T_0)
        elif self.T_cur >= int(self.T_i):
            self.restart_count += 1
            self.T_cur = 0
            self.current_max_lrs = [lr * self.decay_factor for lr in self.current_max_lrs]
            self.T_i = self.T_i * self.T_mult

        lrs = []
        for max_lr in self.current_max_lrs:
            lr = self.eta_min + (max_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / self.T_i)
            ) / 2
            lrs.append(lr)

        self.T_cur += 1
        return lrs

    def get_schedule_info(self, max_epochs: int) -> dict:
        """Simulate schedule for visualization."""
        original_state = {
            'last_epoch': self.last_epoch,
            'T_cur': self.T_cur,
            'T_i': self.T_i,
            'restart_count': self.restart_count,
            'current_max_lrs': self.current_max_lrs.copy(),
        }

        self.last_epoch = -1
        self.T_cur = 0
        self.T_i = float(self.T_0)
        self.restart_count = 0
        self.current_max_lrs = self.base_lrs.copy()

        epochs = []
        lrs = []
        restart_epochs = []

        for epoch in range(max_epochs):
            self.step()
            epochs.append(epoch)
            lrs.append(self.get_last_lr()[0])

            adjusted_epoch = epoch - self.warmup_epochs
            if adjusted_epoch > 0 and self.T_cur == 1:
                restart_epochs.append(epoch)

        # Restore state
        self.last_epoch = original_state['last_epoch']
        self.T_cur = original_state['T_cur']
        self.T_i = original_state['T_i']
        self.restart_count = original_state['restart_count']
        self.current_max_lrs = original_state['current_max_lrs']

        return {
            'epochs': epochs,
            'lrs': lrs,
            'restart_epochs': restart_epochs,
            'warmup_end': self.warmup_epochs - 1 if self.warmup_epochs > 0 else None,
        }
