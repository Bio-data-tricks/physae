"""
Utilities for distributed training.
"""
import torch


def is_rank0() -> bool:
    """
    Check if the current process is rank 0 in distributed training.

    Returns:
        True if this is the main process (rank 0) or if distributed training is not initialized.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
