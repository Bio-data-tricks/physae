"""
DataLoader creation utilities.
"""
from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, **kwargs):
    """
    Create a PyTorch DataLoader.

    Args:
        dataset: PyTorch Dataset object.
        batch_size: Batch size.
        shuffle: Shuffle data (default: True).
        num_workers: Number of worker processes (default: 0).
        **kwargs: Additional DataLoader arguments.

    Returns:
        PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
