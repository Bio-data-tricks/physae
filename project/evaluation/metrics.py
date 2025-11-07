"""
Evaluation metrics for model performance.
"""
import torch


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        pred: Predicted values.
        target: Target values.

    Returns:
        MAE value.
    """
    return torch.abs(pred - target).mean().item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        pred: Predicted values.
        target: Target values.

    Returns:
        RMSE value.
    """
    return torch.sqrt(((pred - target) ** 2).mean()).item()


def compute_r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute R² score.

    Args:
        pred: Predicted values.
        target: Target values.

    Returns:
        R² score.
    """
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()
