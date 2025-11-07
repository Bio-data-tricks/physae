"""
Spectral angle loss for spectral data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAngleLoss(nn.Module):
    """
    Loss based on spectral angle (cosine similarity of L2-normalized spectra).
    More robust than MSE for global amplitude variations.
    Replaces: w_corr_raw, w_corr_d1, w_corr_d2
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral angle loss.

        Args:
            pred: [B, N] predicted spectra.
            target: [B, N] target spectra.

        Returns:
            Scalar loss (mean over batch).
        """
        # L2 normalization
        pred_norm = F.normalize(pred, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=1, eps=self.eps)

        # Cosine similarity
        cosine_sim = (pred_norm * target_norm).sum(dim=1)

        # Clamp for numerical stability
        cosine_sim = torch.clamp(cosine_sim, -1.0 + self.eps, 1.0 - self.eps)

        # Convert to loss (1 - similarity)
        return (1.0 - cosine_sim).mean()
