"""PhysAE neural network modules."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from physae.config import ModelConfig, OptimizerConfig
from physae.metrics import mae, mse


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], latent_dim: int, dropout: float) -> None:
        super().__init__()
        layers = []
        last_c = in_channels
        for c in channels:
            layers.extend(
                [
                    nn.Conv1d(last_c, c, kernel_size=5, padding=2, stride=2),
                    nn.BatchNorm1d(c),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            last_c = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(last_c, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)
        pooled = self.pool(feats).squeeze(-1)
        return self.head(pooled)


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int, channels: list[int], latent_dim: int, dropout: float) -> None:
        super().__init__()
        layers = []
        last_c = latent_dim
        for c in channels:
            layers.append(nn.Linear(last_c, c))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            last_c = c
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(last_c, out_channels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.mlp(latent)
        return self.output(h)


class PhysaeModel(nn.Module):
    """Autoencoder predicting clean spectra and physical parameters."""

    def __init__(self, config: ModelConfig, optimizer_cfg: OptimizerConfig) -> None:
        super().__init__()
        self.config = config
        self.optimizer_cfg = optimizer_cfg
        self.encoder = ConvEncoder(1, config.encoder_channels, config.latent_dim, config.dropout)
        self.decoder = ConvDecoder(config.input_points, config.decoder_channels, config.latent_dim, config.dropout)
        self.param_head = nn.Linear(config.latent_dim, len(config.predict_parameters))

    def forward(self, noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(noisy.unsqueeze(1))
        recon = self.decoder(latent)
        params = self.param_head(latent)
        return recon, params

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        noisy = batch["noisy"]
        clean = batch["clean"]
        target_params = batch["params"]
        recon, params = self.forward(noisy)
        loss_recon = mse(recon, clean)
        loss_params = mae(params, target_params)
        loss = loss_recon + 0.1 * loss_params
        return loss, {"loss": loss, "loss_recon": loss_recon, "loss_params": loss_params}

    def predict(self, noisy: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            recon, params = self.forward(noisy)
        return {"reconstruction": recon, "params": params}

    def configure_optimiser(self) -> torch.optim.Optimizer:
        if self.optimizer_cfg.name.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay)
        raise ValueError(f"Unsupported optimizer {self.optimizer_cfg.name}")

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.config.__dict__}, path)

    def load_state(self, checkpoint: dict) -> None:
        self.load_state_dict(checkpoint["state_dict"])

    @classmethod
    def load(cls, path: str, optimizer_cfg: OptimizerConfig) -> "PhysaeModel":
        checkpoint = torch.load(path, map_location="cpu")
        model_cfg = ModelConfig(**checkpoint.get("config", {}))
        model = cls(model_cfg, optimizer_cfg)
        model.load_state(checkpoint)
        model.eval()
        return model
