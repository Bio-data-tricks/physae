"""Inference service utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from physae.config import ExperimentConfig
from physae.models import PhysaeModel
from physae.utils import select_device


class PredictionRequest(BaseModel):
    spectra: List[float]


class PredictionResponse(BaseModel):
    reconstruction: List[float]
    params: List[float]


class InferenceService:
    def __init__(self, model: PhysaeModel, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, config: ExperimentConfig) -> "InferenceService":
        model = PhysaeModel(config.model, config.optimizer)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state(state)
        device = select_device(config.inference.device)
        return cls(model, device)

    def predict(self, spectra: torch.Tensor) -> Dict[str, torch.Tensor]:
        spectra = spectra.to(self.device)
        return self.model.predict(spectra)

    def predict_batch(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.predict(batch)


def create_app(service: InferenceService) -> FastAPI:
    app = FastAPI()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictionRequest) -> PredictionResponse:
        tensor = torch.tensor(req.spectra, dtype=torch.float32).unsqueeze(0)
        result = service.predict_batch(tensor)
        recon = result["reconstruction"].squeeze(0).cpu().tolist()
        params = result["params"].squeeze(0).cpu().tolist()
        return PredictionResponse(reconstruction=recon, params=params)

    return app


__all__ = ["InferenceService", "create_app", "PredictionRequest", "PredictionResponse"]
