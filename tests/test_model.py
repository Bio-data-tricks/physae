from __future__ import annotations

import torch

from physae.config import load_config
from physae.models import PhysaeModel


def test_model_forward_shapes():
    config = load_config("configs/train.yaml")
    model = PhysaeModel(config.model, config.optimizer)
    dummy = torch.randn(2, config.model.input_points)
    recon, params = model(dummy)
    assert recon.shape == (2, config.model.input_points)
    assert params.shape == (2, len(config.model.predict_parameters))
