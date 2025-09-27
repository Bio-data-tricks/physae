"""Smoke tests for the public factory helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from physae.factory import build_data_and_model


def test_build_data_and_model_supports_legacy_refiner_keys() -> None:
    """Ensure the refiner accepts legacy configuration keys via overrides."""

    model, train_loader, val_loader = build_data_and_model(
        config_overrides={
            "n_train": 1,
            "n_val": 1,
            "batch_size": 1,
            "model": {
                "refiner": {
                    # ``width_mult`` is the historical key that should now map to
                    # ``encoder_width_mult`` inside the refiner kwargs.
                    "width_mult": 0.75,
                }
            },
        }
    )

    assert model.refiner is not None
    assert train_loader is not None
    assert val_loader is not None
