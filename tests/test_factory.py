"""Regression tests for :mod:`physae.factory`."""

from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on test environment
    torch = None  # type: ignore[assignment]


def _small_build_kwargs(**overrides):
    params = {
        "seed": 0,
        "n_points": 32,
        "n_train": 4,
        "n_val": 2,
        "batch_size": 2,
    }
    params.update(overrides)
    return params


def test_build_data_and_model_default_refiner_config():
    """The default configuration should construct without keyword errors."""

    if torch is None:  # pragma: no cover - depends on test environment
        pytest.skip("torch is required for factory tests")

    from physae.factory import build_data_and_model
    from physae.model import PhysicallyInformedAE

    model, (train_loader, val_loader), metadata = build_data_and_model(**_small_build_kwargs())

    assert isinstance(model, PhysicallyInformedAE)

    assert metadata["input_shape"] == (32,)

    batch = next(iter(train_loader))
    assert batch["noisy_spectra"].ndim == 2

    val_batch = next(iter(val_loader))
    assert val_batch["clean_spectra"].ndim == 2


def test_noise_integer_ranges_generate_batches_without_error():
    """Integer-based noise ranges should still produce valid batches."""

    if torch is None:  # pragma: no cover - depends on test environment
        pytest.skip("torch is required for factory tests")

    from physae.factory import build_data_and_model

    integer_noise = {
        "std_add_range": (0.0, 0.0),
        "std_mult_range": (0.0, 0.0),
        "p_drift": 0.0,
        "p_fringes": 1.0,
        "n_fringes_range": (1, 2),
        "fringe_freq_range": (1.0, 1.0),
        "fringe_amp_range": (0.002, 0.002),
        "p_spikes": 1.0,
        "spikes_count_range": (1, 3),
        "spike_amp_range": (0.001, 0.001),
        "spike_width_range": (2, 3),
        "clip": (0.0, 1.5),
    }

    _, (train_loader, _), _ = build_data_and_model(
        **_small_build_kwargs(noise_train=integer_noise, noise_val=integer_noise)
    )

    dataset = train_loader.dataset
    manual_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batches = list(manual_loader)
    assert batches, "Expected at least one batch from the manual loader."
    for batch in batches:
        assert batch["noisy_spectra"].ndim == 2
        assert batch["params"].ndim == 2
