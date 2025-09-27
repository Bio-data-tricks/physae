"""Tests for noise generation utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from physae.noise import add_noise_variety


def test_add_noise_variety_handles_small_signals():
    """Ensure the drift padding logic works for very short spectra."""

    generator = torch.Generator(device="cpu").manual_seed(1234)
    spectra = torch.zeros((2, 256), dtype=torch.float32)

    out = add_noise_variety(
        spectra,
        generator=generator,
        drift_sigma_range=(80.0, 90.0),
        p_drift=1.0,
        p_fringes=0.0,
        p_spikes=0.0,
    )

    assert out.shape == spectra.shape
