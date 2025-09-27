import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from physae.factory import build_data_and_model


def test_noise_integer_ranges_generate_batches_without_error():
    train_loader, _, _ = build_data_and_model(
        n_points=32,
        n_train=4,
        n_val=2,
        batch_size=2,
        config_overrides={
            "noise": {
                "train": {
                    "n_fringes_range": [1, 2],
                    "spikes_count_range": [1, 3],
                }
            }
        },
    )

    noise_range = train_loader.dataset.noise_profile["n_fringes_range"]
    assert noise_range == (1, 2)
    assert all(isinstance(v, int) for v in noise_range)

    batch = next(iter(train_loader))
    spectra, _ = batch
    assert spectra.shape[0] > 0
