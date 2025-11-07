from __future__ import annotations

from physae.config import load_config
from physae.data import PhysaeDataModule, SpectraDataset


def test_spectra_dataset_shapes():
    config = load_config("configs/train.yaml")
    dataset = SpectraDataset(config.data)
    sample = dataset[0]
    assert sample["noisy"].shape[0] == config.data.num_points
    assert sample["clean"].shape[0] == config.data.num_points
    assert sample["params"].shape[0] == len(config.data.parameter_bounds)


def test_datamodule_train_batch():
    config = load_config("configs/train.yaml")
    module = PhysaeDataModule(config.data)
    module.setup("fit")
    batch = next(iter(module.train_dataloader()))
    assert batch["noisy"].shape[1] == config.data.num_points
    assert batch["clean"].shape == batch["noisy"].shape
    assert batch["params"].shape[1] == len(config.data.parameter_bounds)
