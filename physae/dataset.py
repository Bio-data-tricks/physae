"""Dataset and data generation utilities."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from . import config
from .baseline import lowess_max_value
from .noise import add_noise_variety
from .normalization import norm_param_value
from .physics import batch_physics_forward_multimol_vgrid


class SpectraDataset(Dataset):
    """Generate synthetic spectra paired with their physical parameters."""

    def __init__(
        self,
        n_samples: int,
        num_points: int,
        poly_freq_CH4,
        transitions_dict,
        sample_ranges: Optional[Dict[str, tuple[float, float]]] = None,
        *,
        strict_check: bool = True,
        with_noise: bool = True,
        noise_profile: Optional[Dict] = None,
        freeze_noise: bool = False,
    ) -> None:
        super().__init__()
        self.n_samples = int(n_samples)
        self.num_points = int(num_points)
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict
        self.sample_ranges = sample_ranges or config.NORM_PARAMS
        self.with_noise = bool(with_noise)
        self.noise_profile = dict(noise_profile or {})
        self.freeze_noise = bool(freeze_noise)
        self.epoch = 0

        if strict_check:
            for name in config.PARAMS:
                smin, smax = self.sample_ranges[name]
                nmin, nmax = config.NORM_PARAMS[name]
                if smin < nmin or smax > nmax:
                    raise ValueError(
                        f"sample_ranges['{name}']={self.sample_ranges[name]} "
                        f"not contained in NORM_PARAMS[{name}]={config.NORM_PARAMS[name]}."
                    )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_generator(self, idx: int) -> torch.Generator:
        base_seed = torch.initial_seed()
        generator = torch.Generator(device="cpu")
        if self.freeze_noise:
            seed = (123_456_789 + 97 * idx) % (2**63 - 1)
        else:
            seed = (base_seed + 1_000_003 * self.epoch + 97 * idx) % (2**63 - 1)
        generator.manual_seed(seed)
        return generator

    def __len__(self) -> int:  # type: ignore[override]
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        device, dtype = "cpu", torch.float32
        samples = [torch.empty(1, dtype=dtype).uniform_(*self.sample_ranges[k]) for k in config.PARAMS]
        sig0, dsig, mf_CH4, b0, b1, b2, P, T = samples
        baseline_coeffs = torch.cat([b0, b1, b2]).unsqueeze(0)
        mf_dict = {"CH4": mf_CH4}
        v_grid_idx = torch.arange(self.num_points, dtype=dtype, device=device)

        params_norm = torch.tensor(
            [norm_param_value(name, value.item()) for name, value in zip(config.PARAMS, samples)],
            dtype=torch.float32,
        )

        spectra_clean, _ = batch_physics_forward_multimol_vgrid(
            sig0,
            dsig,
            self.poly_freq_CH4,
            v_grid_idx,
            baseline_coeffs,
            self.transitions_dict,
            P,
            T,
            mf_dict,
            device=device,
        )
        spectra_clean = spectra_clean.to(torch.float32)

        if self.with_noise:
            generator = self._make_generator(idx)
            spectra_noisy = add_noise_variety(spectra_clean, generator=generator, **self.noise_profile)
        else:
            spectra_noisy = spectra_clean

        scale_noisy = lowess_max_value(spectra_noisy).unsqueeze(1).clamp_min(1e-8)
        noisy_spectra = spectra_noisy / scale_noisy

        max_clean = spectra_clean.amax(dim=1, keepdim=True).clamp_min(1e-8)
        clean_spectra = spectra_clean / max_clean

        return {
            "noisy_spectra": noisy_spectra[0].detach(),
            "clean_spectra": clean_spectra[0].detach(),
            "params": params_norm,
            "scale": scale_noisy.squeeze(1).to(torch.float32)[0],
        }
