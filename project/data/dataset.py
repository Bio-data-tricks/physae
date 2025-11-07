"""
PyTorch Dataset for synthetic spectral data generation with physics simulation.
"""
from typing import Optional, Sequence
import torch
from torch.utils.data import Dataset
from config.params import PARAMS, NORM_PARAMS
from data.normalization import norm_param_value
from data.noise import add_noise_variety
from physics.forward import batch_physics_forward_multimol_vgrid
from physics.tips import Tips2021QTpy, resolve_tipspy
from utils.lowess import lowess_value


class SpectraDataset(Dataset):
    """
    PyTorch Dataset class for synthetic spectral data generation with physics simulation.

    Args:
        n_samples: Number of samples in the dataset.
        num_points: Number of spectral points per sample.
        poly_freq_CH4: Polynomial frequency coefficients for CH4. Pass ``None``
            to use a purely linear grid defined by ``sig0`` and ``dsig``.
        transitions_dict: Dictionary of transitions per molecule.
        sample_ranges: Parameter sampling ranges (default: NORM_PARAMS).
        strict_check: Check if sample ranges are within NORM_PARAMS (default: True).
        with_noise: Add noise to spectra (default: True).
        noise_profile: Noise configuration parameters (default: None).
        freeze_parameters: Pre-sample parameter draws once so each index
            returns the same physical conditions at every epoch. When
            ``False`` (default), parameters are resampled on-the-fly like in
            ``physae.py``. The legacy keyword ``freeze_parameter_draws`` is
            also accepted for backwards compatibility.
        freeze_noise: Use fixed noise seed per sample (default: False).
        tipspy: Tips2021QTpy object for partition functions (default: None).
            When ``None``, the dataset attempts to locate a QTpy directory
            alongside the repository.
    """

    def __init__(
        self,
        n_samples,
        num_points,
        poly_freq_CH4: Sequence[float] | torch.Tensor | None,
        transitions_dict,
        sample_ranges: Optional[dict] = None,
        strict_check: bool = True,
        with_noise: bool = True,
        noise_profile: Optional[dict] = None,
        freeze_parameters: bool = False,
        freeze_noise: bool = False,
        tipspy: Tips2021QTpy | None = None,
        **legacy_kwargs,
    ):
        if "freeze_parameter_draws" in legacy_kwargs:
            legacy_value = bool(legacy_kwargs.pop("freeze_parameter_draws"))
            if freeze_parameters and not legacy_value:
                raise ValueError(
                    "freeze_parameters=True conflicts with freeze_parameter_draws=False"
                )
            freeze_parameters = legacy_value or freeze_parameters

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        self.n_samples = n_samples
        self.num_points = num_points
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict
        self.sample_ranges = sample_ranges if sample_ranges is not None else NORM_PARAMS
        self.with_noise = bool(with_noise)
        self.noise_profile = dict(noise_profile or {})
        self.freeze_parameters = bool(freeze_parameters)
        self.freeze_noise = bool(freeze_noise)
        needs_tipspy = any(len(v) for v in transitions_dict.values())
        self.tipspy = resolve_tipspy(
            tipspy,
            required=needs_tipspy,
            device="cpu",
        )
        self.epoch = 0
        self._frozen_params: dict[str, torch.Tensor] | None = None

        if strict_check:
            for k in PARAMS:
                smin, smax = self.sample_ranges[k]
                nmin, nmax = NORM_PARAMS[k]
                if smin < nmin or smax > nmax:
                    raise ValueError(
                        f"sample_ranges['{k}']={self.sample_ranges[k]} out of NORM_PARAMS[{k}]={NORM_PARAMS[k]}."
                    )

        if self.freeze_parameters:
            self.freeze_parameter_draws(True)

    def set_epoch(self, e: int):
        """Set current epoch for noise generation."""
        self.epoch = int(e)

    def freeze_parameter_draws(
        self,
        freeze: bool = True,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Toggle deterministic parameter draws across epochs.

        When ``freeze`` is ``True``, the dataset samples and stores the parameter
        vectors for every index so repeated epochs return identical physical
        conditions. When ``False``, the cached draws are cleared and future
        iterations resample parameters on-the-fly.

        Args:
            freeze: Enable (``True``) or disable (``False``) parameter freezing.
            generator: Optional random generator used when (re-)sampling frozen
                parameters. When ``None`` the global RNG state is used.
        """

        freeze = bool(freeze)
        self.freeze_parameters = freeze
        if not freeze:
            self._frozen_params = None
            return

        frozen: dict[str, torch.Tensor] = {}
        for k in PARAMS:
            lo, hi = self.sample_ranges[k]
            tens = torch.empty(self.n_samples, dtype=torch.float32)
            tens.uniform_(lo, hi, generator=generator)
            frozen[k] = tens
        self._frozen_params = frozen

    def refresh_frozen_parameters(self, generator: torch.Generator | None = None) -> None:
        """Resample cached parameters when freezing is enabled."""

        if not self.freeze_parameters:
            raise RuntimeError(
                "refresh_frozen_parameters() called while freeze_parameters is disabled."
            )
        self.freeze_parameter_draws(True, generator=generator)

    def _make_generator(self, idx: int) -> torch.Generator:
        """Create random generator for reproducible noise."""
        base = torch.initial_seed()
        g = torch.Generator(device='cpu')
        if self.freeze_noise:
            seed = (123456789 + 97 * idx) % (2 ** 63 - 1)
        else:
            seed = (base + 1_000_003 * self.epoch + 97 * idx) % (2 ** 63 - 1)
        g.manual_seed(seed)
        return g

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        device, dtype = 'cpu', torch.float32

        # Sample parameters
        if self.freeze_parameters:
            if self._frozen_params is None:
                self.freeze_parameter_draws(True)
            sampled = {
                k: self._frozen_params[k][idx].to(dtype).view(1)
                for k in PARAMS
            }
        else:
            sampled = {k: torch.empty(1, dtype=dtype).uniform_(*self.sample_ranges[k]) for k in PARAMS}
        sig0 = sampled['sig0']
        dsig = sampled['dsig']
        b0 = sampled['baseline0']
        b1 = sampled['baseline1']
        b2 = sampled['baseline2']
        P = sampled['P']
        T = sampled['T']

        baseline_coeffs = torch.cat([b0, b1, b2]).unsqueeze(0)
        v_grid_idx = torch.arange(self.num_points, dtype=dtype, device=device)

        # Mole fractions for physics
        mf_dict = {}
        if 'CH4' in self.transitions_dict:
            mf_dict['CH4'] = sampled['mf_CH4']
        if 'H2O' in self.transitions_dict:
            mf_dict['H2O'] = sampled['mf_H2O']

        # Normalized parameter vector
        params_norm = torch.tensor([norm_param_value(k, sampled[k].item()) for k in PARAMS], dtype=torch.float32)

        # Generate clean spectrum
        spectra_clean, _ = batch_physics_forward_multimol_vgrid(
            sig0, dsig, self.poly_freq_CH4, v_grid_idx,
            baseline_coeffs, self.transitions_dict, P, T, mf_dict,
            tipspy=self.tipspy, device=device
        )
        spectra_clean = spectra_clean.to(torch.float32)

        # Add noise
        if self.with_noise:
            g = self._make_generator(idx)
            spectra_noisy = add_noise_variety(spectra_clean, generator=g, **self.noise_profile)
        else:
            spectra_noisy = spectra_clean

        # Scale for input (noisy)
        scale_noisy = lowess_value(spectra_noisy, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)
        noisy_spectra = spectra_noisy / scale_noisy

        # Max on clean
        max_clean = lowess_value(spectra_clean, kind="start", win=30).unsqueeze(1).clamp_min(1e-8)
        clean_spectra = spectra_clean / max_clean

        return {
            'noisy_spectra': noisy_spectra[0].detach(),
            'clean_spectra': clean_spectra[0].detach(),
            'params': params_norm,
            'scale': scale_noisy.squeeze(1).to(torch.float32)[0]
        }
