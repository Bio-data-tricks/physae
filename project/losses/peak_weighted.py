"""
Peak-weighted MSE loss for spectral data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakWeightedMSELoss(nn.Module):
    """
    MSE loss with peak detection weighting (emphasizes absorption peaks).

    Peak weighting via derivatives (100% PyTorch):
      - Gaussian smoothing BEFORE derivatives,
      - One-sided derivatives at borders + cosine attenuation,
      - Soft saliency map (0..1) smoothed (+ gamma),
      - Optional widening (gaussian / maxpool),
      - Conversion to weights then weighted MSE.
    """

    def __init__(
        self,
        peak_weight: float = 4.0,
        baseline_weight: float = 1.0,
        pre_smooth_sigma: float = 1.3,  # points
        salience_smooth_sigma: float = 1.8,  # points
        peak_kind: str = "min",  # "min" (absorption) or "max"
        curv_scale_k: float = 2.2,
        border_policy: str = "taper",  # "taper" or "zero"
        border_extra_margin: int = 2,
        weight_normalize: str = "mean",  # "mean" or "none"
        renorm_after_smoothing: bool = True,
        salience_gamma: float = 0.9,
        spread_kind: str | None = "gaussian",  # "gaussian", "maxpool", or None
        spread_sigma: float = 2.5,  # if gaussian (← somewhat wide by default)
        spread_kernel: int = 11,  # if maxpool (odd)
    ):
        """
        Args:
            peak_weight: Weight for peaks.
            baseline_weight: Weight for baseline.
            pre_smooth_sigma: Smoothing before derivatives.
            salience_smooth_sigma: Smoothing for saliency map.
            peak_kind: Type of peaks ("min" for absorption, "max" for emission).
            curv_scale_k: Curvature scaling factor.
            border_policy: Border handling ("taper" or "zero").
            border_extra_margin: Extra margin at borders.
            weight_normalize: Weight normalization ("mean" or "none").
            renorm_after_smoothing: Renormalize after smoothing.
            salience_gamma: Gamma for saliency map.
            spread_kind: Spreading method ("gaussian", "maxpool", or None).
            spread_sigma: Sigma for Gaussian spreading.
            spread_kernel: Kernel size for maxpool spreading.
        """
        super().__init__()
        self.peak_weight = float(peak_weight)
        self.baseline_weight = float(baseline_weight)
        self.pre_smooth_sigma = float(pre_smooth_sigma)
        self.salience_smooth_sigma = float(salience_smooth_sigma)
        self.peak_kind = str(peak_kind)
        self.curv_scale_k = float(curv_scale_k)
        self.border_policy = str(border_policy)
        self.border_extra_margin = int(border_extra_margin)
        self.weight_normalize = str(weight_normalize).lower()
        self.renorm_after_smoothing = bool(renorm_after_smoothing)
        self.salience_gamma = float(salience_gamma)
        self.spread_kind = (None if spread_kind is None else str(spread_kind).lower())
        self.spread_sigma = float(spread_sigma)
        self.spread_kernel = int(spread_kernel)

    @staticmethod
    def _gaussian_kernel_1d(sigma: float, device, dtype):
        """Create 1D Gaussian kernel."""
        if sigma <= 0:
            return torch.ones(1, 1, 1, device=device, dtype=dtype)
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        x = torch.arange(k, device=device, dtype=dtype) - (k // 2)
        g = torch.exp(-(x ** 2) / (2 * (sigma ** 2)))
        g = g / (g.sum() + 1e-12)
        return g.view(1, 1, -1)

    def _gaussian_smooth1d(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply 1D Gaussian smoothing."""
        if sigma <= 0:
            return x
        g = self._gaussian_kernel_1d(sigma, x.device, x.dtype)
        pad = g.shape[-1] // 2
        x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
        return F.conv1d(x_pad, g).squeeze(1)

    def _central_one_sided_diffs(self, x: torch.Tensor):
        """Compute first and second derivatives with special border handling."""
        B, N = x.shape
        d1 = torch.zeros_like(x)
        d2 = torch.zeros_like(x)
        if N >= 2:
            if N > 2:
                d1[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
            d1[:, 0] = (x[:, 1] - x[:, 0])
            d1[:, -1] = (x[:, -1] - x[:, -2])
        if N >= 3:
            d2[:, 1:-1] = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
            d2[:, 0] = x[:, 0] - 2.0 * x[:, 1] + x[:, 2]
            d2[:, -1] = x[:, -3] - 2.0 * x[:, -2] + x[:, -1]
        return d1, d2

    @staticmethod
    def _robust_scale_central(t: torch.Tensor, margin: int) -> torch.Tensor:
        """Compute robust scaling using MAD."""
        B, N = t.shape
        m = max(0, min(margin, (N - 1) // 2))
        tc = t[:, m:N - m] if (N - 2 * m >= 3) else t
        med = torch.median(tc, dim=-1, keepdim=True).values
        mad = torch.median(torch.abs(tc - med), dim=-1, keepdim=True).values + 1e-12
        return 1.4826 * mad

    @staticmethod
    def _border_taper_window(N: int, device, dtype, margin: int):
        """Create cosine-tapered border window."""
        if margin <= 0 or 2 * margin >= N:
            return torch.ones(1, N, device=device, dtype=dtype)
        t = torch.linspace(0, 1, steps=margin, device=device, dtype=dtype)
        ramp = 0.5 * (1 - torch.cos(torch.pi * t))
        w = torch.ones(N, device=device, dtype=dtype)
        w[:margin] = ramp
        w[-margin:] = torch.flip(ramp, dims=[0])
        return w.view(1, -1)

    def _detect(self, spectra: torch.Tensor):
        """Detect peaks and generate weight map."""
        B, N = spectra.shape
        device, dtype = spectra.device, spectra.dtype

        # margin = sum of pads used → avoid false peaks at borders
        m_pre = int(3 * max(self.pre_smooth_sigma, 0.0))
        m_sal = int(3 * max(self.salience_smooth_sigma, 0.0))
        m_spread = int(3 * max(self.spread_sigma if self.spread_kind == "gaussian" else 0.0, 0.0))
        border_margin = max(2, m_pre + m_sal + m_spread + self.border_extra_margin)

        x = self._gaussian_smooth1d(spectra, self.pre_smooth_sigma)
        d1, d2 = self._central_one_sided_diffs(x)
        curv = d2 if self.peak_kind.lower() == "min" else (-d2)

        sigma1 = self._robust_scale_central(d1, border_margin)
        gate = torch.exp(- (d1 / (3.0 * sigma1)) ** 2)

        sigma2 = self._robust_scale_central(curv, border_margin)
        s_raw = F.softplus(curv / (self.curv_scale_k * sigma2)) * gate

        w_border = self._border_taper_window(N, device, dtype, border_margin)
        if self.border_policy.lower() == "taper":
            s_raw = s_raw * w_border
        elif self.border_policy.lower() == "zero":
            s_raw = s_raw.clone()
            s_raw[:, :border_margin] = 0
            s_raw[:, -border_margin:] = 0

        s_raw = s_raw / (s_raw.amax(dim=-1, keepdim=True) + 1e-12)
        s_smooth = self._gaussian_smooth1d(s_raw, self.salience_smooth_sigma).clamp_(0, 1)
        if self.salience_gamma != 1.0:
            s_smooth = s_smooth ** self.salience_gamma
        if self.border_policy.lower() == "taper":
            s_smooth = s_smooth * w_border

        s_wide = s_smooth
        if self.spread_kind == "gaussian" and self.spread_sigma > 0:
            s_wide = self._gaussian_smooth1d(s_wide, self.spread_sigma)
        elif self.spread_kind == "maxpool":
            k = max(3, self.spread_kernel | 1)
            pad = k // 2
            s_wide = F.max_pool1d(s_wide.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)

        if self.border_policy.lower() == "taper":
            s_wide = s_wide * w_border
        if self.renorm_after_smoothing:
            s_wide = s_wide / (s_wide.amax(dim=-1, keepdim=True) + 1e-12)

        weights = self.baseline_weight + s_wide * (self.peak_weight - self.baseline_weight)
        if self.weight_normalize == "mean":
            weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-12)
        return weights, s_wide, border_margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_debug: bool = False):
        """
        Compute weighted MSE loss.

        Args:
            pred: [B, N] predicted spectra.
            target: [B, N] target spectra.
            return_debug: Return debug info.

        Returns:
            loss (and optionally debug dict).
        """
        assert pred.shape == target.shape and pred.ndim == 2
        weights, mask_soft, border_margin = self._detect(target)
        se = (pred - target) ** 2
        loss = (se * weights).sum() / (weights.sum() + 1e-12)
        if return_debug:
            return loss, {"weights": weights, "mask_soft": mask_soft, "border_margin": border_margin}
        return loss
