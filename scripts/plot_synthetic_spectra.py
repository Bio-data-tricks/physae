"""Generate illustrative synthetic spectra for the documentation (pure Python)."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Physical constants reused from ``physae.physics``.
C = 2.99792458e10
NA = 6.02214129e23
KB = 1.380649e-16
P0 = 1013.25
T0 = 273.15
L0 = 2.6867773e19
TREF = 296.0
PL = 15.12
M_CH4 = 16.04

POLY_FREQ_CH4 = [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]

_B = [
    complex(-0.0173, -0.0463),
    complex(-0.7399, 0.8395),
    complex(5.8406, 0.9536),
    complex(-5.5834, -11.2086),
]
_B = _B + [b.conjugate() for b in _B]
_C = [
    complex(2.2377, -1.626),
    complex(1.4652, -1.7896),
    complex(0.8393, -1.892),
    complex(0.2739, -1.9418),
]
_C = _C + [(-c).conjugate() for c in _C]

_TRANSITIONS_CSV = """\
6;1;3085.861015;1.013E-19;0.06;0.078;219.9411;0.73;-0.00712;0.0;0.0221;0.96;0.584;1.12
6;1;3085.832038;1.693E-19;0.0597;0.078;219.9451;0.73;-0.00712;0.0;0.0222;0.91;0.173;1.11
6;1;3085.893769;1.011E-19;0.0602;0.078;219.9366;0.73;-0.00711;0.0;0.0184;1.14;-0.516;1.37
6;1;3086.030985;1.659E-19;0.0595;0.078;219.9197;0.73;-0.00711;0.0;0.0193;1.17;-0.204;0.97
6;1;3086.071879;1.000E-19;0.0585;0.078;219.9149;0.73;-0.00703;0.0;0.0232;1.09;-0.0689;0.82
6;1;3086.085994;6.671E-20;0.055;0.078;219.9133;0.70;-0.00610;0.0;0.0300;0.54;0.00;0.0
"""


class Transitions:
    def __init__(self, csv_str: str):
        self.amplitude: List[float] = []
        self.center: List[float] = []
        self.gamma_air: List[float] = []
        self.gamma_self: List[float] = []
        self.n_air: List[float] = []
        self.shift_air: List[float] = []
        self.gDicke: List[float] = []
        self.nDicke: List[float] = []
        self.lmf: List[float] = []
        self.nlmf: List[float] = []
        for line in csv_str.strip().splitlines():
            tokens = [token.strip() for token in line.split(";")]
            while len(tokens) < 14:
                tokens.append("0")
            self.center.append(float(tokens[2]))
            self.amplitude.append(float(tokens[3]))
            self.gamma_air.append(float(tokens[4]))
            self.gamma_self.append(float(tokens[5]))
            self.n_air.append(float(tokens[7]))
            self.shift_air.append(float(tokens[8]))
            self.gDicke.append(float(tokens[10]))
            self.nDicke.append(float(tokens[11]))
            self.lmf.append(float(tokens[12]))
            self.nlmf.append(float(tokens[13]))


TRANSITIONS = Transitions(_TRANSITIONS_CSV)


def _wofz(z: complex) -> complex:
    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
    w = 0j
    for b, c in zip(_B, _C):
        w += b / (z - c)
    w *= 1j * inv_sqrt_pi
    if z.imag < 0:
        reflected = math.exp(-(z.real**2 - z.imag**2)) * math.cos(-2 * z.real * z.imag)
        reflected += 1j * math.exp(-(z.real**2 - z.imag**2)) * math.sin(-2 * z.real * z.imag)
        reflected = 2.0 * reflected - w.conjugate()
        return reflected
    return w


def _pine_profile(x: float, sigma_hwhm: float, gamma: float, gDicke: float) -> Tuple[float, float]:
    sigma = sigma_hwhm / math.sqrt(2 * math.log(2.0))
    xh = math.sqrt(math.log(2.0)) * x / sigma_hwhm
    yh = math.sqrt(math.log(2.0)) * gamma / sigma_hwhm
    zD = math.sqrt(math.log(2.0)) * gDicke / sigma_hwhm
    z = complex(xh, yh + zD)
    k = -_wofz(z)
    k_r, k_i = k.real, k.imag
    denom = (1 - zD * math.sqrt(math.pi) * k_r) ** 2 + (zD * math.sqrt(math.pi) * k_i) ** 2
    real = (k_r - zD * math.sqrt(math.pi) * (k_r**2 + k_i**2)) / denom
    imag = k_i / denom
    factor = math.sqrt(math.log(2.0) / math.pi) / sigma
    return real * factor, imag * factor


def _polyval(coeffs: Sequence[float], x: float) -> float:
    total = 0.0
    power = 1.0
    for coeff in coeffs:
        total += coeff * power
        power *= x
    return total


def simulate_spectrum(params: Dict[str, float], n_points: int = 800) -> Tuple[List[float], List[float]]:
    coeffs = [params["sig0"], params["dsig"], *POLY_FREQ_CH4]
    baseline_coeffs = [params["baseline0"], params["baseline1"], params["baseline2"]]
    v_grid: List[float] = []
    baseline: List[float] = []
    for idx in range(n_points):
        v_grid.append(_polyval(coeffs, idx))
        baseline.append(
            baseline_coeffs[0]
            + baseline_coeffs[1] * idx
            + baseline_coeffs[2] * (idx**2)
        )

    total_profile = [0.0 for _ in range(n_points)]
    T = params["T"]
    P = params["P"]
    mf = params["mf_CH4"]

    for i in range(len(TRANSITIONS.center)):
        center = TRANSITIONS.center[i] + params["shift_air"]
        amp = TRANSITIONS.amplitude[i]
        ga = TRANSITIONS.gamma_air[i]
        gs = TRANSITIONS.gamma_self[i]
        na = TRANSITIONS.n_air[i]
        gd = TRANSITIONS.gDicke[i]
        lmf = TRANSITIONS.lmf[i]
        nlmf = TRANSITIONS.nlmf[i]
        sigma_hwhm = (center / C) * math.sqrt(2 * NA * KB * T * math.log(2.0) / M_CH4)
        gamma = P / P0 * (TREF / T) ** na * (ga * (1 - mf) + gs * mf)
        flm = lmf * ((T / TREF) ** nlmf)
        for idx, v_val in enumerate(v_grid):
            x = v_val - center
            real_prof, imag_prof = _pine_profile(x, sigma_hwhm, gamma, gd)
            profile = real_prof + imag_prof * flm
            band = profile * amp * PL * 100 * mf * L0 * P / P0 * T0 / T
            total_profile[idx] += band

    spectra = []
    for idx in range(n_points):
        spectra.append(math.exp(-total_profile[idx]) * baseline[idx])
    return spectra, v_grid


def apply_noise(signal: Sequence[float], rng: random.Random, cfg: Dict[str, float]) -> List[float]:
    noisy = list(signal)
    std_add = cfg.get("std_add", 0.0)
    std_mult = cfg.get("std_mult", 0.0)
    drift_amp = cfg.get("drift_amp", 0.0)
    fringe_amp = cfg.get("fringe_amp", 0.0)
    fringe_freq = cfg.get("fringe_freq", 1.0)
    spike_amp = cfg.get("spike_amp", 0.0)
    spike_prob = cfg.get("spike_prob", 0.0)

    if std_add > 0:
        for i in range(len(noisy)):
            noisy[i] += rng.gauss(0.0, std_add)
    if std_mult > 0:
        for i in range(len(noisy)):
            noisy[i] *= 1.0 + rng.gauss(0.0, std_mult)
    if drift_amp > 0:
        kernel_width = max(3, int(cfg.get("drift_width", 50)))
        kernel = []
        centre = kernel_width // 2
        sigma = 0.2 * kernel_width
        total = 0.0
        for i in range(kernel_width):
            value = math.exp(-0.5 * ((i - centre) / sigma) ** 2)
            kernel.append(value)
            total += value
        kernel = [value / total for value in kernel]
        random_walk = [rng.gauss(0.0, 1.0) for _ in range(len(noisy))]
        drift = []
        for idx in range(len(noisy)):
            acc = 0.0
            for k_idx, weight in enumerate(kernel):
                src = idx + k_idx - centre
                if 0 <= src < len(random_walk):
                    acc += random_walk[src] * weight
            drift.append(acc)
        max_abs = max(abs(value) for value in drift) or 1.0
        for i in range(len(noisy)):
            noisy[i] += drift_amp * drift[i] / max_abs
    if fringe_amp > 0:
        for i in range(len(noisy)):
            angle = 2 * math.pi * fringe_freq * (i / max(len(noisy) - 1, 1))
            noisy[i] += fringe_amp * math.sin(angle + rng.random() * 2 * math.pi)
    if spike_amp > 0 and spike_prob > 0:
        for i in range(len(noisy)):
            if rng.random() < spike_prob:
                noisy[i] += rng.uniform(-spike_amp, spike_amp)
    return noisy


def _scale_points(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: float,
    height: float,
    offset_x: float,
    offset_y: float,
) -> str:
    scale_x = width / (x_max - x_min or 1.0)
    scale_y = height / (y_max - y_min or 1.0)
    coords: List[str] = []
    for x, y in zip(x_values, y_values):
        sx = offset_x + (x - x_min) * scale_x
        sy = offset_y + height - (y - y_min) * scale_y
        coords.append(f"{sx:.2f},{sy:.2f}")
    return " ".join(coords)


def render_svg(
    clean_series: Sequence[Sequence[float]],
    noisy_series: Sequence[Sequence[float]],
    x_axis: Sequence[Sequence[float]],
    titles: Sequence[str],
    output: Path,
) -> Path:
    width, height = 900, 700
    margin_left, margin_top = 70, 60
    plot_height = 160
    plot_width = width - 2 * margin_left

    global_y_min = min(min(series) for series in clean_series + noisy_series)
    global_y_max = max(max(series) for series in clean_series + noisy_series)

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;font-size:16px;}</style>",
        f"<rect width='{width}' height='{height}' fill='white' stroke='none'/>",
        "<text x='50%' y='30' text-anchor='middle' font-size='22'>Spectres synthétiques générés pour la documentation</text>",
    ]

    for idx, title in enumerate(titles):
        top = margin_top + idx * (plot_height + 40)
        svg_parts.append(
            f"<rect x='{margin_left}' y='{top}' width='{plot_width}' height='{plot_height}' fill='none' stroke='#cccccc' stroke-width='1'/>"
        )
        svg_parts.append(
            f"<text x='{margin_left}' y='{top - 10}' text-anchor='start'>{title}</text>"
        )
        coords_clean = _scale_points(
            x_axis[idx],
            clean_series[idx],
            x_axis[idx][0],
            x_axis[idx][-1],
            global_y_min,
            global_y_max,
            plot_width,
            plot_height,
            margin_left,
            top,
        )
        coords_noisy = _scale_points(
            x_axis[idx],
            noisy_series[idx],
            x_axis[idx][0],
            x_axis[idx][-1],
            global_y_min,
            global_y_max,
            plot_width,
            plot_height,
            margin_left,
            top,
        )
        svg_parts.append(
            f"<polyline fill='none' stroke='#1f77b4' stroke-width='1.8' points='{coords_clean}'/>"
        )
        svg_parts.append(
            f"<polyline fill='none' stroke='#ff7f0e' stroke-width='1.4' stroke-opacity='0.85' points='{coords_noisy}'/>"
        )
        svg_parts.append(
            f"<text x='{margin_left + plot_width - 10}' y='{top + plot_height - 10}' text-anchor='end' font-size='12'>Indice spectral</text>"
        )

    svg_parts.append(
        f"<text x='{margin_left}' y='{height - 30}' text-anchor='start'>Courbes bleues : spectres propres — Courbes orange : spectres bruités</text>"
    )
    svg_parts.append("</svg>")

    output.write_text("\n".join(svg_parts), encoding="utf-8")
    return output


def main(output: Path | None = None) -> Path:
    rng = random.Random(1234)
    scenarios = [
        (
            "Conditions nominales",
            {
                "sig0": 3085.44,
                "dsig": 0.00153,
                "mf_CH4": 8e-6,
                "baseline0": 1.0,
                "baseline1": -3.5e-4,
                "baseline2": -3.5e-8,
                "P": 500.0,
                "T": 308.15,
                "shift_air": 0.0,
            },
            {"std_add": 0.004, "std_mult": 0.004, "drift_amp": 0.01, "fringe_amp": 0.003, "fringe_freq": 4.0},
        ),
        (
            "Pression élevée",
            {
                "sig0": 3085.47,
                "dsig": 0.00154,
                "mf_CH4": 1.5e-5,
                "baseline0": 1.02,
                "baseline1": -3.2e-4,
                "baseline2": -3.2e-8,
                "P": 680.0,
                "T": 315.15,
                "shift_air": -0.002,
            },
            {"std_add": 0.003, "std_mult": 0.003, "drift_amp": 0.006, "fringe_amp": 0.002, "fringe_freq": 5.5},
        ),
        (
            "Bruit fort",
            {
                "sig0": 3085.42,
                "dsig": 0.00152,
                "mf_CH4": 2.5e-5,
                "baseline0": 0.98,
                "baseline1": -3.8e-4,
                "baseline2": -3.8e-8,
                "P": 420.0,
                "T": 303.15,
                "shift_air": 0.001,
            },
            {
                "std_add": 0.01,
                "std_mult": 0.01,
                "drift_amp": 0.02,
                "fringe_amp": 0.006,
                "fringe_freq": 8.0,
                "spike_amp": 0.03,
                "spike_prob": 0.01,
            },
        ),
    ]

    clean_series: List[List[float]] = []
    noisy_series: List[List[float]] = []
    x_axis: List[List[float]] = []

    for title, params, noise_cfg in scenarios:
        clean, x = simulate_spectrum(params, n_points=800)
        noisy = apply_noise(clean, rng, noise_cfg)
        clean_series.append(clean)
        noisy_series.append(noisy)
        x_axis.append(x)

    if output is None:
        output = Path(__file__).resolve().parents[1] / "docs" / "_static" / "generated_spectra.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    render_svg(clean_series, noisy_series, x_axis, [s[0] for s in scenarios], output)
    return output


if __name__ == "__main__":
    path = main()
    print(f"Figure sauvegardée dans {path}")
