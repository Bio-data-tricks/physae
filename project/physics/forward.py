"""
Forward spectral model with HITRAN line-by-line calculations.
"""
import math
import torch
from .constants import C, R, P0, T0, TREF, L0, MOLECULE_PARAMS
from .profiles import pine_profile_torch_complex, apply_line_mixing_complex, polyval_torch
from .transitions import transitions_to_tensors
from .tips import Tips2021QTpy


def ST_hitran_with_qtpy(
    Sref,  # intensity at Tref (HITRAN, 296 K) — [1,L,1] or [B,L,1]
    nu0,  # transition frequency (cm^-1)   — [1,L,1] or [B,L,1]
    e0,  # lower state energy (cm^-1)      — [1,L,1] or [B,L,1]
    abundance,  # (kept for compat)       — [1,L,1] or [B,L,1]
    def_abundance,  # (kept for compat)   — [1,L,1] or [B,L,1]
    mid_arr,  # molecule ID per line       — [1,L,1] (long)
    iso_arr,  # isotopologue ID per line   — [1,L,1] (long)
    T_exp,  # experimental temperature     — [B,1,1] or [B] (K)
    mf=None,  # ignored (for compat)
    tipspy=None,  # Tips2021QTpy object (required)
    Tref: float = 296.0,  # HITRAN reference temperature
    device=None,
):
    """
    Compute temperature-dependent spectral line intensity using HITRAN + TIPS_2021 partition functions.

    Returns S(T) for each batch sample, using Q(T) from QTpy/TIPS_2021,
    calculated **per spectral line and per sample** (via tipspy.q_torch).
    Output shapes: [B, L, 1] (broadcasts correctly with the rest of physics).

    Args:
        Sref: Reference line intensity at Tref.
        nu0: Line center frequency.
        e0: Lower state energy.
        abundance: Isotopologue abundance.
        def_abundance: Default abundance.
        mid_arr: Molecule ID array.
        iso_arr: Isotopologue ID array.
        T_exp: Experimental temperature.
        mf: Mole fraction (ignored, for compatibility).
        tipspy: Tips2021QTpy object.
        Tref: Reference temperature (default: 296.0 K).
        device: PyTorch device.

    Returns:
        Temperature-dependent line intensity S(T) [B, L, 1].
    """
    if tipspy is None:
        raise RuntimeError("tipspy (QTpy/TIPS) is required for ST_hitran_with_qtpy().")

    # Consistent device/dtype
    if device is None:
        if torch.is_tensor(Sref):
            device = Sref.device
        elif torch.is_tensor(T_exp):
            device = T_exp.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = Sref.dtype if torch.is_tensor(Sref) else torch.float32

    def to_tensor(x, dtype_=dtype):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype_, non_blocking=True)
        return torch.as_tensor(x, device=device, dtype=dtype_)

    Sref = to_tensor(Sref)
    nu0 = to_tensor(nu0)
    e0 = to_tensor(e0)
    abundance = to_tensor(abundance)
    def_abundance = to_tensor(def_abundance)
    mid_arr = to_tensor(mid_arr, dtype_=torch.long)
    iso_arr = to_tensor(iso_arr, dtype_=torch.long)
    T_exp = to_tensor(T_exp)

    # Reshape T_exp -> [B,1,1] and extract Ts -> [B]
    if T_exp.ndim == 1:
        T_exp = T_exp.view(-1, 1, 1)
    elif T_exp.ndim == 2:
        T_exp = T_exp.view(T_exp.shape[0], 1, 1)
    B = T_exp.shape[0]

    # Deduce L (number of lines)
    if Sref.ndim == 3:
        L = Sref.shape[1]
    else:
        L = mid_arr.view(-1).numel()

    # Constant c2 in correct dtype/device
    c2 = torch.tensor(1.438776877, device=device, dtype=dtype)

    # === Q(T) and Q(Tref) vectorized per line & per sample ===
    mid_L = mid_arr.view(-1).to(torch.long)[:L]
    iso_L = iso_arr.view(-1).to(torch.long)[:L]
    key = mid_L * 100 + iso_L
    uniq = torch.unique(key)

    # Outputs [B, L, 1]
    Q_T = torch.empty((B, L, 1), device=device, dtype=dtype)
    Q_refT = torch.empty((B, L, 1), device=device, dtype=dtype)

    # Sample temperatures [B]
    Ts = T_exp.view(B)

    # For each group (molecule, isotopologue), call tipspy.q_torch once
    for k in uniq:
        k = int(k.item())
        mid_i = k // 100
        iso_i = k % 100
        cols = (key == k).nonzero(as_tuple=True)[0]

        qT_b = tipspy.q_torch(mid_i, iso_i, Ts).to(device=device, dtype=dtype)
        qRef_b = tipspy.q_torch(mid_i, iso_i, torch.full_like(Ts, float(Tref))).to(device=device, dtype=dtype)

        Q_T[:, cols, 0] = qT_b.view(B, 1).expand(B, cols.numel())
        Q_refT[:, cols, 0] = qRef_b.view(B, 1).expand(B, cols.numel())

    # === Boltzmann factors and ratio (broadcast [B,L,1]) ===
    Tref_t = torch.tensor(float(Tref), device=device, dtype=dtype)
    invT = 1.0 / T_exp
    invTref = 1.0 / Tref_t

    expo_fac = torch.exp(-c2 * e0 * (invT - invTref))
    num_fac = 1.0 - torch.exp(-c2 * nu0 * invT)
    den_fac = 1.0 - torch.exp(-c2 * nu0 * invTref)

    # Numerical safeguards
    eps = torch.tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
    den_fac = torch.where(den_fac == 0, eps, den_fac)
    Q_T = torch.where(Q_T == 0, eps, Q_T)

    # Intensity at T — shape [B, L, 1]
    S_T = Sref * (Q_refT / Q_T) * expo_fac * (num_fac / den_fac)
    return S_T


def batch_physics_forward_multimol_vgrid(
    sig0, dsig, poly_freq, v_grid_idx, baseline_coeffs,
    transitions_dict, P, T, mf_dict, *, tipspy: Tips2021QTpy, device='cpu', USE_LM: bool = True
):
    """
    Complete spectral forward model with HITRAN line-by-line calculations.

    Implements:
      - shift_air applied
      - Pine + Dicke effective narrowing
      - Line mixing with flm ∝ (Tref/T)^nlmf * (P/P0), absorption sign
      - S(T) HITRAN with Q(T) via QTpy
      - Column density: (P/P0)*(T0/T)*L0*PL*100
      - Transmission = exp(total_profile) (profile already signed)

    Args:
        sig0: Starting frequency [B].
        dsig: Frequency step [B].
        poly_freq: Polynomial frequency coefficients.
        v_grid_idx: Frequency grid indices [N].
        baseline_coeffs: Baseline polynomial coefficients [B, n_coeffs].
        transitions_dict: Dictionary of transitions per molecule.
        P: Pressure [B] (mbar).
        T: Temperature [B] (K).
        mf_dict: Mole fraction dictionary per molecule.
        tipspy: Tips2021QTpy object.
        device: PyTorch device.
        USE_LM: Use line mixing (default: True).

    Returns:
        Tuple of (transmission, v_grid_batch):
            - transmission: Spectral transmission [B, N].
            - v_grid_batch: Frequency grid [B, N].
    """
    B, N = sig0.shape[0], v_grid_idx.shape[0]
    v_grid_idx = v_grid_idx.to(device=device, dtype=torch.float64)
    sig0 = sig0.to(dtype=torch.float64, device=device).unsqueeze(1)
    dsig = dsig.to(dtype=torch.float64, device=device).unsqueeze(1)
    P = P.to(dtype=torch.float64, device=device).unsqueeze(1)
    T = T.to(dtype=torch.float64, device=device).unsqueeze(1)

    if baseline_coeffs.dim() == 1:
        baseline_coeffs = baseline_coeffs.unsqueeze(0)
    baseline_coeffs = baseline_coeffs.to(dtype=torch.float64, device=device)

    poly_freq_torch = torch.tensor(poly_freq, dtype=torch.float64, device=device).unsqueeze(0).expand(B, -1)
    coeffs = torch.cat([sig0, dsig, poly_freq_torch], dim=1)
    v_grid_batch = polyval_torch(coeffs, v_grid_idx)  # (B,N)

    total_profile = torch.zeros((B, N), device=device, dtype=torch.float64)

    P0_t = torch.tensor(P0, dtype=torch.float64, device=device)
    T0_t = torch.tensor(T0, dtype=torch.float64, device=device)
    TREF_t = torch.tensor(TREF, dtype=torch.float64, device=device)
    L0_t = torch.tensor(L0, dtype=torch.float64, device=device)
    C_t = torch.tensor(C, dtype=torch.float64, device=device)
    R_t = torch.tensor(R, dtype=torch.float64, device=device)

    for mol, trans in transitions_dict.items():
        (amp, center, ga, gs, na, sa, gd, nd, lmf, nlmf) = [
            t.to(dtype=torch.float64, device=device).view(1, -1, 1)
            for t in transitions_to_tensors(trans, device)
        ]
        e0 = torch.tensor([t['e0'] for t in trans], dtype=torch.float64, device=device).view(1, -1, 1)
        abn = torch.tensor([t['abundance'] for t in trans], dtype=torch.float64, device=device).view(1, -1, 1)
        def_abn = torch.ones_like(abn)
        mid_arr = torch.tensor([t['mid'] for t in trans], dtype=torch.int64, device=device).view(1, -1, 1)
        iso_arr = torch.tensor([t['lid'] for t in trans], dtype=torch.int64, device=device).view(1, -1, 1)

        mf = mf_dict[mol].to(dtype=torch.float64, device=device).view(B, 1, 1)
        Mmol = torch.tensor(MOLECULE_PARAMS[mol]['M'], dtype=torch.float64, device=device)
        PL = torch.tensor(MOLECULE_PARAMS[mol]['PL'], dtype=torch.float64, device=device)

        T_exp = T.view(B, 1, 1)
        P_exp = P.view(B, 1, 1)
        v_exp = v_grid_batch.view(B, 1, N)

        # Pressure shift
        x = v_exp - (center + sa * (P_exp / P0_t))

        # Doppler HWHM
        sigma_HWHM = (center / C_t) * torch.sqrt(2.0 * R_t * T_exp * math.log(2.0) / Mmol)

        # Collisional broadening + effective Dicke
        gamma = (P_exp / P0_t) * (TREF_t / T_exp) ** na * (ga * (1 - mf) + gs * mf)
        gN_eff = gd * (P_exp / P0_t) * (TREF_t / T_exp) ** nd

        real_prof, imag_prof = pine_profile_torch_complex(x, sigma_HWHM, gamma, gN_eff)
        if USE_LM:
            profile = apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T=T_exp, P=P_exp)
        else:
            profile = -real_prof  # absorption sign without LM

        # Exact S(T) (HITRAN + Q(T))
        S_T = ST_hitran_with_qtpy(
            Sref=amp, nu0=center, e0=e0, abundance=abn, def_abundance=def_abn,
            mid_arr=mid_arr, iso_arr=iso_arr, T_exp=T_exp, mf=mf, tipspy=tipspy,
            device=device,
        )

        # Column density (m -> cm via *100)
        col = (P_exp / P0_t) * (T0_t / T_exp) * L0_t * PL * 100.0 * mf

        band = profile * S_T * col  # (B, L, N) → sum over L
        total_profile += band.sum(dim=1)

    transmission = torch.exp(total_profile)  # profile already signed

    # Polynomial baseline on index
    x_bl = torch.arange(N, device=device, dtype=torch.float64)
    powers_bl = torch.arange(baseline_coeffs.shape[1], device=device, dtype=torch.float64)
    baseline = torch.sum(baseline_coeffs.unsqueeze(2) * x_bl.unsqueeze(0).pow(powers_bl.view(1, -1, 1)), dim=1)
    return transmission * baseline, v_grid_batch
