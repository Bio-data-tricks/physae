"""
HITRAN TIPS_2021 partition function reader (QTpy format).
"""
import pickle
from pathlib import Path
import numpy as np
import torch


def find_qtpy_dir(pref: str | Path) -> Path:
    """
    Locate a QTpy directory by trying common repository layouts.

    Args:
        pref: Preferred path to the QTpy directory (absolute or relative).

    Returns:
        Resolved path to the QTpy directory.

    Raises:
        FileNotFoundError: If no directory is found in the probed locations.
    """

    pref_path = Path(pref)
    if pref_path.exists() and pref_path.is_dir():
        return pref_path.resolve()

    here = Path.cwd()
    candidates = [
        here / "QTpy",
        here.parent / "QTpy",
        here / "project" / "QTpy",
        here.parent / "project" / "QTpy",
    ]

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()

    tried = [str(pref_path)] + [str(c) for c in candidates]
    raise FileNotFoundError(
        "QTpy directory not found (tried: " + ", ".join(tried) + ")."
    )


class Tips2021QTpy:
    """
    HITRAN TIPS_2021 partition function reader with linear temperature interpolation and caching.
    """

    def __init__(self, qtpy_dir: str | Path, device: str = 'cpu'):
        """
        Initialize TIPS reader.

        Args:
            qtpy_dir: Path to QTpy directory containing partition function files.
            device: PyTorch device for tensor operations.
        """
        self.base = Path(qtpy_dir).resolve()
        if not self.base.exists():
            raise FileNotFoundError(f"QTpy directory not found: {self.base}")
        self.device = device
        self.cache_dict = {}
        self.cache_table = {}
        self.cache_tmax = {}

    def _path_for(self, mid: int, iso: int) -> Path:
        """Get path for a specific molecule and isotopologue."""
        return self.base / f"{int(mid)}_{int(iso)}.QTpy"

    def _load_one(self, mid: int, iso: int):
        """Load partition function data for a specific molecule and isotopologue."""
        key = (int(mid), int(iso))
        if key in self.cache_dict:
            return
        p = self._path_for(mid, iso)
        if not p.exists():
            raise FileNotFoundError(f"QTpy file missing for (mol={mid}, iso={iso}): {p}")
        with open(p, "rb") as h:
            d = pickle.loads(h.read())
        dd = {int(k): float(v) for k, v in d.items()}
        tmax = int(max(dd.keys()))
        table = np.zeros(tmax, dtype=np.float64)
        for T in range(1, tmax + 1):
            if T in dd:
                table[T - 1] = dd[T]
            else:
                prev = max([k for k in dd.keys() if k < T], default=min(dd.keys()))
                nxt = min([k for k in dd.keys() if k > T], default=max(dd.keys()))
                if nxt == prev:
                    table[T - 1] = dd[prev]
                else:
                    a = (T - prev) / (nxt - prev)
                    table[T - 1] = dd[prev] + a * (dd[nxt] - dd[prev])
        self.cache_dict[key] = dd
        self.cache_table[key] = table
        self.cache_tmax[key] = tmax

    def q_scalar(self, mid: int, iso: int, T: float) -> float:
        """
        Get partition function for a single temperature (scalar).

        Args:
            mid: Molecule ID.
            iso: Isotopologue ID.
            T: Temperature (K).

        Returns:
            Partition function value.
        """
        self._load_one(mid, iso)
        key = (int(mid), int(iso))
        table = self.cache_table[key]
        tmax = self.cache_tmax[key]
        if T <= 1:
            return float(table[0])
        if T >= tmax:
            return float(table[-1])
        t0 = int(np.floor(T))
        t1 = t0 + 1
        f = (T - t0) / (t1 - t0)
        q1, q2 = table[t0 - 1], table[t1 - 1]
        return float(q1 + f * (q2 - q1))

    def q_torch(self, mid: int, iso: int, T: torch.Tensor) -> torch.Tensor:
        """
        Get partition function for temperature tensor.

        Args:
            mid: Molecule ID.
            iso: Isotopologue ID.
            T: Temperature tensor (K).

        Returns:
            Partition function tensor.
        """
        self._load_one(mid, iso)
        key = (int(mid), int(iso))
        table = self.cache_table[key]
        tmax = self.cache_tmax[key]
        Ts = T.detach().to(dtype=torch.float64, device=self.device)
        Ts = torch.clamp(Ts, 1.0, float(tmax))
        t0 = torch.floor(Ts)
        t1 = torch.clamp(t0 + 1.0, max=float(tmax))
        f = (Ts - t0) / torch.clamp(t1 - t0, min=1e-12)
        i0 = (t0.to(torch.int64) - 1).clamp(0, tmax - 1)
        i1 = (t1.to(torch.int64) - 1).clamp(0, tmax - 1)
        tab = torch.from_numpy(table).to(dtype=torch.float64, device=self.device)
        q1 = tab[i0]
        q2 = tab[i1]
        return q1 + f * (q2 - q1)


def resolve_tipspy(
    tipspy: "Tips2021QTpy" | None,
    *,
    device: str = "cpu",
    required: bool = True,
    qtpy_hint: str | Path = "QTpy",
) -> "Tips2021QTpy" | None:
    """Return a Tips2021QTpy instance, instantiating it if necessary.

    Args:
        tipspy: Existing Tips2021QTpy instance provided by the caller.
        device: Device where interpolated tensors should be allocated.
        required: Whether to raise when the QTpy directory cannot be located.
        qtpy_hint: Preferred location of the QTpy directory.

    Returns:
        A Tips2021QTpy instance, or ``None`` when ``required`` is False and no
        QTpy directory is found.
    """

    if tipspy is not None:
        return tipspy

    try:
        qtpy_dir = find_qtpy_dir(qtpy_hint)
    except FileNotFoundError:
        if not required:
            return None
        raise RuntimeError(
            "QTpy partition functions are required but no QTpy directory was "
            "found. Pass an explicit Tips2021QTpy instance or place the QTpy "
            "folder alongside the repository or inside project/QTpy."
        ) from None

    return Tips2021QTpy(qtpy_dir, device=device)
