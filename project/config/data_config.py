"""Utilities to load dataset-related YAML configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

from utils.io import load_config
from config.params import PARAMS, NORM_PARAMS, LOG_SCALE_PARAMS


_TRANSITION_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "mid",
        "lid",
        "center",
        "amplitude",
        "gamma_air",
        "gamma_self",
        "e0",
        "n_air",
        "shift_air",
        "abundance",
        "gDicke",
        "nDicke",
        "lmf",
        "nlmf",
    }
)


def _as_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    return p


def load_parameter_ranges(path: str | Path, *, update_globals: bool = True) -> Dict[str, tuple[float, float]]:
    """Load parameter ranges from a YAML configuration file.

    The YAML file must contain a ``parameters`` mapping where each key is a
    parameter name defined in :data:`config.params.PARAMS` and provides
    ``min``/``max`` values. An optional ``log_scale`` flag updates
    :data:`config.params.LOG_SCALE_PARAMS`.

    Args:
        path: Path to the YAML file.
        update_globals: When ``True`` (default), ``config.params.NORM_PARAMS``
            and ``config.params.LOG_SCALE_PARAMS`` are updated in-place.

    Returns:
        Dictionary mapping parameter names to ``(min, max)`` tuples.
    """

    cfg = load_config(str(_as_path(path)))
    raw_params = cfg.get("parameters")
    if not isinstance(raw_params, Mapping):
        raise ValueError("YAML file must define a 'parameters' mapping")

    missing: set[str] = set(PARAMS) - set(raw_params)
    if missing:
        raise ValueError(f"Missing parameter entries in YAML: {sorted(missing)}")

    unknown = set(raw_params) - set(PARAMS)
    if unknown:
        raise ValueError(f"Unknown parameters in YAML: {sorted(unknown)}")

    ranges: Dict[str, tuple[float, float]] = {}
    new_log_scale: set[str] = set()

    for name in PARAMS:
        spec = raw_params[name]
        if not isinstance(spec, Mapping):
            raise ValueError(f"Parameter '{name}' configuration must be a mapping")
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Parameter '{name}' must define 'min' and 'max' values")
        min_val = float(spec["min"])
        max_val = float(spec["max"])
        if min_val >= max_val:
            raise ValueError(
                f"Parameter '{name}' has invalid range: min ({min_val}) must be < max ({max_val})"
            )
        ranges[name] = (min_val, max_val)
        if bool(spec.get("log_scale", False)):
            new_log_scale.add(name)

    if update_globals:
        NORM_PARAMS.clear()
        NORM_PARAMS.update(ranges)
        # Update log-scale set without mutating the object reference
        LOG_SCALE_PARAMS.clear()
        LOG_SCALE_PARAMS.update(new_log_scale)

    return ranges


def load_noise_profile(path: str | Path) -> Dict[str, float | Iterable[float] | int]:
    """Load noise augmentation parameters from YAML.

    The YAML file must expose a ``noise`` mapping containing keyword arguments
    compatible with :func:`data.noise.add_noise_variety`.
    """

    cfg = load_config(str(_as_path(path)))
    noise = cfg.get("noise")
    if not isinstance(noise, Mapping):
        raise ValueError("YAML file must define a 'noise' mapping")
    return dict(noise)


def load_transitions(path: str | Path) -> Dict[str, list[dict]]:
    """Load molecular transitions from YAML configuration.

    Args:
        path: Path to the YAML file describing molecular transitions.

    Returns:
        Dictionary mapping molecule identifiers (e.g. ``"CH4"``) to a list of
        transition dictionaries compatible with
        :func:`physics.transitions.transitions_to_tensors`.
    """

    cfg = load_config(str(_as_path(path)))
    raw_transitions = cfg.get("transitions")
    if not isinstance(raw_transitions, Mapping):
        raise ValueError("YAML file must define a 'transitions' mapping")

    transitions_dict: Dict[str, list[dict]] = {}
    for mol, entries in raw_transitions.items():
        if entries is None:
            transitions_dict[mol] = []
            continue
        if not isinstance(entries, list):
            raise ValueError(f"Transitions for molecule '{mol}' must be a list")
        parsed_entries: list[dict] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Transition #{idx} for molecule '{mol}' must be a mapping, got {type(entry).__name__}"
                )
            missing = _TRANSITION_REQUIRED_FIELDS - set(entry)
            if missing:
                raise ValueError(
                    f"Transition #{idx} for molecule '{mol}' is missing fields: {sorted(missing)}"
                )
            parsed_entries.append(
                {
                    "mid": int(entry["mid"]),
                    "lid": int(entry["lid"]),
                    "center": float(entry["center"]),
                    "amplitude": float(entry["amplitude"]),
                    "gamma_air": float(entry["gamma_air"]),
                    "gamma_self": float(entry["gamma_self"]),
                    "e0": float(entry["e0"]),
                    "n_air": float(entry["n_air"]),
                    "shift_air": float(entry["shift_air"]),
                    "abundance": float(entry["abundance"]),
                    "gDicke": float(entry["gDicke"]),
                    "nDicke": float(entry["nDicke"]),
                    "lmf": float(entry["lmf"]),
                    "nlmf": float(entry["nlmf"]),
                }
            )
        transitions_dict[mol] = parsed_entries
    return transitions_dict


__all__ = [
    "load_parameter_ranges",
    "load_noise_profile",
    "load_transitions",
]
