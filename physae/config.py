"""Configuration utilities for the PhysAE package.

This module centralises every global constant that is required by the
physically-informed auto-encoder.  It also exposes helper utilities used to
manipulate parameter ranges in a way that keeps the original implementation's
behaviour but in a more explicit and testable form.
"""

from __future__ import annotations

from typing import Dict, Mapping

PARAMS = [
    "sig0",
    "dsig",
    "mf_CH4",
    "baseline0",
    "baseline1",
    "baseline2",
    "P",
    "T",
]
"""Ordered list of physical parameters modelled by the network."""

PARAM_TO_IDX = {name: idx for idx, name in enumerate(PARAMS)}
"""Quick lookup table mapping a parameter name to its index in :data:`PARAMS`."""

LOG_SCALE_PARAMS = {"mf_CH4"}
"""Parameters that are normalised in the log10 domain."""

LOG_FLOOR = 1e-7
"""Lower bound used to avoid numerical issues when taking logarithms."""

NORM_PARAMS: Dict[str, tuple[float, float]] = {}
"""Mutable dictionary containing the normalisation range for every parameter."""


def set_norm_params(ranges: Mapping[str, tuple[float, float]]) -> None:
    """Populate :data:`NORM_PARAMS` with validated parameter ranges.

    Args:
        ranges: Mapping associating each parameter name with a ``(min, max)``
            tuple expressed in physical units.
    """

    NORM_PARAMS.clear()
    for name in PARAMS:
        if name not in ranges:
            raise KeyError(f"Missing normalisation range for parameter '{name}'.")
        lo, hi = ranges[name]
        NORM_PARAMS[name] = (float(lo), float(hi))


def get_norm_params() -> Dict[str, tuple[float, float]]:
    """Return a copy of the currently configured normalisation ranges."""

    return dict(NORM_PARAMS)


def expand_interval(a: float, b: float, factor: float) -> tuple[float, float]:
    """Expand an interval around its centre by ``factor``.

    The original code performed this inline.  Extracting it into a function
    clarifies the intent and eases unit testing.
    """

    centre = 0.5 * (a + b)
    half_span = 0.5 * (b - a) * float(factor)
    return float(centre - half_span), float(centre + half_span)


def map_ranges(
    base: Mapping[str, tuple[float, float]],
    fn,
    per_param: Mapping[str, float] | None = None,
) -> Dict[str, tuple[float, float]]:
    """Apply ``fn`` to each interval contained in ``base``.

    Args:
        base: Mapping of ``param -> (min, max)`` intervals that will be passed
            through ``fn``.
        fn: Callable with signature ``fn(min, max, factor)`` returning a new
            interval.  The function is invoked once per parameter.
        per_param: Optional mapping controlling the scaling factor applied to
            each interval.  The special key ``"_default"`` mirrors the behaviour
            of the legacy script.
    """

    result: Dict[str, tuple[float, float]] = {}
    per_param = dict(per_param or {})
    default = per_param.get("_default", 1.0)
    for name, (lo, hi) in base.items():
        factor = per_param.get(name, default)
        result[name] = fn(lo, hi, factor)
    return result


def assert_subset(
    child: Mapping[str, tuple[float, float]],
    parent: Mapping[str, tuple[float, float]],
    name_child: str = "child",
    name_parent: str = "parent",
) -> None:
    """Ensure that ``child`` intervals are contained inside ``parent`` ones."""

    offending = [
        name
        for name in child
        if not (parent[name][0] <= child[name][0] and child[name][1] <= parent[name][1])
    ]
    if offending:
        raise ValueError(f"{name_child} ⊄ {name_parent} for parameters: {offending}")
