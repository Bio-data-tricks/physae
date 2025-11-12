"""Convenience imports and discovery helpers for training callbacks."""

from __future__ import annotations

from textwrap import dedent

from .epoch_sync import (
    AdvanceDistributedSamplerEpochAll,
    UpdateEpochInDataset,
    UpdateEpochInValDataset,
)
from .loss_curves import LossCurvePlotCallback
from .visualization import PT_PredVsExp_VisuCallback, StageAwarePlotCallback

_CALLBACK_EXPORTS = [
    "AdvanceDistributedSamplerEpochAll",
    "LossCurvePlotCallback",
    "PT_PredVsExp_VisuCallback",
    "StageAwarePlotCallback",
    "UpdateEpochInDataset",
    "UpdateEpochInValDataset",
]

__all__ = [*_CALLBACK_EXPORTS, "list_available_callbacks"]


def list_available_callbacks(include_docstrings: bool = True) -> dict[str, str | None]:
    """Return the public callbacks mapped to a short, single-line summary.

    Parameters
    ----------
    include_docstrings:
        When ``True`` (default), the mapping values contain the first line of the
        callback docstring.  Otherwise the mapping values are ``None``.

    Examples
    --------
    >>> for name, summary in list_available_callbacks().items():
    ...     print(f"{name}: {summary}")

    """

    callbacks: dict[str, str | None] = {}
    for name in _CALLBACK_EXPORTS:
        obj = globals().get(name)
        if obj is None:
            continue

        if include_docstrings:
            raw_doc = getattr(obj, "__doc__", "") or ""
            # Normalise multi-line docstrings to a compact summary.
            doc = dedent(raw_doc).strip().splitlines()
            callbacks[name] = doc[0] if doc else None
        else:
            callbacks[name] = None

    return callbacks
