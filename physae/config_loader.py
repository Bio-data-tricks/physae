"""Utilities to load YAML based configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def _ensure_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    return candidate


def load_yaml_file(path: str | Path) -> Dict[str, Any]:
    """Load a YAML document and return a dictionary copy."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise TypeError(f"Configuration at {path} must define a mapping, got {type(data)!r}.")
    return dict(data)


def load_data_config(path: str | Path | None = None, *, name: str = "default") -> Dict[str, Any]:
    """Load the dataset/model configuration.

    Args:
        path: Explicit path to the YAML file or directory containing the configuration.
        name: Named configuration within ``configs/data`` to resolve when ``path`` points to a
            directory or is omitted.
    """

    base = CONFIG_ROOT / "data" / f"{name}.yaml"
    target = _ensure_path(path, base)
    if target.is_dir():
        target = target / f"{name}.yaml"
    return load_yaml_file(target)


def load_stage_config(stage: str, path: str | Path | None = None) -> Dict[str, Any]:
    """Load the training-stage configuration for ``stage``.

    Args:
        stage: Stage identifier (e.g. ``"A"`` or ``"B1"``). The lookup is case-insensitive.
        path: Explicit path to a YAML file or a directory containing the stage configuration.
    """

    stage_norm = stage.lower()
    filename = f"stage_{stage_norm}.yaml" if not stage_norm.startswith("stage_") else f"{stage_norm}.yaml"
    if stage_norm.startswith("stage_"):
        stage_key = stage_norm.split("stage_", 1)[-1]
    else:
        stage_key = stage_norm
    base = CONFIG_ROOT / "stages" / filename
    target = _ensure_path(path, base)
    if target.is_dir():
        target = target / filename
    config = load_yaml_file(target)
    config.setdefault("stage_name", stage_key.upper())
    return config


def merge_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Shallow merge ``base`` with ``overrides`` returning a new dictionary."""

    result: Dict[str, Any] = dict(base)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
    return result


def coerce_sequence(values: Iterable[Any] | None) -> list:
    """Return ``values`` as a concrete list, filtering out ``None`` entries."""

    if values is None:
        return []
    return [item for item in values if item is not None]
