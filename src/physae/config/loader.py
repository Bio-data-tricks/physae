"""Configuration loading helpers."""
from __future__ import annotations

from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

from .schema import ExperimentConfig

T = TypeVar("T")


def _coerce_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively instantiate dataclasses from dictionaries."""

    if not is_dataclass(cls):  # type: ignore[arg-type]
        raise TypeError(f"{cls} is not a dataclass type")

    kwargs: Dict[str, Any] = {}
    for field_ in fields(cls):
        if field_.name not in data:
            if field_.default is not MISSING:
                kwargs[field_.name] = field_.default
            elif field_.default_factory is not MISSING:  # type: ignore[attr-defined]
                kwargs[field_.name] = field_.default_factory()  # type: ignore[misc]
            else:
                raise KeyError(f"Missing required config field: {field_.name}")
            continue
        value = data[field_.name]
        field_type = field_.type
        if isinstance(value, dict) and is_dataclass(field_type):  # type: ignore[arg-type]
            kwargs[field_.name] = _coerce_dataclass(field_type, value)
        elif field_type is Path:
            kwargs[field_.name] = Path(value)
        else:
            kwargs[field_.name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path, *, config_cls: Type[T] = ExperimentConfig) -> T:
    """Load YAML configuration into the provided dataclass type."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    return _coerce_dataclass(config_cls, raw)


def _convert(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_to_dict(config: Any) -> Dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("config_to_dict expects a dataclass instance")
    return _convert(config)


__all__ = ["load_config", "config_to_dict"]
