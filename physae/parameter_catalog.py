"""Utilities to inspect and document configurable parameters for PhysAE."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from .config_loader import load_data_config, load_stage_config, merge_dicts


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _normalise_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in mapping.items() if key != "optuna"}


@dataclass(frozen=True)
class ParameterInfo:
    """Metadata describing a configurable value."""

    key: str
    default: Any

    @property
    def value_type(self) -> str:
        value = self.default
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return "sequence"
        return type(value).__name__


TRAINER_PARAMETER_INFO: Dict[str, str] = {
    "accelerator": "Type d'accélérateur Lightning (cpu, gpu, auto, mps, ...)",
    "devices": "Nombre ou liste d'ID de GPU/CPU à utiliser (ex: 1, 'auto', [0,1])",
    "num_nodes": "Nombre de nœuds pour l'entraînement distribué",
    "precision": "Précision numérique (32, 16, 'bf16-mixed', etc.)",
    "strategy": "Stratégie Lightning pour le multi-GPU (ddp, deepspeed, ...)",
    "default_root_dir": "Répertoire où Lightning enregistre les logs et checkpoints",
    "gradient_clip_val": "Valeur de clipping du gradient",
    "accumulate_grad_batches": "Accumulation de gradient sur N mini-batchs",
    "limit_train_batches": "Sous-échantillonnage de la boucle d'entraînement",
    "limit_val_batches": "Sous-échantillonnage de la boucle de validation",
    "max_time": "Durée maximale d'entraînement (ex: '01:00:00')",
}


def list_data_parameters(*, config_path: str | None = None, config_name: str = "default") -> list[ParameterInfo]:
    """Return flattened information about data configuration parameters."""

    data_cfg = load_data_config(config_path, name=config_name)
    flat = _flatten(_normalise_mapping(data_cfg))
    return [ParameterInfo(key=key, default=value) for key, value in sorted(flat.items())]


def list_stage_parameters(stage: str, *, config_path: str | None = None) -> list[ParameterInfo]:
    """Return flattened information about stage configuration parameters."""

    stage_cfg = load_stage_config(stage, path=config_path)
    flat = _flatten(_normalise_mapping(stage_cfg))
    return [ParameterInfo(key=key, default=value) for key, value in sorted(flat.items())]


def describe_parameters(parameters: Iterable[ParameterInfo]) -> list[Dict[str, Any]]:
    """Convert parameter metadata to serialisable dictionaries."""

    return [
        {
            "key": param.key,
            "default": param.default,
            "type": param.value_type,
        }
        for param in parameters
    ]


def apply_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge overrides into a deep copy of ``base``."""

    result = copy.deepcopy(dict(base))
    if overrides:
        result = merge_dicts(result, overrides)
    return result


def set_by_path(mapping: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set ``value`` in ``mapping`` using a dotted key path."""

    parts = path.split(".")
    current: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], MutableMapping):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


__all__ = [
    "ParameterInfo",
    "TRAINER_PARAMETER_INFO",
    "list_data_parameters",
    "list_stage_parameters",
    "describe_parameters",
    "apply_overrides",
    "set_by_path",
]
