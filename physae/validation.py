"""Lightweight validation helpers for PhysAE configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from . import config


@dataclass
class ValidationReport:
    """Container storing warnings and errors detected during validation."""

    name: str
    errors: List[str]
    warnings: List[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "errors": list(self.errors), "warnings": list(self.warnings)}

    def __bool__(self) -> bool:  # pragma: no cover - convenience proxy
        return self.is_valid

    def __str__(self) -> str:
        status = "✅ OK" if self.is_valid else "⚠️ Erreurs"
        lines = [f"{self.name}: {status}"]
        if self.errors:
            lines.append("  Erreurs:")
            lines.extend(f"    • {msg}" for msg in self.errors)
        if self.warnings:
            lines.append("  Avertissements:")
            lines.extend(f"    • {msg}" for msg in self.warnings)
        return "\n".join(lines)


def _ensure_interval(value: Any, name: str, *, allow_equal: bool = False) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} doit être une séquence de longueur 2.")
    lo, hi = float(value[0]), float(value[1])
    if allow_equal:
        if lo > hi:
            raise ValueError(f"{name} doit vérifier min ≤ max (reçu {lo}, {hi}).")
    else:
        if lo >= hi:
            raise ValueError(f"{name} doit vérifier min < max (reçu {lo}, {hi}).")
    return lo, hi


def _normalise_ranges(ranges: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    result: Dict[str, Tuple[float, float]] = {}
    for key, value in ranges.items():
        result[key] = _ensure_interval(value, f"intervalle '{key}'")
    return result


def validate_data_config(cfg: Mapping[str, Any], *, name: str = "data") -> ValidationReport:
    """Validate the YAML data configuration without instantiating datasets."""

    errors: List[str] = []
    warnings: List[str] = []

    for field in ("n_points", "n_train", "n_val", "batch_size"):
        value = cfg.get(field)
        if value is None or int(value) <= 0:
            errors.append(f"{field} doit être strictement positif (reçu {value!r}).")

    try:
        train_base = _normalise_ranges(cfg.get("train_ranges_base", {}))
    except ValueError as exc:
        errors.append(str(exc))
        train_base = {}

    try:
        train_direct = _normalise_ranges(cfg.get("train_ranges", {})) if cfg.get("train_ranges") else {}
    except ValueError as exc:
        errors.append(str(exc))
        train_direct = {}

    missing = [param for param in config.PARAMS if param not in (train_direct or train_base)]
    if missing:
        errors.append(
            "Paramètres sans intervalle d'entraînement défini: " + ", ".join(sorted(missing))
        )

    try:
        val_ranges = _normalise_ranges(cfg.get("val_ranges", {}))
    except ValueError as exc:
        errors.append(str(exc))
        val_ranges = {}

    if val_ranges:
        reference_train = train_direct or train_base
        for name, (vlo, vhi) in val_ranges.items():
            if name not in reference_train:
                warnings.append(f"{name} défini pour la validation mais pas pour l'entraînement.")
                continue
            tlo, thi = reference_train[name]
            if not (tlo <= vlo and vhi <= thi):
                warnings.append(
                    f"{name}: intervalle validation {vlo:.3g}-{vhi:.3g} hors de l'entraînement {tlo:.3g}-{thi:.3g}."
                )

    noise_cfg = cfg.get("noise", {}) or {}
    for split in ("train", "val"):
        split_cfg = noise_cfg.get(split) or {}
        for key, value in split_cfg.items():
            if isinstance(value, (list, tuple)):
                try:
                    _ensure_interval(value, f"bruit {split}.{key}", allow_equal=True)
                except ValueError as exc:
                    warnings.append(str(exc))

    return ValidationReport(name=name, errors=errors, warnings=warnings)


def _validate_bool(cfg: Mapping[str, Any], key: str, errors: List[str]) -> None:
    if key in cfg and not isinstance(cfg[key], bool):
        errors.append(f"{key} doit être un booléen (reçu {cfg[key]!r}).")


def validate_stage_config(cfg: Mapping[str, Any], *, name: str | None = None) -> ValidationReport:
    """Validate the configuration of a training stage."""

    stage_name = name or str(cfg.get("stage_name", "?"))
    errors: List[str] = []
    warnings: List[str] = []

    for field in ("epochs", "base_lr", "refiner_lr", "refine_steps", "delta_scale"):
        if field not in cfg:
            errors.append(f"Champ obligatoire manquant: {field}.")
    for field in ("epochs", "refine_steps"):
        if field in cfg and int(cfg[field]) <= 0:
            errors.append(f"{field} doit être strictement positif (reçu {cfg[field]!r}).")
    for field in ("base_lr", "refiner_lr", "delta_scale"):
        if field in cfg and float(cfg[field]) <= 0:
            errors.append(f"{field} doit être strictement positif (reçu {cfg[field]!r}).")

    for key in ("train_base", "train_heads", "train_film", "train_refiner"):
        _validate_bool(cfg, key, errors)

    optuna_space = cfg.get("optuna", {}) or {}
    for param, spec in optuna_space.items():
        if "type" not in spec:
            warnings.append(f"{param}: type Optuna non précisé, 'float' sera supposé.")
            spec_type = "float"
        else:
            spec_type = str(spec["type"]).lower()
        if spec_type in {"float", "int"}:
            if "low" not in spec or "high" not in spec:
                errors.append(f"{param}: bornes 'low'/'high' requises pour un paramètre numérique.")
                continue
            low, high = float(spec["low"]), float(spec["high"])
            if low >= high:
                errors.append(f"{param}: borne low >= high ({low} ≥ {high}).")
        elif spec_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, Iterable) or not list(choices):
                errors.append(f"{param}: une liste non vide 'choices' est attendue pour un choix catégoriel.")
        else:
            errors.append(f"{param}: type Optuna inconnu '{spec_type}'.")

    return ValidationReport(name=f"stage {stage_name}", errors=errors, warnings=warnings)


__all__ = [
    "ValidationReport",
    "validate_data_config",
    "validate_stage_config",
]
