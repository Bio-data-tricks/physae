"""Registries to enable pluggable model components."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable


class _ComponentRegistry:
    """Utility class maintaining a mapping of component builders."""

    def __init__(self, component_type: str) -> None:
        self.component_type = component_type
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register ``builder`` under ``name``.

        The builder must be a callable returning the concrete component instance. It is
        expected to accept keyword arguments and ignore those it does not use so that
        configuration dictionaries can be shared across different implementations.
        """

        def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
            key = (name or builder.__name__).lower()
            if key in self._builders:
                raise ValueError(
                    f"A {self.component_type} builder named '{key}' is already registered."
                )
            self._builders[key] = builder
            return builder

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._builders[name.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise KeyError(
                f"Unknown {self.component_type} '{name}'. Available options: {available}."
            ) from exc

    def build(self, name: str, **kwargs: Any) -> Any:
        builder = self.get(name)
        try:
            return builder(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Unable to initialise {self.component_type} '{name}' with arguments {kwargs}."
            ) from exc

    def list(self) -> Iterable[str]:
        return sorted(self._builders)


_ENCODER_REGISTRY = _ComponentRegistry("encoder")
_REFINER_REGISTRY = _ComponentRegistry("refiner")


def register_encoder(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering an encoder builder under ``name``."""

    return _ENCODER_REGISTRY.register(name)


def build_encoder(name: str, **kwargs: Any) -> Any:
    """Instantiate the encoder identified by ``name``."""

    return _ENCODER_REGISTRY.build(name, **kwargs)


def available_encoders() -> Iterable[str]:
    """Return the list of registered encoder identifiers."""

    return _ENCODER_REGISTRY.list()


def register_refiner(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a refiner builder under ``name``."""

    return _REFINER_REGISTRY.register(name)


def build_refiner(name: str, **kwargs: Any) -> Any:
    """Instantiate the refiner identified by ``name``."""

    return _REFINER_REGISTRY.build(name, **kwargs)


def available_refiners() -> Iterable[str]:
    """Return the list of registered refiner identifiers."""

    return _REFINER_REGISTRY.list()
