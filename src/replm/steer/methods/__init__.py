"""Registry and lazy loader for steering methods."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, TypeVar

REGISTRY: dict[str, type[Any]] = {}
T = TypeVar("T")


def register_method(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator that registers a steering method class under `name`."""

    def deco(cls: type[T]) -> type[T]:
        REGISTRY[name] = cls
        return cls

    return deco


_AUTOLOAD_MODULES = {
    "control": "replm.steer.methods.control",
    "contrastive_layer": "replm.steer.methods.contrastive_layer",
    "neuron_topk": "replm.steer.methods.neuron_topk",
    "probe_layer": "replm.steer.methods.probe",
}


def get_method_class(name: str):
    """Return the class registered under `name`, importing lazily if needed."""

    if name in REGISTRY:
        return REGISTRY[name]
    module_path = _AUTOLOAD_MODULES.get(name)
    if module_path:
        import_module(module_path)
    return REGISTRY.get(name)


__all__ = ["register_method", "get_method_class", "REGISTRY"]
