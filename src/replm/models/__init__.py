# src/replm/models/__init__.py
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from .base import ModelBackend

REGISTRY: dict[str, type[ModelBackend]] = {}
TModel = TypeVar("TModel", bound=ModelBackend)


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Decorator to register a model backend class under a short name."""

    def deco(cls: type[TModel]) -> type[TModel]:
        REGISTRY[name] = cls
        return cls

    return deco


def _ensure_imported(name: str) -> None:
    if name in REGISTRY:
        return
    module_path = _AUTOLOAD_MODULES.get(name)
    if not module_path:
        return
    importlib.import_module(module_path)


def get_model_class(name: str) -> type[ModelBackend]:
    _ensure_imported(name)
    if name not in REGISTRY:
        raise KeyError(f"Unknown model backend '{name}'. Registered: {list(REGISTRY)}")
    return REGISTRY[name]


def build_model(name: str, **cfg: Any) -> ModelBackend:
    cls = get_model_class(name)
    return cls(**cfg)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(set(REGISTRY) | set(_AUTOLOAD_MODULES)))


# Map short names to modules that (when imported) will register themselves.
_AUTOLOAD_MODULES = {
    "esm3": "replm.models.masked.esm3_backend",
    "esm2": "replm.models.masked.esm2_backend",
    "hf_causal_lm": "replm.models.autoregressive.hf_causal_lm",
    "protgpt2": "replm.models.autoregressive.hf_causal_lm",
    "progen2_small": "replm.models.autoregressive.hf_causal_lm",
    "progen2_base": "replm.models.autoregressive.hf_causal_lm",
}


__all__ = [
    "REGISTRY",
    "available_models",
    "build_model",
    "get_model_class",
    "register_model",
]
