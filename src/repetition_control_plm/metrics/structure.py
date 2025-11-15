"""Structure-based metrics powered by proxy folding models.

Each proxy exposes a uniform API for computing fold-confidence metrics
such as pLDDT and pTM. Aggregating these into final utility scores is
handled elsewhere; this module focuses solely on producing per-sequence
structure confidence outputs via interchangeable proxy implementations.
The default proxy targets ESM3 single-sequence inference, and additional
variants (ESMFold, AlphaFold, etc.) can register their own metric sets.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar
import torch

logger = logging.getLogger(__name__)

TProxy = TypeVar("TProxy", bound="StructureProxyModel")


# --------------------------------------------------------------------------- #
#                            Result containers                                #
# --------------------------------------------------------------------------- #


@dataclass
class StructureConfidenceResult:
    """Container for structure confidence metrics produced by a proxy model."""

    sequence: str
    length: int
    metrics: Dict[str, float]
    model_name: str
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        data = {
            "sequence": self.sequence,
            "length": self.length,
            "model_name": self.model_name,
            "metrics": dict(self.metrics),
        }
        if self.extras:
            data["extras"] = self.extras
        return data


# --------------------------------------------------------------------------- #
#                         Proxy model abstraction                             #
# --------------------------------------------------------------------------- #


class StructureProxyModel(ABC):
    """Abstract base class for structure-evaluation proxy models."""

    #: Registry identifier (e.g., ``"esm3"`` or ``"alphafold"``)
    name: str = "base"

    def __init__(self, **config: Any) -> None:
        self._config: Dict[str, Any] = dict(config)

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Parameters applied unless explicitly overridden."""
        return {}

    def evaluate(self, sequence: str, **overrides: Any) -> StructureConfidenceResult:
        """Run the proxy model on ``sequence`` and return structured outputs."""
        normalized = self._normalize_sequence(sequence)
        params = {**self.get_default_params(), **self._config, **overrides}
        metrics, extras = self._predict(normalized, **params)
        return StructureConfidenceResult(
            sequence=normalized,
            length=len(normalized),
            metrics=dict(metrics),
            model_name=self.name,
            extras=extras,
        )

    @abstractmethod
    def _predict(self, sequence: str, **params: Any) -> Tuple[Mapping[str, float], Dict[str, Any]]:
        """Subclass hook that returns ``(metrics, extras)``."""

    @staticmethod
    def _normalize_sequence(sequence: str) -> str:
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError("`sequence` must be a non-empty string.")
        return sequence.replace(" ", "").replace("\n", "").upper()


# --------------------------------------------------------------------------- #
#                                Registry                                     #
# --------------------------------------------------------------------------- #


_STRUCTURE_MODEL_REGISTRY: Dict[str, Type[StructureProxyModel]] = {}


def register_structure_model(cls: Type[TProxy]) -> Type[TProxy]:
    """Decorator to register proxy implementations."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError("Structure proxy classes must define `name`.")
    if name in _STRUCTURE_MODEL_REGISTRY:
        raise ValueError(f"Structure proxy '{name}' already registered.")
    _STRUCTURE_MODEL_REGISTRY[name] = cls
    return cls


def get_structure_model(name: str, **kwargs: Any) -> StructureProxyModel:
    """Instantiate a registered proxy model."""
    if name not in _STRUCTURE_MODEL_REGISTRY:
        raise KeyError(f"Unknown structure proxy '{name}'. Registered: {list(_STRUCTURE_MODEL_REGISTRY)}")
    return _STRUCTURE_MODEL_REGISTRY[name](**kwargs)


def available_structure_models() -> Sequence[str]:
    """List registered proxy names."""
    return tuple(sorted(_STRUCTURE_MODEL_REGISTRY))


# --------------------------------------------------------------------------- #
#                           ESM3 implementation                               #
# --------------------------------------------------------------------------- #


@register_structure_model
class Esm3StructureProxy(StructureProxyModel):
    """Default structure scorer using ESM3 single-sequence inference."""

    name = "esm3"

    _GENERATION_ARGS = (
        "num_steps",
        "temperature",
        "schedule",
        "strategy",
        "top_p",
        "temperature_annealing",
        "condition_on_coordinates_only",
        "only_compute_backbone_rmsd",
    )

    _DEFAULT_PARAMS: Dict[str, Any] = {
        "model_name": "esm3-open",
        "device": None,
        "num_steps": 8,
        "temperature": 1.0,
        "schedule": "cosine",
        "strategy": "random",
        "top_p": 1.0,
        "temperature_annealing": True,
        "condition_on_coordinates_only": True,
        "only_compute_backbone_rmsd": False,
    }

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return dict(cls._DEFAULT_PARAMS)

    def _predict(self, sequence: str, **params: Any) -> Tuple[Mapping[str, float], Dict[str, Any]]:
        model_name = params.pop("model_name", self._DEFAULT_PARAMS["model_name"])
        device = params.pop("device", None)
        generation_kwargs = {k: params.get(k, self._DEFAULT_PARAMS[k]) for k in self._GENERATION_ARGS}
        client = self._load_client(model_name=model_name, device=device)

        try:
            from esm.sdk.api import ESMProtein, GenerationConfig
        except ImportError as exc:  # pragma: no cover - dependency may not be installed
            raise RuntimeError("ESM3 dependencies are missing. Install `esm` to enable structure metrics.") from exc

        protein = ESMProtein(sequence=sequence)
        gen_config = GenerationConfig(track="structure", **generation_kwargs)

        with torch.no_grad():
            protein = client.generate(protein, gen_config)

        plddt = getattr(protein, "plddt", None)
        if plddt is None:
            raise RuntimeError("ESM3 returned no pLDDT; cannot compute scores.")
        if not torch.is_tensor(plddt):
            plddt = torch.tensor(plddt)
        plddt = plddt.float()
        if plddt.dim() == 2 and plddt.shape[0] == 1:
            plddt = plddt[0]
        if plddt.dim() != 1:
            raise ValueError(f"Unexpected pLDDT shape: {tuple(plddt.shape)}")
        if not (float(plddt.min()) >= 0.0 and float(plddt.max()) <= 1.0):
            raise ValueError("Expected pLDDT in [0,1] from ESM3.")

        plddt_mean_01 = float(plddt.mean().item())
        ptm_val = getattr(protein, "ptm", None)
        if ptm_val is not None and torch.is_tensor(ptm_val):
            ptm_val = float(ptm_val.detach().cpu().item())
        elif ptm_val is not None:
            ptm_val = float(ptm_val)

        metrics = {
            "plddt_mean_01": plddt_mean_01,
            "ptm": ptm_val,
        }
        extras = {
            "generation_config": generation_kwargs,
            "model_name": model_name,
        }
        return metrics, extras

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_client(model_name: str, device: Optional[str]) -> Any:
        try:
            from esm.models.esm3 import ESM3
        except ImportError as exc:  # pragma: no cover - dependency may not be installed
            raise RuntimeError("ESM3 dependencies are missing. Install `esm`.") from exc
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading ESM3 model '%s' on device '%s'.", model_name, resolved_device)
        client = ESM3.from_pretrained(model_name)
        return client.to(resolved_device)
