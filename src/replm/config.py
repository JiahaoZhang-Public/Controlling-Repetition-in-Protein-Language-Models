from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

TaskType = Literal["mlm", "causal"]
Pooling = Literal["mean", "last_nonpad"]
T = TypeVar("T")

ScheduleType = Literal["cosine", "linear"]  # ESM3GenerationConfig
StrategyType = Literal["random", "entropy"]  # ESM3GenerationConfig


def _materialize_config(obj: Any) -> Any:
    """
    Convert nested Mapping/Sequence configs (e.g., OmegaConf DictConfig/ListConfig)
    into builtin python containers so dataclass constructors can consume them.
    """
    from collections.abc import Sequence

    if isinstance(obj, Mapping):
        return {key: _materialize_config(value) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_materialize_config(value) for value in obj]
    return obj


def coerce_config(value: T | Mapping[str, Any] | None, cls: type[T]) -> T:
    """
    Normalize user/Hydra-provided config payloads into dataclass instances.

    Accepts:
      - Already-initialized dataclass instances
      - dict-like structures (including OmegaConf's DictConfig) that match __init__ kwargs
      - None, which produces the dataclass defaults
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**_materialize_config(value))
    raise TypeError(f"Cannot coerce type {type(value)!r} into {cls.__name__}")


@dataclass
class BackendConfig:
    """
    Framework-agnostic configuration for model backends.
    """

    task_type: TaskType
    device: str = "cpu"
    dtype: Any | None = None
    default_pooling: Pooling | None = None

    def resolved_pooling(self) -> Pooling:
        if self.default_pooling is not None:
            return self.default_pooling
        return "mean" if self.task_type == "mlm" else "last_nonpad"


@dataclass
class ModelBuildConfig:
    """
    Generic container for instantiating a backend via registry.
    """

    name: str
    backend: BackendConfig
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ESM3InitConfig:
    model_name: str = "esm3-open"
    torch_autocast: bool = True
    include_final_norm: bool = False
    exclude_special_tokens: bool | None = None


@dataclass
class ESM3GenerationConfig:
    """
    Mirror of esm.sdk.api.GenerationConfig's high-level knobs.
    """

    track: str = "sequence"
    invalid_ids: list[int] = field(default_factory=list)
    schedule: ScheduleType = "cosine"
    strategy: StrategyType = "random"
    num_steps: int = 20
    temperature: float = 1.0
    temperature_annealing: bool = True
    top_p: float = 1.0
    condition_on_coordinates_only: bool = True
    only_compute_backbone_rmsd: bool = False


__all__ = [
    "BackendConfig",
    "ModelBuildConfig",
    "ESM3InitConfig",
    "ESM3GenerationConfig",
    "Pooling",
    "TaskType",
    "coerce_config",
]
