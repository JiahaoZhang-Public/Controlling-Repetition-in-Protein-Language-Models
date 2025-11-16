# src/replm/models/masked/esm3_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ScheduleType = Literal["cosine", "linear"]
StrategyType = Literal["random", "entropy"]


@dataclass
class ESM3InitConfig:
    """Initialization knobs specific to the ESM3 backend."""

    model_name: str = "esm3-open"
    torch_autocast: bool = True
    include_final_norm: bool = False
    exclude_special_tokens: bool | None = None


@dataclass
class ESM3GenerationConfig:
    """Mirror of esm.sdk.api.GenerationConfig's high-level knobs."""

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


__all__ = ["ESM3GenerationConfig", "ESM3InitConfig", "ScheduleType", "StrategyType"]
