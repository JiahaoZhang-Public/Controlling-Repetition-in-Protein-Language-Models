# src/replm/steer/methods/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..ops import AffineEdit

Tensor = torch.Tensor


# ---- Input spec ----
@dataclass(frozen=True)
class InputSpec:
    need_pos: bool = False
    need_neg: bool = False
    need_all: bool = False
    need_y: bool = False
    need_targets: list[str] | None = None


# ---- Method protocol ----
class SteerMethod(Protocol):
    def requires(self) -> InputSpec: ...
    def fit(self, batch: ActivationBatch) -> SteerResult: ...


# ---- Data carriers for method IO ----
@dataclass(frozen=True)
class ActivationBatch:
    """A thin, library-agnostic wrapper for activations and labels."""

    by_layer: dict[int, Tensor]  # layer -> (N, T, D) or (N, D)
    y: Tensor | None = None  # task labels or scores
    positive_idx: Tensor | None = None  # indices
    negative_idx: Tensor | None = None


@dataclass(frozen=True)
class SteerResult:
    """
    Unified result type returned by all steer methods.

    Each layer maps to a list of AffineEdit objects that *already encode*
    the desired mul/add on either dense or sparse dims. Directional methods
    simply return dense add (optionally normalized) as AffineEdit.
    """

    by_layer: dict[int, list[AffineEdit]]
    meta: dict[str, Any]
