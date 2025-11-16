# src/replm/steer/methods/contrastive_layer.py
"""
Naive contrastive mean direction on a user-selected layer.

- User specifies `layer` (int). No automatic scoring/selection.
- Returns a dense-add `AffineEdit` on that layer only: direction = mu_pos - mu_neg
  (optionally variance-scaled and/or L2-normalized), then scaled by `alpha`.

Inputs (ActivationBatch):
- `by_layer`: dict[int, Tensor] with shapes (N, D) or (N, T, D)
- `positive_idx` / `negative_idx`: LongTensor index into the first dimension

Notes:
- This is a minimal baseline useful for ablations and manual sweeps over layers.
- For automatic best-layer picking, use `contrastive` (ContrastiveMeanDir).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..methods import register_method
from ..ops import AffineEdit
from .base import ActivationBatch, InputSpec, SteerMethod, SteerResult


@register_method("contrastive_layer")
@dataclass
class ContrastiveLayer(SteerMethod):
    layer: int = 0           # REQUIRED: the layer to edit
    normalize: bool = False   # L2 normalize the direction
    var_scale: bool = False  # divide by (var_pos + var_neg) per dim
    alpha: float = 1.0       # global strength multiplier
    eps: float = 1e-6

    # ---------------- protocol ----------------
    def requires(self) -> InputSpec:
        return InputSpec(need_pos=True, need_neg=True)

    # ---------------- helpers -----------------
    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            N, T, D = x.shape
            return x.reshape(N * T, D)
        raise ValueError("Activation tensor must be (N,D) or (N,T,D)")

    def _direction(self, Xp: Tensor, Xn: Tensor) -> Tensor:
        mu_p = Xp.mean(dim=0, dtype=torch.float64)
        mu_n = Xn.mean(dim=0, dtype=torch.float64)
        w = mu_p - mu_n  # (D,)
        if self.var_scale:
            v = Xp.var(dim=0, unbiased=False, dtype=torch.float64) + Xn.var(dim=0, unbiased=False, dtype=torch.float64) + self.eps # noqa: E501
            w = w / v
        if self.normalize:
            n = torch.linalg.norm(w) + self.eps
            w = w / n
        return (self.alpha * w).to(torch.float32)

    # ---------------- main --------------------
    def fit(self, data: ActivationBatch) -> SteerResult:
        if data.by_layer is None:
            raise ValueError("ActivationBatch.by_layer is required")
        if data.positive_idx is None or data.negative_idx is None:
            raise ValueError("ContrastiveLayer requires positive_idx and negative_idx in ActivationBatch.")

        # fetch layer activations
        if self.layer not in data.by_layer:
            avail = sorted(int(k) for k in data.by_layer.keys())
            raise KeyError(f"Requested layer {self.layer} not in by_layer. Available: {avail}")
        X = data.by_layer[self.layer]

        # slice pos/neg along batch axis
        try:
            Xp = X.index_select(0, data.positive_idx)
            Xn = X.index_select(0, data.negative_idx)
        except Exception as e:
            raise ValueError(f"Failed to index layer {self.layer} with positive/negative indices: {e}")

        Xp = self._flatten(Xp)
        Xn = self._flatten(Xn)
        if Xp.shape[0] == 0 or Xn.shape[0] == 0:
            raise ValueError("positive_idx/negative_idx cannot be empty after flattening")

        w = self._direction(Xp, Xn)  # (D,)
        edit = AffineEdit(layer=int(self.layer), add=w)

        meta: dict[str, Any] = {
            "selected_layer": int(self.layer),
            "normalize": bool(self.normalize),
            "var_scale": bool(self.var_scale),
            "alpha": float(self.alpha),
            "D": int(w.numel()),
        }
        return SteerResult(by_layer={int(self.layer): [edit]}, meta=meta)
