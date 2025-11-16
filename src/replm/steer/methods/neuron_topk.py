# src/replm/steer/methods/neuron_topk.py
"""
Neuron-based deactivation via global top-K ranking.

For each layer L and neuron j we compute the Pearson correlation between the
neuron activation h^L_j(x) and a scalar supervision signal (normalized entropy).
Neurons with the strongest absolute correlation are ranked globally, and the top
K are set to zero (mul=0) during steering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..methods import register_method
from ..ops import AffineEdit
from .base import ActivationBatch, InputSpec, SteerMethod, SteerResult


@register_method("neuron_topk")
@dataclass
class NeuronTopK(SteerMethod):
    topk: int = 32
    eps: float = 1e-6

    # ---------------- protocol ----------------
    def requires(self) -> InputSpec:
        return InputSpec(need_y=True)

    # ---------------- helpers ----------------
    @staticmethod
    def _sequence_level(X: Tensor) -> Tensor:
        """
        Ensure activations align with sequence-level supervision:
        (N, D) -> unchanged, (N, T, D) -> mean over tokens.
        """
        if X.ndim == 2:
            return X
        if X.ndim == 3:
            return X.mean(dim=1)
        raise ValueError("Activation tensor must be rank 2 or 3.")

    def _corr_scores(self, X: Tensor, y: Tensor) -> Tensor:
        """
        Compute absolute Pearson correlation between each neuron (column) and y.
        """
        X = X.to(torch.float64)
        y = y.to(torch.float64)
        N = X.shape[0]
        if y.shape[0] != N:
            raise ValueError("Supervision length does not match activation batch.")

        y_centered = y - y.mean()
        y_norm = y_centered / (y_centered.std(unbiased=False) + self.eps)

        X_centered = X - X.mean(dim=0, keepdim=True)
        X_norm = X_centered / (X.std(dim=0, unbiased=False, keepdim=True) + self.eps)

        corr = (X_norm.T @ y_norm) / (N - 1.0)
        return corr.abs()

    # ---------------- main --------------------
    def fit(self, data: ActivationBatch) -> SteerResult:
        if not data.by_layer:
            raise ValueError("ActivationBatch.by_layer is required.")
        if data.y is None:
            raise ValueError("NeuronTopK requires supervision signal in ActivationBatch.y.")

        scores_by_layer: dict[int, Tensor] = {}
        refs: list[tuple[int, int]] = []

        for layer_idx in sorted(data.by_layer.keys()):
            acts = self._sequence_level(data.by_layer[layer_idx])
            scores = self._corr_scores(acts, data.y)
            scores_by_layer[layer_idx] = scores
            refs.extend((layer_idx, int(j)) for j in range(scores.numel()))

        if not refs:
            return SteerResult(by_layer={}, meta={"topk": 0})

        all_scores = torch.cat([scores_by_layer[layer] for layer in sorted(scores_by_layer)])
        k = max(1, min(self.topk, all_scores.numel()))
        top_vals, top_idx = torch.topk(all_scores, k)

        selected = [refs[i] for i in top_idx.tolist()]
        bucket: dict[int, list[int]] = {}
        for layer_idx, dim in selected:
            bucket.setdefault(layer_idx, []).append(dim)

        edits: dict[int, list[AffineEdit]] = {}
        for layer_idx, dims_list in bucket.items():
            dims = torch.tensor(sorted(set(dims_list)), dtype=torch.long)
            mul = torch.zeros(dims.numel(), dtype=torch.float32)
            edits[layer_idx] = [AffineEdit(layer=layer_idx, dims=dims, mul=mul)]

        meta = {
            "topk": int(k),
            "selected_dims": {layer: len(dims) for layer, dims in bucket.items()},
            "scores": top_vals.tolist(),
        }
        return SteerResult(by_layer=edits, meta=meta)
