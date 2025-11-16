# src/replm/steer/methods/probe.py
"""
Naive binary probe using logistic regression (no explicit y):
- Inputs: by_layer[layer], positive_idx, negative_idx
- Train a logistic regressor: z = w^T h + b on pos/neg
- Steer vector: w (optionally L2-normalized) scaled by alpha
- Edit: h' = h + alpha * w_hat

Minimal knobs:
- layer (required)
- normalize (default: False)
- alpha (default: 1.0)

Fixed internal training hyperparams:
- epochs = 200, lr = 1e-2, weight_decay = 0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..methods import register_method
from ..ops import AffineEdit
from .base import ActivationBatch, InputSpec, SteerMethod, SteerResult


def _flatten(X: Tensor) -> Tensor:
    if X.ndim == 2:  # (N, D)
        return X
    if X.ndim == 3:  # (N, T, D) -> (N*T, D)
        N, T, D = X.shape
        return X.reshape(N * T, D)
    raise ValueError("Activation tensor must be (N, D) or (N, T, D)")


@register_method("probe_layer")
@dataclass
class ProbeNaiveBinary(SteerMethod):
    layer: int
    normalize: bool = False
    alpha: float = 1.0
    eps: float = 1e-6

    # expose training hyperparams to Hydra
    lr: float = 1e-2
    epochs: int = 200
    weight_decay: float = 0.0

    # ---- protocol ----
    def requires(self) -> InputSpec:
        # no y; we rely on positive/negative indices only
        return InputSpec(need_pos=True, need_neg=True)

    # ---- helpers ----
    def _finalize_vec(self, w: Tensor) -> Tensor:
        if self.normalize:
            n = torch.linalg.norm(w) + self.eps
            w = w / n
        return (self.alpha * w).to(torch.float32)

    # ---- main ----
    def fit(self, data: ActivationBatch) -> SteerResult:
        if data.by_layer is None:
            raise ValueError("ActivationBatch.by_layer is required")
        if data.positive_idx is None or data.negative_idx is None:
            raise ValueError("ProbeNaiveBinary requires positive_idx and negative_idx in ActivationBatch.")

        # fetch layer activations
        if self.layer not in data.by_layer:
            avail = sorted(int(k) for k in data.by_layer.keys())
            raise KeyError(f"Requested layer {self.layer} not in by_layer. Available: {avail}")
        X = data.by_layer[self.layer]

        # slice pos/neg along batch axis (before flatten)
        try:
            Xp = X.index_select(0, data.positive_idx)
            Xn = X.index_select(0, data.negative_idx)
        except Exception as e:
            raise ValueError(f"Failed to index layer {self.layer} with positive/negative indices: {e}")

        Xp = _flatten(Xp)
        Xn = _flatten(Xn)
        if Xp.shape[0] == 0 or Xn.shape[0] == 0:
            raise ValueError("positive_idx/negative_idx cannot be empty after flattening")

        # build training matrix and labels
        Xf = torch.cat([Xp, Xn], dim=0).to(torch.float32)  # (M, D)
        yf = torch.cat(
            [
                torch.ones(Xp.shape[0], dtype=torch.float32, device=Xf.device),
                torch.zeros(Xn.shape[0], dtype=torch.float32, device=Xf.device),
            ],
            dim=0,
        )  # (M,)

        M, D = Xf.shape
        device = Xf.device

        model = nn.Linear(D, 1, bias=True).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        for _ in range(self.epochs):
            optim.zero_grad(set_to_none=True)
            logits = model(Xf).squeeze(-1)  # (M,)
            loss = criterion(logits, yf)
            loss.backward()
            optim.step()

        # extract steer vector
        model.eval()
        with torch.no_grad():
            logits = model(Xf).squeeze(-1)
            pred = (logits > 0).to(torch.float32)
            acc = (pred == yf).float().mean().item()
            train_loss = criterion(logits, yf).item()

        w: Tensor = model.weight.detach().reshape(-1)  # (D,)
        w = self._finalize_vec(w)
        edit = AffineEdit(layer=int(self.layer), add=w)

        meta: dict[str, Any] = {
            "selected_layer": int(self.layer),
            "normalize": bool(self.normalize),
            "alpha": float(self.alpha),
            "D": int(D),
            "num_examples": int(M),
            "train_acc": float(acc),
            "train_loss": float(train_loss),
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
        }
        return SteerResult(by_layer={int(self.layer): [edit]}, meta=meta)
