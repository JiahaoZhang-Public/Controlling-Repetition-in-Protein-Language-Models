# ==============================================
# file: replm/steer/methods/helpers.py
# ==============================================
from __future__ import annotations

import torch

from ..ops import AffineEdit

EPS = 1e-12


def dense_add(layer: int, vec: torch.Tensor, *, normalize: bool = False, alpha: float = 1.0,
              token_mask: torch.Tensor | None = None) -> AffineEdit:
    v = vec.to(torch.float32)
    if normalize:
        denom = torch.linalg.norm(v).clamp_min(EPS)
        v = v / denom
    return AffineEdit(layer=layer, add=alpha * v, token_mask=token_mask)


def sparse_add(layer: int, dims: torch.Tensor, values: torch.Tensor,
               token_mask: torch.Tensor | None = None) -> AffineEdit:
    dims = dims.to(torch.long).reshape(-1)
    vals = values.to(torch.float32).reshape(-1)
    assert dims.numel() == vals.numel(), "dims and values length mismatch"
    return AffineEdit(layer=layer, dims=dims, add=vals, token_mask=token_mask)


def sparse_mul(layer: int, dims: torch.Tensor, scales: torch.Tensor,
               token_mask: torch.Tensor | None = None) -> AffineEdit:
    dims = dims.to(torch.long).reshape(-1)
    sc = scales.to(torch.float32).reshape(-1)
    assert dims.numel() == sc.numel(), "dims and scales length mismatch"
    return AffineEdit(layer=layer, dims=dims, mul=sc, token_mask=token_mask)


def sparse_replace(layer: int, dims: torch.Tensor, values: torch.Tensor,
                    token_mask: torch.Tensor | None = None) -> AffineEdit:
    # replace via mul=0, add=v
    dims = dims.to(torch.long).reshape(-1)
    vals = values.to(torch.float32).reshape(-1)
    zeros = torch.zeros_like(vals)
    return AffineEdit(layer=layer, dims=dims, mul=zeros, add=vals, token_mask=token_mask)

