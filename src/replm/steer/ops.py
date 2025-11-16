"""Affine edit primitives used by steerers and steering methods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class AffineEdit:
    """
    Per-layer affine edit operating on neuron dimensions.

    Updates follow `y := (y * mul) + add` (restricted to `dims` if specified).
    Either dense (no `dims`, tensors shaped `(D,)`) or sparse (`dims` indexes).
    When `token_mask` is provided the edit only applies on masked time steps.
    """

    layer: int
    dims: Tensor | None = None
    mul: Tensor | None = None
    add: Tensor | None = None
    token_mask: Tensor | None = None


@dataclass(frozen=True, slots=True)
class LayerProgram:
    """
    Execution plan compiled from potentially many `AffineEdit`s on a layer.

    Dense and sparse paths are tracked separately so we can chain operations in
    the canonical order: dense mul -> dense add -> sparse mul -> sparse add.
    """

    dense_mul: Tensor | None = None
    dense_add: Tensor | None = None
    dims: Tensor | None = None
    sparse_mul: Tensor | None = None
    sparse_add: Tensor | None = None
    dense_mask: Tensor | None = None
    sparse_mask: Tensor | None = None


def _broadcast_mask(mask: Tensor) -> Tensor:
    """Normalize token masks to `(1, T, 1)` for easy broadcasting."""

    if mask.ndim == 1:
        return mask.view(1, -1, 1)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 2:
        if 1 in mask.shape:
            return mask.view(1, -1, 1)
    raise ValueError("token_mask must be broadcastable to (1, T, 1)")


def coalesce_layer(edits: Sequence[AffineEdit]) -> LayerProgram:
    """
    Collapse several affine edits that target the same layer into one program.

    Dense mul/add tensors are multiplied/summed respectively. Sparse edits are
    combined per index before producing contiguous tensors for efficient gather.
    """

    dense_mul: Tensor | None = None
    dense_add: Tensor | None = None
    dense_mask: Tensor | None = None

    sparse_mul_map: dict[int, float] = {}
    sparse_add_map: dict[int, float] = {}
    sparse_mask: Tensor | None = None

    for edit in edits:
        if edit.dims is None:
            if edit.mul is not None:
                dense_mul = edit.mul if dense_mul is None else dense_mul * edit.mul
            if edit.add is not None:
                dense_add = edit.add if dense_add is None else dense_add + edit.add
            if edit.token_mask is not None:
                mask = _broadcast_mask(edit.token_mask)
                dense_mask = mask if dense_mask is None else dense_mask * mask
            continue

        dims = edit.dims.reshape(-1).tolist()
        mul_vals = edit.mul.reshape(-1).tolist() if edit.mul is not None else None
        add_vals = edit.add.reshape(-1).tolist() if edit.add is not None else None
        for i, dim in enumerate(dims):
            if mul_vals is not None:
                sparse_mul_map[dim] = sparse_mul_map.get(dim, 1.0) * float(mul_vals[i])
            if add_vals is not None:
                sparse_add_map[dim] = sparse_add_map.get(dim, 0.0) + float(add_vals[i])
        if edit.token_mask is not None:
            mask = _broadcast_mask(edit.token_mask)
            sparse_mask = mask if sparse_mask is None else sparse_mask * mask

    if sparse_mul_map or sparse_add_map:
        dims_sorted = sorted(set(sparse_mul_map) | set(sparse_add_map))
        dims_tensor = torch.as_tensor(dims_sorted, dtype=torch.long)
        sparse_mul = torch.ones(len(dims_sorted), dtype=torch.float32)
        sparse_add = torch.zeros(len(dims_sorted), dtype=torch.float32)
        for j, dim in enumerate(dims_sorted):
            if dim in sparse_mul_map:
                sparse_mul[j] = sparse_mul_map[dim]
            if dim in sparse_add_map:
                sparse_add[j] = sparse_add_map[dim]
    else:
        dims_tensor = sparse_mul = sparse_add = None

    return LayerProgram(
        dense_mul=dense_mul,
        dense_add=dense_add,
        dims=dims_tensor,
        sparse_mul=sparse_mul,
        sparse_add=sparse_add,
        dense_mask=dense_mask,
        sparse_mask=sparse_mask,
    )


__all__ = [
    "AffineEdit",
    "LayerProgram",
    "coalesce_layer",
]
