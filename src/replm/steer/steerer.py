"""Context manager that applies affine steering edits via forward hooks."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Dict, List, Tuple

import torch
from torch import nn

from .ops import AffineEdit, LayerProgram, coalesce_layer

Tensor = torch.Tensor


class Steerer(AbstractContextManager):
    """
    Register affine edits on transformer blocks and clean them up automatically.

    Parameters
    ----------
    model:
        Module that exposes a `ModuleList` of residual blocks (identified via
        `layer_attr_path`, defaulting to `model.transformer.blocks`).
    specs:
        Iterable of :class:`AffineEdit` objects. Multiple edits on the same
        layer are composed before being applied inside the forward hook.
    layer_attr_path:
        Tuple of attribute names describing how to reach the `ModuleList`.
    output_index:
        When a block returns a tuple/list we edit the element at this index.
    """

    def __init__(
        self,
        model: nn.Module,
        specs: Sequence[AffineEdit] | None = None,
        *,
        layer_attr_path: Tuple[str, str] = ("transformer", "blocks"),
        output_index: int = 0,
    ) -> None:
        self.model = model
        self.output_index = int(output_index)

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._active = False

        self._by_layer: Dict[int, List[AffineEdit]] = {}
        for edit in specs or ():
            self._by_layer.setdefault(int(edit.layer), []).append(edit)

        self._blocks = self._resolve_blocks(layer_attr_path)
        self._num_layers = len(self._blocks)

        bad_layers = [layer for layer in self._by_layer if layer < 0 or layer >= self._num_layers]
        if bad_layers:
            raise IndexError(
                f"Layer index out of range: {bad_layers}; module list has {self._num_layers} blocks."
            )

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> "Steerer":
        if self._active:
            return self

        for layer_idx, edits in sorted(self._by_layer.items()):
            program = coalesce_layer(edits)
            handle = self._blocks[layer_idx].register_forward_hook(self._make_hook(program))
            self._handles.append(handle)

        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        return False

    # ------------------------------------------------------------------ helpers
    def _resolve_blocks(self, layer_attr_path: Tuple[str, ...]) -> nn.ModuleList:
        obj: nn.Module = self.model
        for attr in layer_attr_path:
            if not hasattr(obj, attr):
                chain = ".".join(layer_attr_path)
                raise AttributeError(f"Model is missing attribute '{attr}' while resolving {chain}")
            obj = getattr(obj, attr)
        if not isinstance(obj, nn.ModuleList):
            chain = ".".join(layer_attr_path)
            raise TypeError(f"Resolved object at {chain} is not an nn.ModuleList.")
        return obj

    def _make_hook(self, program: LayerProgram):
        """Apply compiled affine updates in the order: dense mul -> dense add -> sparse mul -> sparse add."""

        def _apply_with_mask(original: Tensor, edited: Tensor, mask: Tensor | None) -> Tensor:
            if mask is None:
                return edited
            m = mask.to(dtype=original.dtype, device=original.device)
            if m.ndim == 1:
                m = m.view(1, -1, 1)
            elif m.ndim == 2:
                if 1 in m.shape:
                    m = m.view(1, -1, 1)
                else:
                    raise ValueError("token_mask must broadcast to (1, T, 1)")
            elif m.ndim == 3:
                pass
            else:
                raise ValueError("token_mask must have ndim in {1, 2, 3}")
            return original * (1 - m) + edited * m

        out_idx = self.output_index

        def hook(_: nn.Module, __, output):
            container = None
            if isinstance(output, (tuple, list)):
                if len(output) == 0:
                    return output
                container = output
                y = output[out_idx]
            else:
                y = output

            if not isinstance(y, torch.Tensor):
                return output

            squeeze_back = False
            if y.ndim == 2:
                y3 = y.unsqueeze(-2)
                squeeze_back = True
            elif y.ndim == 3:
                y3 = y
            else:
                return output

            # Dense path
            if program.dense_mul is not None:
                dm = program.dense_mul.to(dtype=y3.dtype, device=y3.device).view(1, 1, -1)
                y3 = _apply_with_mask(y3, y3 * dm, program.dense_mask)
            if program.dense_add is not None:
                da = program.dense_add.to(dtype=y3.dtype, device=y3.device).view(1, 1, -1)
                y3 = _apply_with_mask(y3, y3 + da, program.dense_mask)

            # Sparse path
            if program.dims is not None:
                idx = program.dims.to(device=y3.device)
                selected = y3.index_select(dim=-1, index=idx).contiguous()

                if program.sparse_mul is not None:
                    sm = program.sparse_mul.to(dtype=y3.dtype, device=y3.device).view(1, 1, -1)
                    selected = _apply_with_mask(selected, selected * sm, program.sparse_mask)

                if program.sparse_add is not None:
                    sa = program.sparse_add.to(dtype=y3.dtype, device=y3.device).view(1, 1, -1)
                    selected = _apply_with_mask(selected, selected + sa, program.sparse_mask)

                y3 = y3.clone()
                y3[..., idx] = selected

            y_final = y3.squeeze(-2) if squeeze_back else y3

            if isinstance(container, tuple):
                as_list = list(container)
                as_list[out_idx] = y_final
                return tuple(as_list)
            if isinstance(container, list):
                container[out_idx] = y_final
                return container
            return y_final

        return hook
