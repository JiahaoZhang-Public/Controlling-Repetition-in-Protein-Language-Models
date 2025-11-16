# src/replm/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

from ..config import BackendConfig, Pooling


class ModelBackend(ABC):
    """
    Minimal abstract backend:

      Public API:
        - load()
        - activations(sequences, layers) -> (N, L, D) sequence-level tensor
        - generate_uncond(length, **gen_cfg) -> str
        - generate_with_prefix(target_len, prefix, **gen_cfg) -> str

      Subclass responsibilities:
        - load(): create model and move to cfg.device/dtype
        - tokenize(): batchify input strings into tensors/structures your model expects
        - detokenize(): convert token ids back to a string
        - _forward_hidden_batch(): return per-token hidden states for requested layers
        - generate_uncond() / generate_with_prefix(): implement your own decoding
    """

    def __init__(self, backend_cfg: BackendConfig | None = None, **cfg: Any):
        if backend_cfg is not None and cfg:
            raise ValueError("Pass either backend_cfg or keyword overrides, not both.")
        if backend_cfg is None:
            self.cfg = BackendConfig(**cfg)
        else:
            self.cfg = backend_cfg
        self.model: Any = None
        self.tokenizer: Any = None

    # ----- lifecycle -----
    @abstractmethod
    def load(self) -> None:
        """Load model artifacts to cfg.device/dtype."""

    # ----- tokenization -----
    @abstractmethod
    def tokenize(self, sequences: Sequence[str]) -> Any:
        """Return a batch object (e.g., dict with input_ids, attention_mask)."""

    @abstractmethod
    def detokenize(self, token_ids: Sequence[int] | Any) -> str:
        """Convert token ids to a sequence string."""

    # ----- activations: sequence-level (N, L, D) -----
    def activations(
        self,
        sequences: Sequence[str],
        layers: Sequence[int],
        *,
        batch_size: int = 8,
        pooling: Pooling | None = None,
        as_numpy: bool = False,
        requires_grad: bool | None = None,
    ) -> np.ndarray | torch.Tensor:
        """
        Compute sequence-level activations for given layers.

        Returns:
            Tensor of shape (N, L, D) or numpy array if as_numpy=True.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for activations()")

        if not sequences:
            raise ValueError("sequences must be non-empty")
        if not layers:
            raise ValueError("layers must be non-empty")

        pool_mode = pooling or self.cfg.resolved_pooling()
        use_grad = bool(requires_grad) if requires_grad is not None else False
        grad_ctx = torch.enable_grad if use_grad else torch.no_grad

        batches: list[Tensor] = []  # each (B, L, D)
        with grad_ctx():
            for start in range(0, len(sequences), batch_size):
                end = min(len(sequences), start + batch_size)
                token_batch = self.tokenize(sequences[start:end])

                # Expect: hidden_list = list of (B, T, D); attn_mask = (B, T) or None
                hidden_list, attn_mask = self._forward_hidden_batch(token_batch, layers)
                if len(hidden_list) != len(layers):
                    raise RuntimeError(
                        f"_forward_hidden_batch returned {len(hidden_list)} tensors; expected {len(layers)}"
                    )

                pooled_layers: list[Tensor] = []
                for h_btd in hidden_list:
                    pooled_layers.append(self._pool_hidden(h_btd, attn_mask, mode=pool_mode))
                bld = torch.stack(pooled_layers, dim=1)  # (B, L, D)
                batches.append(bld)

        out = torch.cat(batches, dim=0)  # (N, L, D)
        return out.detach().cpu().numpy() if as_numpy else out

    @staticmethod
    def _pool_hidden(h_btd: Tensor, attn_mask: Tensor | None, *, mode: Pooling) -> Tensor:
        """
        Pool (B, T, D) -> (B, D) with padding-awareness if mask is provided.
        - mean: average over valid tokens
        - last_nonpad: take last valid token
        """
        if attn_mask is None:
            # No mask: treat all positions as valid
            if mode == "mean":
                return h_btd.mean(dim=1)
            elif mode == "last_nonpad":
                return h_btd[:, -1, :]
            raise ValueError(f"Unknown pooling mode: {mode}")

        if mode == "mean":
            masked = h_btd * attn_mask.unsqueeze(-1)  # (B,T,D)
            lengths = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            return masked.sum(dim=1) / lengths

        if mode == "last_nonpad":
            lengths = attn_mask.sum(dim=1).clamp(min=1).to(h_btd.device).to(torch.long)  # (B,)
            idx = lengths - 1
            b = torch.arange(h_btd.size(0), device=h_btd.device)
            return h_btd[b, idx, :]

        raise ValueError(f"Unknown pooling mode: {mode}")

    @abstractmethod
    def _forward_hidden_batch(
        self,
        token_batch: Any,
        layers: Sequence[int],
    ) -> tuple[list[Tensor], Tensor | None]:
        """
        Returns:
          - hidden_list: list(len=layers) of tensors (B, T, D), each for one requested layer
          - attn_mask: (B, T) with 1 for valid tokens and 0 for padding, or None
        """

    # ----- generation (framework-agnostic; to be implemented by subclasses) -----
    @abstractmethod
    def generate_uncond(self, length: int, **gen_cfg: Any) -> str:
        """Return an unconditioned sequence of `length`."""

    @abstractmethod
    def generate_with_prefix(self, target_len: int, prefix: str, **gen_cfg: Any) -> str:
        """Return a sequence up to `target_len`, conditioned on `prefix`."""
