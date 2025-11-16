# src/replm/models/utils.py
"""Utility helpers shared across model backends."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class HookManager:
    """Context manager that keeps track of registered forward hooks."""

    def __init__(self):
        self.handles: list[Any] = []

    def __enter__(self):
        return self

    def add(self, module: nn.Module, fn) -> None:
        self.handles.append(module.register_forward_hook(fn))

    def add_many(self, modules: Iterable[nn.Module], fn_maker) -> None:
        for idx, module in enumerate(modules):
            self.add(module, fn_maker(idx))

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass


def resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    s = str(dtype).lower().replace("torch.", "")
    mapping = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype!r}")
    return mapping[s]


def tokenize_sequence(seq: str, tokenizer: Any, *, add_special_tokens: bool = True) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(seq, add_special_tokens=add_special_tokens))
    if hasattr(tokenizer, "tokenize"):
        return list(tokenizer.tokenize(seq, add_special_tokens=add_special_tokens))

    try:
        from esm.utils import encoding as esm_encoding  # type: ignore
    except Exception as e:  # pragma: no cover - dependency missing
        raise ImportError(
            "Tokenizer does not provide encode/tokenize and esm.utils.encoding is unavailable."
        ) from e
    return list(
        esm_encoding.tokenize_sequence(seq, tokenizer, add_special_tokens=add_special_tokens)
    )


def get_special_ids(tokenizer: Any) -> tuple[set[int], int]:
    pad_attr = getattr(tokenizer, "pad_token_id", None)
    try:
        pad_id = int(pad_attr) if pad_attr is not None else 0
    except Exception:
        pad_id = 0
    specials: set[int] = set()
    for attr in ("special_token_ids", "all_special_ids"):
        if hasattr(tokenizer, attr):
            try:
                specials = set(int(x) for x in getattr(tokenizer, attr))
                break
            except Exception:
                pass
    for name in ("cls_token_id", "eos_token_id", "mask_token_id", "sep_token_id"):
        if hasattr(tokenizer, name):
            try:
                specials.add(int(getattr(tokenizer, name)))
            except Exception:
                pass
    specials.add(pad_id)
    return specials, pad_id


def batch_tokenize(seqs: list[str], tokenizer: Any, pad_id: int) -> torch.LongTensor:
    ids = [torch.tensor(tokenize_sequence(seq, tokenizer), dtype=torch.long) for seq in seqs]
    from torch.nn.utils.rnn import pad_sequence

    return pad_sequence(ids, batch_first=True, padding_value=pad_id)


def build_attention_mask(
    tokens: torch.LongTensor,
    *,
    pad_id: int,
    special_token_ids: Iterable[int] | None = None,
    exclude_special_tokens: bool = False,
) -> torch.Tensor:
    mask = tokens != pad_id
    if exclude_special_tokens and special_token_ids:
        specials = torch.as_tensor(
            list(special_token_ids), dtype=torch.long, device=tokens.device
        )
        if specials.numel() > 0:
            try:
                special_mask = torch.isin(tokens, specials)
            except AttributeError:  # older torch without isin
                special_mask = torch.zeros_like(tokens, dtype=torch.bool)
                for sid in specials:
                    special_mask |= tokens.eq(int(sid))
            mask = mask & (~special_mask)
    return mask


def find_transformer_blocks(model: Any) -> tuple[list[nn.Module], nn.Module | None]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        blocks = list(model.transformer.blocks)
        return blocks, getattr(model.transformer, "norm", None)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        blocks = list(model.encoder.layers)
        return blocks, getattr(model.encoder, "norm", None)
    raise AttributeError(
        "Could not locate transformer blocks on model (expected .transformer.blocks or .encoder.layers)."
    )
