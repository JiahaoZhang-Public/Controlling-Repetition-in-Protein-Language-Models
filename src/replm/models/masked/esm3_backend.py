# src/replm/models/masked/esm3_backend.py
"""ESM3 backend wired into the new ModelBackend abstraction."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, cast

import torch
from torch import nn

from ...config import BackendConfig, coerce_config
from .. import register_model
from ..base import ModelBackend
from ..utils import (
    HookManager,
    batch_tokenize,
    build_attention_mask,
    find_transformer_blocks,
    get_special_ids,
    resolve_torch_dtype,
)
from .esm3_config import ESM3GenerationConfig, ESM3InitConfig


@register_model("esm3")
class ESM3Backend(ModelBackend):
    def __init__(
        self,
        *,
        backend_cfg: BackendConfig,
        init_cfg: ESM3InitConfig | Mapping[str, Any] | None = None,
        gen_cfg: ESM3GenerationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(backend_cfg=backend_cfg)
        self.init_cfg = coerce_config(init_cfg, ESM3InitConfig)
        self.gen_params = coerce_config(gen_cfg, ESM3GenerationConfig)

        self.model_name = self.init_cfg.model_name
        self.torch_autocast = self.init_cfg.torch_autocast
        self.include_final_norm = self.init_cfg.include_final_norm
        task_kind = self.cfg.task_type
        self.exclude_special_tokens = (
            self.init_cfg.exclude_special_tokens
            if self.init_cfg.exclude_special_tokens is not None
            else task_kind == "mlm"
        )

        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._blocks: list[nn.Module] = []
        self._final_norm: nn.Module | None = None
        self._special_ids: set[int] = set()
        self._pad_id = 0

        self._device = torch.device(self.cfg.device)
        self._target_dtype = resolve_torch_dtype(self.cfg.dtype)
        self._autocast_device = self._device.type if self._device.type in {"cuda", "mps"} else None
        self._autocast_enabled = bool(self.torch_autocast and self._autocast_device)
        self._autocast_dtype = torch.float16

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        from esm.models.esm3 import ESM3  # type: ignore

        model = ESM3.from_pretrained(self.model_name)
        if self._target_dtype is not None:
            model = model.to(device=self._device, dtype=self._target_dtype)
        else:
            model = model.to(device=self._device)
        try:
            model.eval()
        except Exception:
            pass

        try:
            tokenizer = model.tokenizers.sequence
        except Exception as exc:  # pragma: no cover - defensive
            raise AttributeError("ESM3 model does not expose a sequence tokenizer") from exc

        self.model = model
        self.tokenizer = tokenizer
        self._special_ids, self._pad_id = get_special_ids(tokenizer)
        blocks, maybe_norm = find_transformer_blocks(model)
        self._blocks = blocks
        self._final_norm = maybe_norm if self.include_final_norm else None

    # ------------------------------------------------------------------
    # Tokenization API required by ModelBackend
    # ------------------------------------------------------------------
    def tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Call load() before tokenize().")
        seqs = list(sequences)
        tokens = batch_tokenize(seqs, self.tokenizer, self._pad_id)
        mask = build_attention_mask(
            tokens,
            pad_id=self._pad_id,
            special_token_ids=self._special_ids,
            exclude_special_tokens=self.exclude_special_tokens,
        )
        tokens = cast(torch.LongTensor, tokens.to(self._device, non_blocking=True))
        mask = mask.to(self._device, non_blocking=True)
        return {"tokens": tokens, "mask": mask}

    def detokenize(self, token_ids: Sequence[int] | Any) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Call load() before detokenize().")
        ids = list(token_ids) if isinstance(token_ids, Sequence) else token_ids
        if hasattr(self.tokenizer, "decode"):
            return str(self.tokenizer.decode(ids))
        if hasattr(self.tokenizer, "convert_tokens_to_string"):
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            return str(self.tokenizer.convert_tokens_to_string(tokens))
        raise AttributeError("Tokenizer does not support detokenize operations")

    # ------------------------------------------------------------------
    # Hidden state capture
    # ------------------------------------------------------------------
    def _forward_hidden_batch(
        self,
        token_batch: dict[str, torch.Tensor],
        layers: Sequence[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Call load() before activations().")

        tokens = token_batch["tokens"]
        attn_mask = token_batch["mask"]
        cache: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def _hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                cache[layer_idx] = hidden

            return _hook

        with HookManager() as hm:
            for layer_idx in layers:
                module = self._module_for_layer(layer_idx)
                hm.add(module, _make_hook(layer_idx))
            self._run_model(tokens)

        hidden_list: list[torch.Tensor] = []
        for layer_idx in layers:
            if layer_idx not in cache:
                raise RuntimeError(f"No activations captured for layer {layer_idx}.")
            hidden_list.append(cache[layer_idx])

        return hidden_list, attn_mask

    def _module_for_layer(self, layer_idx: int) -> nn.Module:
        if layer_idx == -1:
            if self._final_norm is None:
                raise KeyError("Final norm was not captured; set include_final_norm=True.")
            return self._final_norm
        if layer_idx < 0 or layer_idx >= len(self._blocks):
            raise KeyError(f"Layer index {layer_idx} out of range for {len(self._blocks)} blocks.")
        return self._blocks[layer_idx]

    def _run_model(self, tokens: torch.Tensor) -> None:
        model = self.model
        if model is None:
            raise RuntimeError("Call load() before activations().")
        kwargs = {"sequence_tokens": tokens}
        if self._autocast_enabled and self._autocast_device is not None:
            with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                model(**kwargs)
        else:
            model(**kwargs)

    # ------------------------------------------------------------------
    # Generation helpers (unchanged API)
    # ------------------------------------------------------------------
    def generate_uncond(self, length: int, **gen_cfg: Any) -> str:
        if self.model is None:
            raise RuntimeError("Call load() before generate_uncond().")
        from esm.sdk.api import ESMProtein  # type: ignore

        L = int(length)
        cfg = self._build_gen_config(L, gen_cfg)
        protein = ESMProtein(sequence="_" * L)
        out = self._generate_with_fallbacks(protein, cfg)
        return out.sequence

    def generate_with_prefix(self, target_len: int, prefix: str, **gen_cfg: Any) -> str:
        if self.model is None:
            raise RuntimeError("Call load() before generate_with_prefix().")
        from esm.sdk.api import ESMProtein  # type: ignore

        L = int(target_len)
        cfg = self._build_gen_config(L, gen_cfg)
        pref = self._sanitize_prefix(prefix)
        if len(pref) > L:
            raise ValueError(f"Prefix length {len(pref)} is greater than target length {L}")
        masked = pref + "_" * (L - len(pref))
        protein = ESMProtein(sequence=masked)
        out = self._generate_with_fallbacks(protein, cfg)
        return out.sequence

    # ------------------------------------------------------------------
    # Generation internals
    # ------------------------------------------------------------------
    def _build_gen_config(self, length: int, gen_cfg: dict[str, Any] | None):
        from esm.sdk.api import GenerationConfig  # type: ignore

        cfg_kwargs = asdict(self.gen_params)
        cfg_kwargs.update(gen_cfg or {})

        cfg = GenerationConfig(track="sequence", num_steps=max(1, int(length)))
        for k, v in cfg_kwargs.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
        try:
            n = int(getattr(cfg, "num_steps", length))
        except Exception:
            n = length
        cfg.num_steps = max(1, min(int(length), n))
        return cfg

    def _attempt_generate(self, protein, cfg):
        out = self.model.generate(protein, cfg)
        if getattr(out, "sequence", None) is None:
            seq = getattr(out, "get", lambda *_: None)("sequence") if hasattr(out, "get") else None
            if seq is None:
                raise RuntimeError("ESM3 generation returned no sequence")
            return type("_Protein", (), {"sequence": seq})
        return out

    def _generate_with_fallbacks(self, protein, cfg):
        try:
            return self._attempt_generate(protein, cfg)
        except Exception:
            try:
                c2 = self._clone_cfg(cfg)
                setattr(c2, "strategy", "random")
                return self._attempt_generate(protein, c2)
            except Exception:
                c3 = self._clone_cfg(cfg)
                setattr(c3, "strategy", "random")
                try:
                    setattr(c3, "temperature_annealing", False)
                except Exception:
                    pass
                return self._attempt_generate(protein, c3)

    @staticmethod
    def _clone_cfg(cfg):
        try:
            from copy import deepcopy

            return deepcopy(cfg)
        except Exception:  # pragma: no cover
            return cfg

    @staticmethod
    def _sanitize_prefix(prefix: str) -> str:
        return (prefix or "").replace(" ", "").upper()
