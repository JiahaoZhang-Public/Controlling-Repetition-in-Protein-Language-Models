# src/replm/models/autoregressive/hf_causal_lm.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ...config import BackendConfig, Pooling
from ...utils.constants import normalize_sequence
from .. import register_model
from ..base import ModelBackend


@register_model("hf_causal_lm")
class HFCausalLMBackend(ModelBackend):
    """
    HuggingFace causal LM backend wrapping AutoModelForCausalLM.

    Parameters
    ----------
    backend_cfg : BackendConfig
        General backend configuration (task_type should be "causal").
    model_name : str
        Huggingface model identifier (e.g., "nferruz/ProtGPT2 or hugohrban/progen2-small").
    tokenizer_kwargs / model_kwargs : dict
        Optional kwargs passed to AutoTokenizer / AutoModelForCausalLM.
    generation_kwargs : dict
        Default kwargs forwarded to `model.generate`.
    prompt_text : str
        Initial text fed to the tokenizer for unconditional generation.
    """

    def __init__(
        self,
        *,
        backend_cfg: BackendConfig,
        model_name: str,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        prompt_text: str | None = None,
    ) -> None:
        super().__init__(backend_cfg=backend_cfg)
        self.model_name = model_name
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.generation_kwargs = dict(generation_kwargs or {})
        self.prompt_text = prompt_text

        self.tokenizer = None
        self.model = None

    def load(self) -> HFCausalLMBackend:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("transformers is required for HFCausalLMBackend.") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.model_kwargs)
        model = model.to(self.cfg.device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        return self

    def tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Call load() before tokenize().")
        encoded = self.tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.cfg.device) for k, v in encoded.items()}

    def detokenize(self, token_ids: Sequence[int] | Any) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Call load() before detokenize().")
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _forward_hidden_batch(
        self,
        token_batch: dict[str, torch.Tensor],
        layers: Sequence[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if self.model is None:
            raise RuntimeError("Call load() before extracting activations.")
        outputs = self.model(
            **token_batch,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states  # tuple
        hidden_list: list[torch.Tensor] = []
        num_states = len(hidden_states)
        for layer_idx in layers:
            if layer_idx == -1:
                hidden_list.append(hidden_states[-1])
                continue
            if layer_idx < 0 or layer_idx >= num_states:
                raise IndexError(f"Layer index {layer_idx} out of range for {num_states} hidden states.")
            hidden_list.append(hidden_states[layer_idx])
        attn_mask = token_batch.get("attention_mask")
        return hidden_list, attn_mask

    def activations(
        self,
        sequences: Sequence[str],
        layers: Sequence[int],
        *,
        batch_size: int = 8,
        pooling: Pooling = 'last_nonpad',
        as_numpy: bool = False,
        requires_grad: bool | None = None,
    ):
        return super().activations(
            sequences,
            layers,
            batch_size=batch_size,
            pooling=pooling,
            as_numpy=as_numpy,
            requires_grad=requires_grad,
        )

    def _prepare_prompt(self, text: str | None = None) -> torch.Tensor:
        prompt = text if text is not None else (self.prompt_text or "")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")
        if not prompt:
            prompt = self.tokenizer.bos_token or self.tokenizer.eos_token or ""
        if not prompt:
            prompt = "<|endoftext|>"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.cfg.device)
        return input_ids

    def _translate_length(self, target_len: int) -> int:
        return int(target_len)

    def _generate(
        self,
        input_ids: torch.Tensor,
        target_len: int,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Call load() before generation.")
        gen_kwargs = dict(self.generation_kwargs)
        if extra_kwargs:
            gen_kwargs.update(extra_kwargs)
        gen_kwargs.setdefault("do_sample", True)
        effective_len = max(int(input_ids.shape[-1]), self._translate_length(target_len))
        gen_kwargs["max_length"] = effective_len
        gen_kwargs["min_length"] = effective_len
        with torch.no_grad():
            output = self.model.generate(input_ids=input_ids, **gen_kwargs)
        return normalize_sequence(self.detokenize(output[0]))

    def generate_uncond(self, length: int, **gen_cfg: Any) -> str:
        prompt_ids = self._prepare_prompt()
        return self._generate(prompt_ids, length, gen_cfg)

    def generate_with_prefix(self, target_len: int, prefix: str, **gen_cfg: Any) -> str:
        prompt_ids = self._prepare_prompt(prefix)
        if len(prefix or "") > target_len:
            raise ValueError("Prefix length exceeds target length.")
        return self._generate(prompt_ids, target_len, gen_cfg)


@register_model("protgpt2")
class ProtGPT2Backend(HFCausalLMBackend):
    """
    ProtGPT2 backend with length translation from residues to tokens.

    Since length is specified in amino acids but the model operates on tokens (≈4 residues),
    this backend divides the target length by `residues_per_token` before calling generate.
    """

    def __init__(
        self,
        *,
        backend_cfg: BackendConfig,
        residues_per_token: float = 4.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend_cfg=backend_cfg, **kwargs)
        self.residues_per_token = max(float(residues_per_token), 1e-6)

    def _translate_length(self, target_len: int) -> int:
        return max(1, int(round(target_len / self.residues_per_token)))


@register_model("progen2_small")
class Progen2SmallBackend(HFCausalLMBackend):
    """
    ProGen2-small backend that compensates for the extra <|endoftext|> token.
    The requested length L is achieved by generating `L + offset` tokens.
    """

    def __init__(
        self,
        *,
        backend_cfg: BackendConfig,
        length_offset: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend_cfg=backend_cfg, **kwargs)
        self.length_offset = int(length_offset)

    def _translate_length(self, target_len: int) -> int:
        return max(1, int(target_len) + self.length_offset)
