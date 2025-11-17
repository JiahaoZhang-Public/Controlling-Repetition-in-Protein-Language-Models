from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import ModelBackend   



_AA_SET = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")


def _torch_device(device_str: str) -> torch.device:
    try:
        return torch.device(device_str)
    except Exception:
        return torch.device("cpu")


def _ensure_pad_token(tokenizer) -> int:
    """
    GPT-2 家族常无 pad_token。若无则回退到 eos，并显式设置，以便构造 attention_mask。
    """
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            try:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            except Exception:
                pass
            pad = getattr(tokenizer, "pad_token_id", None) or 0
        else:
            tokenizer.pad_token = tokenizer.eos_token
            pad = tokenizer.eos_token_id
    return int(pad)


def _aa_only(s: str) -> str:
    return "".join(ch for ch in s if ch.upper() in _AA_SET)


def _apply_len_trunc(aa: str, max_len: int) -> str:
    """超长截断；短了不补。"""
    if max_len is None or max_len <= 0:
        return ""
    return aa[:max_len]


def _model_pos_limit(cfg) -> int:
    return (
        getattr(cfg, "n_positions", None)
        or getattr(cfg, "max_position_embeddings", None)
        or 1024
    )


def _clean_gen_kwargs(gen_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    将用户传入的 gen_cfg 直接视为 HF generate() 的 kwargs：
    - 值为 None -> 移除，走 HF 默认；
    - 不做自定义回退，避免覆盖 HF 默认；
    """
    cfg = dict(gen_cfg or {})
    for k in list(cfg.keys()):
        if cfg[k] is None:
            cfg.pop(k)
    return cfg


def _with_simple_defaults(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    只固定项目要求的默认：do_sample=True, top_k=50。
    其余保持 HF 默认（若用户已传则尊重用户）。
    """
    out = dict(gen_kwargs)
    out.setdefault("do_sample", True)
    out.setdefault("top_k", 50)
    return out


def _sanitize_gen_kwargs_for_hf(model, cfg_dict: dict) -> dict:
    """
    仅保留 HuggingFace .generate()/GenerationConfig 支持的键。
    """
    if not cfg_dict:
        return {}
    try:
        allowed = set(model.generation_config.to_dict().keys())
    except Exception:
        allowed = set(GenerationConfig().to_dict().keys())

    allowed |= {
        "max_length", "min_length", "max_new_tokens", "min_new_tokens",
        "do_sample", "temperature", "top_k", "top_p",
        "num_beams", "num_beam_groups", "penalty_alpha",
        "repetition_penalty", "length_penalty", "no_repeat_ngram_size",
        "early_stopping", "renormalize_logits", "remove_invalid_values",
        "eos_token_id", "pad_token_id", "bos_token_id",
    }
    return {k: v for k, v in cfg_dict.items() if k in allowed}


def _aa_to_tokens_estimate(aa_len: int, aa_per_token: float = 4.0) -> int:
    """
    将目标 AA 长度粗略折算为 token 数（默认 4 AA ≈ 1 token）。
    """
    if aa_per_token <= 0:
        aa_per_token = 4.0
    return max(1, int(round(aa_len / aa_per_token)))


def _last_indices_from_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    # attn_mask: (B, T) with 1 for valid tokens
    idx = attn_mask.long().sum(dim=1) - 1
    return idx.clamp_min(0)


def _mask_special_for_mean(
    attn_mask: torch.Tensor,
    input_ids: torch.Tensor,
    bos: Optional[int],
    eos: Optional[int],
    pad: Optional[int],
) -> torch.Tensor:
    mask = attn_mask.clone()
    if pad is not None:
        mask = mask * (input_ids != pad)
    if bos is not None:
        mask = mask * (input_ids != bos)
    if eos is not None:
        mask = mask * (input_ids != eos)
    return mask


# --------------------------- backend --------------------------- #

class ProtGPT2Backend(ModelBackend):
    """
    基于 HuggingFace ProtGPT2 的后端，实现新 base 的接口：

      - load()
      - tokenize()
      - detokenize()
      - _forward_hidden_batch()
      - generate_uncond()
      - generate_with_prefix()

    激活流走 ModelBackend.activations()：
      -> (N, L, D) 的 tensor / numpy
    """

    def __init__(self, **cfg: Any):
        """
        cfg 由 Hydra 注入，例如：
          models:
            name: protgpt2
            device: cuda:0
            activation:
              pooling: mean
            init:
              model_name_or_path: nferruz/ProtGPT2
              max_length: 2048
        """
        super().__init__(**cfg)

        init_cfg = getattr(self.cfg, "init", {}) if hasattr(self.cfg, "init") else cfg.get("init", {}) or {}

        self.model_name_or_path: str = init_cfg.get("model_name_or_path", "nferruz/ProtGPT2")
        self.use_fast_tokenizer: bool = bool(init_cfg.get("use_fast_tokenizer", True))
        self.max_length: int = int(init_cfg.get("max_length", 2048))
        # activations 返回类型，"torch" / "numpy"，默认 torch（方便后面再算别的）
        self.return_type: str = str(init_cfg.get("return_type", "torch")).lower()

        # special ids 在 load() 里设置
        self.pad_id: Optional[int] = None
        self.bos_id: Optional[int] = None
        self.eos_id: Optional[int] = None

    # ---------------------- lifecycle ---------------------- #

    def load(self) -> ProtGPT2Backend:
        dev = _torch_device(getattr(self.cfg, "device", "cpu"))

        tok = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=self.use_fast_tokenizer,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32,   # 稳定优先
        ).to(dev)

        # pad/eos 设置
        self.pad_id = _ensure_pad_token(tok)
        self.bos_id = getattr(tok, "bos_token_id", None)
        self.eos_id = getattr(tok, "eos_token_id", None)

        # 若新增 pad_token，可能需要 resize embedding
        try:
            if mdl.config.vocab_size != len(tok):
                mdl.resize_token_embeddings(len(tok))
        except Exception:
            pass

        # 可选：禁用 flash/mem-efficient SDP，避免边界不稳定
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

        self.tokenizer = tok
        self.model = mdl.eval()
        return self

    # ---------------------- tokenization ---------------------- #

    def tokenize(self, sequences: Sequence[str]) -> Dict[str, torch.Tensor]:
        """
        将序列 batch 编成 HF 期望的 batch dict：
          {
            "input_ids": (B, T),
            "attention_mask": (B, T),
          }
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call .load() before .tokenize().")

        dev = next(self.model.parameters()).device
        # 过滤 None，保持字符串
        texts = [s if isinstance(s, str) else "" for s in sequences]

        # 有效最大长度：min(配置, 模型 pos embedding 长度)
        model_max_len = _model_pos_limit(self.model.config)
        effective_max_len = int(min(self.max_length, model_max_len))

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=effective_max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(dev)
        if "attention_mask" in enc:
            attention_mask = enc["attention_mask"].to(dev)
        else:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None and self.pad_id is not None:
                pad_id = self.pad_id
            attention_mask = (input_ids != (pad_id if pad_id is not None else 0)).long().to(dev)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def detokenize(self, token_ids: Sequence[int] | Any) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Call .load() before .detokenize().")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return _aa_only(text)

    # ---------------------- hidden states for base.activations ---------------------- #

    def _forward_hidden_batch(
        self,
        token_batch: Dict[str, torch.Tensor],
        layers: Sequence[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """
        这个函数只负责：
          - 调 HF 模型拿 hidden_states (embedding + 每层)
          - 根据请求的 layers 抽取对应的 (B, T, D) tensors
          - 返回 (hidden_list, attention_mask)
        池化和 (N, L, D) 拼接都交给 ModelBackend.activations() 做。
        """
        if self.model is None:
            raise RuntimeError("Call .load() before activations().")

        input_ids = token_batch["input_ids"]
        attention_mask = token_batch.get("attention_mask", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs: Tuple[torch.Tensor, ...] = outputs.hidden_states  # (embeddings, layer1, layer2, ...)

        hidden_list: list[torch.Tensor] = []
        for lid in layers:
            lid_int = int(lid)
            # hs[0] 是 embedding，hs[i+1] 是第 i 层输出
            if lid_int < 0 or lid_int + 1 >= len(hs):
                raise IndexError(f"Requested layer {lid_int} out of range (have {len(hs)-1} layers).")
            hidden_list.append(hs[lid_int + 1])

        return hidden_list, attention_mask

    # ---------------------- generation ---------------------- #

    @torch.no_grad()
    def generate_uncond(self, length: int, **gen_cfg: Any) -> str:
        """
        无条件生成（从 BOS 或最小起点）：
          - 若 gen_cfg 提供 length_min / length_max（或 aa_length_min / aa_length_max）：
              在 AA 长度上 Uniform[min, max] 采样目标长度；
          - 否则把入参 length 当成“目标 AA 长度”（未提供则默认 50）；
          - 按 aa_per_token（默认 4.0）折算 token 数；
          - 强约束 min/max_new_tokens & min/max_length；
          - 过滤成 AA 序列并按 target_aa 截断。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call .load() before .generate_uncond().")

        dev = next(self.model.parameters()).device
        tok = self.tokenizer

        # 最短起点
        if getattr(tok, "bos_token_id", None) is not None:
            input_ids = torch.tensor([[tok.bos_token_id]], device=dev, dtype=torch.long)
        else:
            input_ids = torch.tensor([[self.pad_id or 0]], device=dev, dtype=torch.long)

        attention_mask = (input_ids != (tok.pad_token_id if tok.pad_token_id is not None else -100)).long()

        kwargs = _with_simple_defaults(_clean_gen_kwargs(gen_cfg))

        aa_min = int(kwargs.pop("length_min", 0) or kwargs.pop("aa_length_min", 0) or 0)
        aa_max = int(kwargs.pop("length_max", 0) or kwargs.pop("aa_length_max", 0) or 0)
        aa_per_token = float(kwargs.pop("aa_per_token", 4.0))

        if aa_min > 0 and aa_max > 0 and aa_max >= aa_min:
            target_aa = random.randint(aa_min, aa_max)
        else:
            target_aa = int(length if length is not None else 50)

        tokens_new = _aa_to_tokens_estimate(target_aa, aa_per_token)
        curr = int(input_ids.shape[1])

        kwargs.setdefault("min_new_tokens", tokens_new)
        kwargs.setdefault("max_new_tokens", tokens_new)
        kwargs.setdefault("min_length", curr + tokens_new)
        kwargs.setdefault("max_length", curr + tokens_new)

        kwargs.setdefault("pad_token_id", self.pad_id)
        kwargs.setdefault("eos_token_id", self.eos_id)

        kwargs = _sanitize_gen_kwargs_for_hf(self.model, kwargs)

        def _sample_once() -> str:
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            aa = _aa_only(tok.decode(out_ids[0], skip_special_tokens=True))
            return _apply_len_trunc(aa, target_aa)

        out = _sample_once()
        if not out:
            try:
                out = _sample_once()
            except Exception:
                pass
        if not out:
            out = "A" * max(1, target_aa)
        return out

    @torch.no_grad()
    def generate_with_prefix(self, target_len: int, prefix: str, **gen_cfg: Any) -> str:
        """
        前缀生成（target_len 视为目标总 AA 长度）：
          - 同样支持 length_min/length_max → 在“目标总 AA 长度”上采样；
          - 折算成“目标总 token 长度”，减去 prefix token 数得到 new_tokens；
          - 用 setdefault 方式设置 min/max_new_tokens & min/max_length；
          - 最后按目标总 AA 截断。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call .load() before .generate_with_prefix().")

        dev = next(self.model.parameters()).device
        tok = self.tokenizer

        pref = _aa_only(prefix)
        enc = tok(pref, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(dev)

        if getattr(tok, "bos_token_id", None) is not None:
            bos = torch.tensor([[tok.bos_token_id]], device=dev, dtype=torch.long)
            input_ids = torch.cat([bos, input_ids], dim=1)

        attention_mask = (input_ids != (tok.pad_token_id if tok.pad_token_id is not None else -100)).long()

        kwargs = _with_simple_defaults(_clean_gen_kwargs(gen_cfg))

        aa_min = int(kwargs.pop("length_min", 0) or kwargs.pop("aa_length_min", 0) or 0)
        aa_max = int(kwargs.pop("length_max", 0) or kwargs.pop("aa_length_max", 0) or 0)
        aa_per_token = float(kwargs.pop("aa_per_token", 4.0))

        if aa_min > 0 and aa_max > 0 and aa_max >= aa_min:
            target_total_aa = random.randint(aa_min, aa_max)
        else:
            target_total_aa = int(target_len if target_len is not None else 50)

        target_total_tokens = _aa_to_tokens_estimate(target_total_aa, aa_per_token)
        prefix_tokens = int(input_ids.shape[1])
        new_tokens = max(1, target_total_tokens - prefix_tokens)

        curr = prefix_tokens
        kwargs.setdefault("min_new_tokens", new_tokens)
        kwargs.setdefault("max_new_tokens", new_tokens)
        kwargs.setdefault("min_length", curr + new_tokens)
        kwargs.setdefault("max_length", curr + new_tokens)

        kwargs.setdefault("pad_token_id", self.pad_id)
        kwargs.setdefault("eos_token_id", self.eos_id)

        kwargs = _sanitize_gen_kwargs_for_hf(self.model, kwargs)

        def _sample_once() -> str:
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            aa_full = _aa_only(tok.decode(out_ids[0], skip_special_tokens=True))
            return _apply_len_trunc(aa_full, target_total_aa)

        out = _sample_once()
        if not out:
            try:
                out = _sample_once()
            except Exception:
                pass
        if not out:
            out = "A" * max(1, target_total_aa)
        return out

    # ---------------------- utils ---------------------- #

    def _num_layers(self) -> int:
        """
        优先按 cfg.layer_attr_path 推断层数；不行则回退到常见 GPT-2 结构或 config.n_layer。
        """
        mdl = self.model
        if mdl is None:
            return 0

        # 1) 尝试 layer_attr_path
        try:
            path = getattr(self.cfg, "layer_attr_path", ("transformer", "h"))
            obj = mdl
            for name in path:
                obj = getattr(obj, name)
            if hasattr(obj, "__len__"):
                return len(obj)
        except Exception:
            pass

        # 2) 常见 GPT-2 结构
        try:
            if hasattr(mdl, "transformer") and hasattr(mdl.transformer, "h"):
                return len(mdl.transformer.h)
        except Exception:
            pass

        # 3) 其他结构回退：model.layers
        try:
            if hasattr(mdl, "model") and hasattr(mdl.model, "layers"):
                return len(mdl.model.layers)
        except Exception:
            pass

        # 4) 最后回退：config.n_layer
        try:
            return int(getattr(mdl.config, "n_layer", 0))
        except Exception:
            return 0
