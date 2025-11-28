# src/replm/models/masked/esm2_backend.py
"""ESM2 backend wired into the ModelBackend abstraction.

职责：
- 加载 fair-esm 的 ESM2 模型
- tokenize / detokenize
- 抽中间层 activations
- 用 masked-LM 方式做伪自回归生成（Gibbs-style）

注意：ESM2 原生是 MLM，这里的 generate_* 是基于反复 mask 并重采样每个位置的近似生成，
不是严格的左到右自回归，但会给出合理的氨基酸序列。
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ...config import BackendConfig, coerce_config
from .. import register_model
from ..base import ModelBackend
from ..utils import (
    HookManager,
    build_attention_mask,
    find_transformer_blocks,
    get_special_ids,
    resolve_torch_dtype,
)

# =============================
# Config
# =============================


class HFAlphabetShim:
    """Minimal shim to mimic fair-esm's alphabet API via a Hugging Face tokenizer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.padding_idx = self._resolve_id("pad_token_id", default=0)
        self.cls_idx = self._resolve_id("cls_token_id", default=self.padding_idx)
        self.eos_idx = self._resolve_id("eos_token_id", default=self.padding_idx)
        self.bos_idx = self._resolve_id("bos_token_id", default=self.cls_idx)
        mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("ESM2 tokenizer must define mask_token_id for MLM usage.")
        self.mask_idx = int(mask_id)

    def _resolve_id(self, attr: str, *, default: int) -> int:
        val = getattr(self.tokenizer, attr, None)
        if val is None:
            return int(default)
        return int(val)

    def get_batch_converter(self):
        tokenizer = self.tokenizer

        def _convert(batch: Sequence[tuple[str, str]]):
            labels = [label for label, _ in batch]
            seqs = [seq for _, seq in batch]
            encoded = tokenizer(
                seqs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            tokens = encoded["input_ids"].to(torch.long)
            return labels, seqs, tokens

        return _convert

    def get_idx(self, token: str) -> int:
        idx = self.tokenizer.convert_tokens_to_ids(token)
        if idx is None:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if not ids:
                raise KeyError(f"Unknown token '{token}' for tokenizer.")
            idx = ids[0]
        return int(idx)

    def get_tok(self, idx: int) -> str:
        tok = self.tokenizer.convert_ids_to_tokens(int(idx))
        if tok is None:
            raise KeyError(f"Unknown token id {idx} for tokenizer.")
        return tok


@dataclass
class ESM2InitConfig:
    """ESM2 初始化配置.

    Parameters
    ----------
    model_name:
        对应 `esm.pretrained` 里的函数名，比如 "esm2_t33_650M_UR50D".
    torch_autocast:
        是否用 torch.autocast 做推理（在 cuda/mps 上）。
    include_final_norm:
        是否把最后一层 norm 当成 layer=-1 暴露出去。
    exclude_special_tokens:
        build_attention_mask 时是否 mask 掉 special tokens。
        为 None 时：若 task_type=mlm 则默认 True，否则 False。
    """

    model_name: str = "facebook/esm2_t33_650M_UR50D"
    torch_autocast: bool = False
    include_final_norm: bool = False
    exclude_special_tokens: bool | None = None


@dataclass
class ESM2GenerationConfig:
    """ESM2 生成配置（模仿 ESM3 / Hugging Face 的 GenerationConfig 风格）.

    字段含义大致对齐 HF:
    - do_sample: False = greedy, True = 采样
    - temperature: 温度
    - top_k / top_p: 采样截断
    - repetition_penalty: >1 抑制重复 token
    - no_repeat_ngram_size: 禁止重复 n-gram（软实现，尽量避免）
    - num_steps: Gibbs 外层迭代轮数；若为空则默认设为长度
    - strategy: "default" = 按顺序更新；"random" = 每轮随机打乱位置
    - temperature_annealing: 是否对 temperature 做简单退火
    """

    num_steps: int = 20
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    strategy: str = "default"  # "default" / "random"
    temperature_annealing: bool = False


@register_model("esm2")
class ESM2Backend(ModelBackend):
    """ESM2 模型后端."""

    def __init__(
        self,
        *,
        backend_cfg: BackendConfig,
        init_cfg: ESM2InitConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(backend_cfg=backend_cfg)

        # dataclass 化 init 配置
        self.init_cfg = coerce_config(init_cfg, ESM2InitConfig)

        self.model_name = self.init_cfg.model_name
        self.torch_autocast = self.init_cfg.torch_autocast
        self.include_final_norm = self.init_cfg.include_final_norm

        # 根据 task_type 决定是否排除 special tokens
        task_kind = self.cfg.task_type  # "mlm" / "causal" 等
        self.exclude_special_tokens = (
            self.init_cfg.exclude_special_tokens
            if self.init_cfg.exclude_special_tokens is not None
            else task_kind == "mlm"
        )

        self.model: Any | None = None
        self.alphabet: Any | None = None
        self._batch_converter = None

        self._blocks: list[nn.Module] = []
        self._final_norm: nn.Module | None = None

        self._pad_id: int = 0
        self._special_ids: set[int] = set()

        # 生成相关：20 个标准氨基酸以及对应 token id 列表
        self._aa_letters: str = "ACDEFGHIKLMNPQRSTVWY"
        self._aa_ids: torch.Tensor | None = None

        self._device = torch.device(self.cfg.device)
        self._target_dtype = resolve_torch_dtype(self.cfg.dtype)

        self._autocast_device = self._device.type if self._device.type in {"cuda", "mps"} else None
        self._autocast_enabled = bool(self.torch_autocast and self._autocast_device)
        self._autocast_dtype = torch.float16

    # =============================
    # Lifecycle
    # =============================

    def load(self) -> None:
        """加载 Hugging Face 版本的 ESM2 模型 + tokenizer."""
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import transformers. 请确认已经安装了 `transformers` 包。"
            ) from exc

        name = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForMaskedLM.from_pretrained(name)

        if self._target_dtype is not None:
            model = model.to(device=self._device, dtype=self._target_dtype)
        else:
            model = model.to(device=self._device)
        model.eval()

        alphabet = HFAlphabetShim(tokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        self._special_ids, self._pad_id = get_special_ids(tokenizer)

        aa_ids: list[int] = []
        for aa in self._aa_letters:
            try:
                idx = int(alphabet.get_idx(aa))
                aa_ids.append(idx)
            except Exception:
                continue
        if aa_ids:
            self._aa_ids = torch.tensor(aa_ids, device=self._device, dtype=torch.long)
        else:
            self._aa_ids = None

        encoder = None
        if hasattr(model, "esm") and hasattr(model.esm, "encoder"):
            encoder = model.esm.encoder
        if encoder is not None:
            if not hasattr(model, "encoder"):
                model.encoder = encoder  # type: ignore[attr-defined]
            if hasattr(encoder, "layer") and not hasattr(encoder, "layers"):
                encoder.layers = encoder.layer  # type: ignore[attr-defined]
            if not hasattr(encoder, "norm"):
                encoder.norm = getattr(model.esm, "emb_layer_norm_after", None)  # type: ignore[attr-defined]
            if not hasattr(model, "layers") and hasattr(encoder, "layers"):
                model.layers = encoder.layers  # type: ignore[attr-defined]

        blocks, maybe_norm = find_transformer_blocks(model)
        self._blocks = blocks
        self._final_norm = maybe_norm if self.include_final_norm else None

    # =============================
    # Tokenization / Detokenization
    # =============================

    def tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        """把氨基酸序列转成 tokens + attention mask."""
        if self._batch_converter is None or self.alphabet is None:
            raise RuntimeError("Call load() before tokenize().")

        seqs = list(sequences)
        # batch_converter 接收 [(label, seq), ...]
        data = [("seq", s) for s in seqs]
        _labels, _strs, tokens = self._batch_converter(data)
        tokens = cast(torch.LongTensor, tokens)

        tokens = tokens.to(self._device, non_blocking=True)
        attn_for_model = (tokens != self._pad_id).to(dtype=torch.long)
        mask = build_attention_mask(
            tokens,
            pad_id=self._pad_id,
            special_token_ids=self._special_ids,
            exclude_special_tokens=self.exclude_special_tokens,
        )
        mask = mask.to(self._device, non_blocking=True)
        attn_for_model = attn_for_model.to(self._device, non_blocking=True)

        return {"tokens": tokens, "mask": mask, "attention_mask": attn_for_model}

    def detokenize(self, token_ids: Sequence[int] | Any) -> str:
        """把 token 序列解码回氨基酸序列（去掉 pad / special）。"""
        if self.alphabet is None:
            raise RuntimeError("Call load() before detokenize().")

        if isinstance(token_ids, torch.Tensor):
            ids_list = token_ids.tolist()
        else:
            ids_list = list(token_ids)

        aa_tokens: list[str] = []
        for idx in ids_list:
            idx_int = int(idx)
            if idx_int == self._pad_id or idx_int in self._special_ids:
                continue
            tok = self.alphabet.get_tok(idx_int)
            aa_tokens.append(tok)

        return "".join(aa_tokens)

    # =============================
    # Hidden states / activations
    # =============================

    def _forward_hidden_batch(
        self,
        token_batch: dict[str, torch.Tensor],
        layers: Sequence[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """跑一批 tokens，并在指定层上挂钩抓 activations."""
        if self.model is None:
            raise RuntimeError("Call load() before activations().")

        tokens = token_batch["tokens"]  # [B, T]
        attn_mask = token_batch["mask"]  # [B, T]
        model_attn_mask = token_batch.get("attention_mask")
        cache: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def _hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                # ESM2 内部用的是 [T, B, D]，而上层期望 [B, T, D]
                if (
                    hidden.dim() == 3
                    and hidden.shape[0] == tokens.shape[1]  # T
                    and hidden.shape[1] == tokens.shape[0]  # B
                ):
                    hidden = hidden.permute(1, 0, 2).contiguous()  # [T,B,D] -> [B,T,D]
                cache[layer_idx] = hidden

            return _hook

        with HookManager() as hm:
            for layer_idx in layers:
                module = self._module_for_layer(layer_idx)
                hm.add(module, _make_hook(layer_idx))
            self._run_model(tokens, attention_mask=model_attn_mask)

        hidden_list: list[torch.Tensor] = []
        for layer_idx in layers:
            if layer_idx not in cache:
                raise RuntimeError(f"No activations captured for layer {layer_idx}.")
            hidden_list.append(cache[layer_idx])

        return hidden_list, attn_mask

    def _module_for_layer(self, layer_idx: int) -> nn.Module:
        """根据 layer_idx 拿到对应的 nn.Module."""
        if layer_idx == -1:
            if self._final_norm is None:
                raise KeyError("Final norm was not captured; set include_final_norm=True.")
            return self._final_norm
        if layer_idx < 0 or layer_idx >= len(self._blocks):
            raise KeyError(f"Layer index {layer_idx} out of range for {len(self._blocks)} blocks.")
        return self._blocks[layer_idx]

    def _run_model(
        self,
        tokens: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """真正 forward 一次 ESM2 模型 (Hugging Face 接口)."""
        model = self.model
        if model is None:
            raise RuntimeError("Call load() before activations().")

        kwargs: dict[str, Any] = {"input_ids": tokens}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        if self._autocast_enabled and self._autocast_device is not None:
            with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                model(**kwargs)
        else:
            model(**kwargs)

    # =============================
    # Generation helpers (Gibbs-style masked sampling)
    # =============================

    def _build_gen_config(
        self,
        length: int,
        gen_cfg: dict[str, Any] | None,
    ) -> ESM2GenerationConfig:
        """把 **gen_cfg 映射到 ESM2GenerationConfig，并做一些别名兼容."""
        cfg_kwargs: dict[str, Any] = {}
        if gen_cfg is not None:
            cfg_kwargs.update(gen_cfg)

        # 兼容旧接口: steps -> num_steps
        if "steps" in cfg_kwargs and "num_steps" not in cfg_kwargs:
            cfg_kwargs["num_steps"] = cfg_kwargs.pop("steps")

        # dataclass 默认值作为“全局默认 GenerationConfig”
        cfg = ESM2GenerationConfig(**cfg_kwargs)

        # 若未指定 num_steps，就用 length
        if cfg.num_steps is None:
            cfg.num_steps = length
        else:
            cfg.num_steps = max(1, min(length, int(cfg.num_steps)))

        return cfg

    @staticmethod
    def _get_annealed_temperature(
        step: int,
        num_steps: int,
        initial_temperature: float,
    ) -> float:
        """简易版 temperature annealing，模仿 ESM3 源码里的逻辑."""
        step_ratio = step / max(1, (num_steps - 1))
        return max(initial_temperature - step_ratio, 0.001) ** 2

    def _sample_aa_from_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        current_seq: str | None = None,
        position: int | None = None,
    ) -> tuple[int, str]:
        """从 logits 中按 Hugging Face 风格解码一个氨基酸 token.

        支持：
        - temperature
        - top_k / top_p（仅 do_sample=True 时生效）
        - repetition_penalty (>1 抑制已出现过 token)
        - no_repeat_ngram_size（基于 current_seq 做 n-gram block）
        - do_sample=False 时执行 greedy（argmax）
        """
        if self.alphabet is None:
            raise RuntimeError("Call load() before generation.")

        device = logits.device
        logits = logits.clone()

        # ------- 候选 token 集合：优先 20 AA -------
        if self._aa_ids is not None and len(self._aa_ids) > 0:
            aa_ids = self._aa_ids.to(device)  # [N_aa]
            cand_ids = aa_ids
            cand_logits = logits[aa_ids]  # [N_aa]
        else:
            # fallback：屏蔽 special tokens，在全 vocab 上操作
            mask = torch.ones_like(logits, dtype=torch.bool)
            for sid in self._special_ids:
                if 0 <= sid < logits.numel():
                    mask[sid] = False
            cand_ids = torch.nonzero(mask, as_tuple=False).view(-1)
            cand_logits = logits[cand_ids]

        # ------- repetition_penalty: 对已经出现过的 AA 做惩罚（HF 风格） -------
        rep_pen = float(repetition_penalty)
        if rep_pen != 1.0 and current_seq is not None:
            used_aas: set[str] = set(current_seq)
            for idx in range(cand_ids.size(0)):
                tok_id = int(cand_ids[idx])
                tok = self.alphabet.get_tok(tok_id)  # type: ignore[union-attr]
                aa = tok[0]
                if aa in used_aas:
                    logit = cand_logits[idx]
                    if logit > 0:
                        cand_logits[idx] = logit / rep_pen
                    else:
                        cand_logits[idx] = logit * rep_pen

        # ------- n-gram blocking: no_repeat_ngram_size -------
        n = int(no_repeat_ngram_size)
        if n > 0 and current_seq is not None and position is not None and len(current_seq) >= n:
            L = len(current_seq)
            # 所有已有 n-gram
            existing_ngrams: set[str] = set(current_seq[i : i + n] for i in range(L - n + 1))
            # 如果当前位置已有一个 n-gram，把它从集合里去掉，
            # 这样允许“原地不动”而不会被当成重复。
            start_idx = position - n + 1
            if 0 <= start_idx <= L - n:
                existing_ngrams.discard(current_seq[start_idx : start_idx + n])

            banned = torch.zeros_like(cand_logits, dtype=torch.bool)
            if start_idx >= 0:
                prefix = current_seq[start_idx:position]
                for idx in range(cand_ids.size(0)):
                    tok_id = int(cand_ids[idx])
                    tok = self.alphabet.get_tok(tok_id)  # type: ignore[union-attr]
                    aa = tok[0]
                    ngram = prefix + aa
                    if ngram in existing_ngrams:
                        banned[idx] = True

            if banned.any():
                cand_logits = cand_logits.masked_fill(banned, float("-inf"))
                # 避免全被 ban 掉
                if not torch.isfinite(cand_logits).any():
                    cand_logits = logits[cand_ids]  # 回退到未 block 的值

        # ------- temperature 缩放 -------
        temp = max(float(temperature), 1e-6)
        cand_logits = cand_logits / temp

        # ------- greedy 模式：不采样，取 argmax -------
        if not do_sample:
            best_idx = int(torch.argmax(cand_logits).item())
            token_id = int(cand_ids[best_idx])
            tok = self.alphabet.get_tok(token_id)  # type: ignore[union-attr]
            aa = tok[0]
            return token_id, aa

        # ------- 采样模式：top-k / top-p + multinomial -------
        # 先按 logit 排序
        sorted_logits, sorted_indices = torch.sort(cand_logits, descending=True)
        sorted_ids = cand_ids[sorted_indices]

        # top-k
        if top_k is not None and top_k > 0 and top_k < sorted_logits.size(0):
            k = int(top_k)
            sorted_logits = sorted_logits[:k]
            sorted_ids = sorted_ids[:k]

        # top-p (nucleus)
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            cutoff = torch.searchsorted(
                cumulative_probs,
                torch.tensor(float(top_p), device=device),
            )
            cutoff = max(int(cutoff.item()) + 1, 1)
            sorted_logits = sorted_logits[:cutoff]
            sorted_ids = sorted_ids[:cutoff]

        probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx = int(torch.multinomial(probs, 1).item())
        token_id = int(sorted_ids[sampled_idx])

        tok = self.alphabet.get_tok(token_id)  # type: ignore[union-attr]
        aa = tok[0]
        return token_id, aa

    def _gibbs_sample_sequence(
        self,
        initial_seq: str,
        cfg,
        mutable_start: int = 0,
    ) -> str:
        """Gibbs-style 采样（ESM2）：
        - 每个 step 只 forward 一次；
        - 在同一次 forward 的 logits 上更新多个位置。

        参数
        ----
        initial_seq : 起始氨基酸序列（长度 L）
        cfg         : ESM2GenerationConfig，至少应包含
                      - num_steps
                      - temperature
                      其它（top_k / top_p 等）后面可以扩展
        mutable_start : 从哪个位置开始允许被更新（0-based，针对氨基酸序列本身）
        """
        if self.model is None or self.alphabet is None or self._batch_converter is None:
            raise RuntimeError("Call load() before generation.")

        model = self.model
        alphabet = self.alphabet
        batch_converter = self._batch_converter
        device = self._device

        # 1) 起始序列 & tokens
        seq = initial_seq  # 长度 L 的氨基酸字符串
        L = len(seq)

        # 从 cfg 里取步数 / 温度等（带默认值）
        num_steps = int(getattr(cfg, "num_steps", 1))
        num_steps = max(num_steps, 1)

        temperature = float(getattr(cfg, "temperature", 1.0))
        strategy = getattr(cfg, "strategy", "default")  # "default" or "random"
        temperature_annealing = bool(getattr(cfg, "temperature_annealing", False))

        data = [("seq", seq)]
        _, _, tokens = batch_converter(data)  # [1, T]，带 BOS/EOS
        tokens = tokens.to(device)
        attn_mask_model = (tokens != self._pad_id).to(dtype=torch.long, device=device)

        token_indices = tokens[0].tolist()
        aa_token_positions = [
            idx for idx, tok_id in enumerate(token_indices) if tok_id not in self._special_ids
        ]
        if len(aa_token_positions) != L:
            raise RuntimeError("Tokenizer special tokens mismatch: AA positions != sequence length.")

        # 2) 逐 step 迭代，每 step 只 forward 一次
        for step_idx in tqdm(
            range(num_steps), desc="Gibbs sampling, Generation step %d/%d", total=num_steps
        ):
            # 当前 step 的温度（支持简单退火）
            if temperature_annealing and num_steps > 1:
                # 简单的线性退火示例：从 temperature -> 0.1
                ratio = step_idx / float(num_steps - 1)
                step_temp = max(0.1, temperature * (1.0 - 0.9 * ratio))
            else:
                step_temp = temperature

            if mutable_start >= L:
                break

            # 这一步要更新的氨基酸位置（不含 BOS/EOS）
            positions = list(range(mutable_start, L))
            if not positions:
                break

            if strategy == "random":
                random.shuffle(positions)

            # 2.1 一次性把这些位置 mask 掉
            tokens_masked = tokens.clone()
            for i in positions:
                token_pos = aa_token_positions[i]
                tokens_masked[0, token_pos] = alphabet.mask_idx

            # 2.2 forward 一次，得到整条序列的 logits
            with torch.no_grad():
                if self._autocast_enabled and self._autocast_device is not None:
                    with torch.autocast(
                        device_type=self._autocast_device,
                        dtype=self._autocast_dtype,
                    ):
                        out = model(
                            input_ids=tokens_masked,
                            attention_mask=attn_mask_model,
                        )
                else:
                    out = model(
                        input_ids=tokens_masked,
                        attention_mask=attn_mask_model,
                    )

            # ESM2 logits: [B, T, V]，这里只有 batch=1
            logits_attr = getattr(out, "logits", out["logits"])  # 支持 dict / Namespace
            full_logits = logits_attr[0]  # [T, V]

            # 2.3 在同一次 forward 的 logits 上，逐位置采样并更新
            for i in positions:
                token_pos = aa_token_positions[i]
                logits_i = full_logits[token_pos, :]  # 对应 seq[i] 位置

                # 这里保持原有的采样接口，只用 temperature，避免改动太多
                token_id, aa = self._sample_aa_from_logits(
                    logits_i,
                    temperature=step_temp,
                )

                # 更新序列和 tokens，用于下一 step
                seq = seq[:i] + aa + seq[i + 1 :]
                tokens[0, token_pos] = token_id

        return seq

    # =============================
    # Generation API
    # =============================

    def generate_uncond(self, length: int, **gen_cfg: Any) -> str:  # type: ignore[override]
        """用 ESM2 通过 masked-LM 的 Gibbs 采样“真生成”一个长为 length 的序列.

        支持参数（全部可选）：

        - steps / num_steps: 外层 Gibbs 迭代轮数（默认 = length）
        - do_sample: bool，默认 True（保持原先“随机生成”的行为）
        - temperature: float，默认 1.0
        - top_k: int | None，默认 None
        - top_p: float | None，默认 None
        - repetition_penalty: float，默认 1.0
        - no_repeat_ngram_size: int，默认 0
        - strategy: "default" / "random"
        - temperature_annealing: bool
        """
        if length <= 0:
            return ""

        cfg = self._build_gen_config(length, gen_cfg)

        # 初始序列：完全随机 20 AA
        letters = self._aa_letters
        seq = "".join(random.choice(letters) for _ in range(length))

        return self._gibbs_sample_sequence(
            seq,
            cfg=cfg,
            mutable_start=0,
        )

    def generate_with_prefix(  # type: ignore[override]
        self,
        target_len: int,
        prefix: str,
        **gen_cfg: Any,
    ) -> str:
        """给定 prefix，通过 masked-LM 的 Gibbs 采样，把序列扩展/修饰到 target_len 长度.

        解码参数同 generate_uncond：
        - steps / num_steps, do_sample, temperature, top_k, top_p,
        - repetition_penalty, no_repeat_ngram_size, strategy, temperature_annealing
        """
        if target_len <= 0:
            return ""

        cfg = self._build_gen_config(target_len, gen_cfg)

        letters = self._aa_letters
        # 去掉 prefix 里面的空格等
        prefix = prefix.replace(" ", "")
        if len(prefix) >= target_len:
            # prefix 太长就直接截断；同时也允许在 prefix 上再 refinement
            seq = prefix[:target_len]
            mutable_start = 0  # 如果想只改后缀，可以改成 len(prefix)
        else:
            suffix_len = target_len - len(prefix)
            suffix = "".join(random.choice(letters) for _ in range(suffix_len))
            seq = prefix + suffix
            # 只更新 suffix 部分，prefix 固定
            mutable_start = len(prefix)

        return self._gibbs_sample_sequence(
            seq,
            cfg=cfg,
            mutable_start=mutable_start,
        )
