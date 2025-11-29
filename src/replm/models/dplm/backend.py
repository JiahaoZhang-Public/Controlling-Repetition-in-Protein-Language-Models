from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence as SeqType, List

import torch
import torch.nn as nn

from .. import register_model
from ..base import ModelBackend
from ...config import BackendConfig, coerce_config
from ..utils import HookManager 
from .dplm import DiffusionProteinLanguageModel


@dataclass
class DPLMInitConfig:
    """
    DPLM 初始化配置
    """
    model_name: str = "airkingbd/dplm_150m"
    torch_autocast: bool = False


@dataclass
class DPLMGenerationConfig:
    """
    DPLM 生成配置，对应 DiffusionProteinLanguageModel.generate 的参数。
    """
    max_iter: int = 500
    temperature: float | None = None
    sampling_strategy: str = "gumbel_argmax"
    disable_resample: bool = False
    resample_ratio: float = 0.25


@register_model("dplm")
class DPLMBackend(ModelBackend):
    """
    把 DPLM 封成 REPLM 的 ModelBackend 接口。

    """

    def __init__(
        self,
        *,
        backend_cfg: BackendConfig,
        init_cfg: DPLMInitConfig | Mapping[str, Any] | None = None,
        gen_cfg: DPLMGenerationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(backend_cfg=backend_cfg)

        self.init_cfg = coerce_config(init_cfg, DPLMInitConfig)
        self.gen_cfg = coerce_config(gen_cfg, DPLMGenerationConfig)

        self.model: DiffusionProteinLanguageModel | None = None
        self.tokenizer: Any | None = None

        # transformer blocks 
        self._blocks: list[nn.Module] = []
        self._final_norm: nn.Module | None = None

        self._device = torch.device(self.cfg.device)
        self._target_dtype = (
            getattr(torch, str(self.cfg.dtype)) if self.cfg.dtype is not None else None
        )

        self._autocast_device = (
            self._device.type if self._device.type in {"cuda", "mps"} else None
        )
        self._autocast_enabled = bool(self.init_cfg.torch_autocast and self._autocast_device)
        self._autocast_dtype = torch.float16


    @property
    def layers(self) -> nn.ModuleList:  # type: ignore[override]
        """
        暴露给上层 / Steerer 的层列表。
        """
        return nn.ModuleList(self._blocks)

    @property
    def steering_layer_attr_path(self) -> tuple[str, ...] | None:  # type: ignore[override]
        """
        Steerer 会从 self.model.layers 下面拿 blocks。
        """
        return ("layers",)



    def _guess_expected_num_layers(self) -> int | None:
        """
        根据 model_name 猜一个 layer 数（来自 HF 模型卡）：
          dplm_150m -> 30
          dplm_650m -> 33
          dplm_3b   -> 36
        猜不到就返回 None。
        """
        name = self.init_cfg.model_name.rsplit("/", 1)[-1].lower()
        if "150m" in name:
            return 30
        if "650m" in name:
            return 33
        if "3b" in name:
            return 36
        return None

    def _init_blocks_from_model(self) -> None:
        """
        不依赖具体实现细节，只在整个 model 里搜：
        - 所有 nn.ModuleList
        - 里边元素都是 nn.Module
        - 优先长度 == 预期 num_layers 的那个
        """
        if self.model is None:
            raise RuntimeError("Call load() before _init_blocks_from_model().")

        expected = self._guess_expected_num_layers()
        exact: list[nn.ModuleList] = []
        others: list[nn.ModuleList] = []

        for mod in self.model.modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0 and all(
                isinstance(b, nn.Module) for b in mod
            ):
                if expected is not None and len(mod) == expected:
                    exact.append(mod)
                else:
                    others.append(mod)

        chosen: nn.ModuleList | None = None
        if exact:
            chosen = exact[0]
        elif others:
            chosen = others[0]

        if chosen is not None:
            self._blocks = list(chosen)
        else:

            self._blocks = [nn.Identity()]

        self._final_norm = None
        for mod in self.model.modules():
            cls = mod.__class__.__name__.lower()
            if "layernorm" in cls or cls.endswith("norm"):
                self._final_norm = mod  

        try:
            self.model.layers = nn.ModuleList(self._blocks)  # type: ignore[attr-defined]
        except Exception:
            pass


    def load(self) -> None:  # type: ignore[override]
        name = self.init_cfg.model_name
        dplm = DiffusionProteinLanguageModel.from_pretrained(
            net_name=name,
            cfg_override={},
            net_override={},
            from_huggingface=True,
        )

        if self._target_dtype is not None:
            dplm = dplm.to(device=self._device, dtype=self._target_dtype)
        else:
            dplm = dplm.to(device=self._device)
        dplm.eval()

        self.model = dplm
        self.tokenizer = dplm.tokenizer

        self._init_blocks_from_model()

    # ---------- hidden state / activations ----------

    def _module_for_layer(self, layer_idx: int) -> nn.Module:

        if layer_idx == -1:
            if self._final_norm is None:
                raise KeyError("Final norm was not captured; no layer -1 available.")
            return self._final_norm
        if layer_idx < 0 or layer_idx >= len(self._blocks):
            raise KeyError(f"Layer index {layer_idx} out of range for {len(self._blocks)} blocks.")
        return self._blocks[layer_idx]

    def _run_model(self, tokens: torch.Tensor) -> None:
        """
        只跑前向，真正的 hidden state 由 hook 提供。
        """
        if self.model is None:
            raise RuntimeError("Call load() before activations().")

        kwargs: dict[str, Any] = {
            "input_ids": tokens,
            "return_last_hidden_state": True,
        }

        if self._autocast_enabled and self._autocast_device is not None:
            with torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype):
                self.model(**kwargs)
        else:
            self.model(**kwargs)

    def _forward_hidden_batch(
        self,
        token_batch: dict[str, torch.Tensor],
        layers: SeqType[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Call load() before activations().")

        tokens = token_batch["tokens"]


        attn_mask = token_batch.get("attention_mask")
        if attn_mask is None:
            attn_mask = token_batch.get("mask")
        if attn_mask is None:
            attn_mask = torch.ones_like(tokens, dtype=torch.float32, device=tokens.device)

        cache: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def _hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                cache[layer_idx] = hidden

            return _hook

        with HookManager() as hm:
            for layer_idx in layers:
                module = self._module_for_layer(int(layer_idx))
                hm.add(module, _make_hook(int(layer_idx)))
            self._run_model(tokens)

        hidden_list: list[torch.Tensor] = []
        for layer_idx in layers:
            li = int(layer_idx)
            if li not in cache:
                raise RuntimeError(f"No activations captured for layer {li}.")
            hidden_list.append(cache[li])

        return hidden_list, attn_mask


    # ---------- token ----------

    def tokenize(self, sequences: SeqType[str]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        if self.tokenizer is None:
            raise RuntimeError("Call load() before tokenize().")

        seqs = list(sequences)
        encoded = self.tokenizer(
            seqs,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokens = encoded["input_ids"].to(torch.long).to(self._device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(self._device, non_blocking=True)

        return {"tokens": tokens, "attention_mask": attention_mask}

    def detokenize(self, token_ids: SeqType[int] | torch.Tensor) -> str:  
        if self.tokenizer is None:
            raise RuntimeError("Call load() before detokenize().")

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return "".join(text.split(" "))

    # ---------- activations()  ----------

    def activations(  
        self,
        sequences: SeqType[str],
        *,
        layers: SeqType[int],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        对每个指定的 layer：
          - 拿到 [B, T, D] hidden
          - 用 attention_mask 做 mean pooling -> [B, D]
        拼成 [N, L, D] 返回。
        """
        if self.model is None:
            raise RuntimeError("Call load() before activations().")

        self.model.eval()
        all_reps: List[torch.Tensor] = []

        with torch.no_grad():
            seqs = list(sequences)
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i : i + batch_size]
                tok = self.tokenize(batch_seqs)

                hidden_list, attn_mask = self._forward_hidden_batch(tok, layers)
                mask = attn_mask.to(hidden_list[0].dtype).unsqueeze(-1)  # [B, T, 1]

                reps_per_layer: List[torch.Tensor] = []
                for h in hidden_list:
                    # h: [B, T, D]
                    summed = (h * mask).sum(dim=1)  # [B, D]
                    denom = mask.sum(dim=1).clamp_min(1e-6)
                    rep = summed / denom            # [B, D]
                    reps_per_layer.append(rep.unsqueeze(1))  # [B, 1, D]

                batch_reps = torch.cat(reps_per_layer, dim=1)  # [B, L, D]
                all_reps.append(batch_reps)

        return torch.cat(all_reps, dim=0)

    # ----------  initialize_generation----------

    @staticmethod
    def _format_check(cond_seq: list[str], cond_position: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
        assert len(cond_seq) == len(cond_position), "The length of cond_seq and cond_position does not match."
        position_list: list[tuple[int, int]] = []
        for pos in cond_position:
            parts = pos.split("-")
            assert len(parts) == 2, "The format of position is illegal, which is not correctly splited by '-'"
            start_pos, end_pos = int(parts[0]), int(parts[1])
            assert end_pos >= start_pos, "The end position is smaller than start position."
            position_list.append((start_pos, end_pos))

        temp_position_list: list[int] = [p for tup in position_list for p in tup]
        for i in range(1, len(temp_position_list) - 2, 2):
            assert temp_position_list[i + 1] > temp_position_list[i], (
                "The position segment has overlap, which is not supported"
            )

        for i, (start_pos, end_pos) in enumerate(position_list):
            assert len(cond_seq[i]) == (end_pos - start_pos + 1), (
                "The length of each position segment and seq segment does not match."
            )

        return cond_seq, position_list

    def _initialize_generation(
        self,
        num_seqs: int,
        length: int,
        *,
        cond_seq: list[str] | None = None,
        cond_position: list[str] | None = None,
    ) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("Call load() before generation.")

        seq = ["<mask>"] * length
        if cond_seq is not None and cond_position is not None:
            seq_segment_list, position_list = self._format_check(cond_seq, cond_position)
            for i, (start_pos, end_pos) in enumerate(position_list):
                seq[start_pos : end_pos + 1] = [char for char in seq_segment_list[i]]

        seq = ["".join(seq)]
        init_seq = seq * num_seqs

        batch = self.tokenizer.batch_encode_plus(
            init_seq,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        batch = {
            "input_ids": batch["input_ids"],
            "input_mask": batch["attention_mask"].bool(),
        }

        def recursive_to(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [recursive_to(o, device=device) for o in obj]
            elif isinstance(obj, tuple):
                return tuple(recursive_to(o, device=device) for o in obj)
            elif isinstance(obj, dict):
                return {k: recursive_to(v, device=device) for k, v in obj.items()}
            else:
                return obj

        batch = recursive_to(batch, self._device)
        return batch["input_ids"]


    def _build_gen_cfg(self, overrides: Mapping[str, Any] | None = None) -> DPLMGenerationConfig:
        base = self.gen_cfg
        if overrides:
            data = base.__dict__ | dict(overrides)
            return DPLMGenerationConfig(**data)
        return base

    def _init_input_tokens(self, length: int, num_seqs: int = 1) -> torch.Tensor:
        return self._initialize_generation(num_seqs=num_seqs, length=length)

    def generate_uncond(self, length: int, **gen_cfg: Any) -> str:  # type: ignore[override]
        if self.model is None:
            raise RuntimeError("Call load() before generation.")

        if length <= 0:
            return ""

        cfg = self._build_gen_cfg(gen_cfg)
        input_tokens = self._init_input_tokens(length=length, num_seqs=1)
        partial_mask = input_tokens.ne(self.model.mask_id)

        with torch.no_grad():
            samples = self.model.generate(
                input_tokens=input_tokens,
                tokenizer=self.tokenizer,
                max_iter=int(cfg.max_iter),
                temperature=cfg.temperature,
                sampling_strategy=cfg.sampling_strategy,
                partial_masks=partial_mask,
                disable_resample=cfg.disable_resample,
                resample_ratio=cfg.resample_ratio,
            )

        text = self.tokenizer.batch_decode(
            samples,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return "".join(text.split(" "))

    def generate_with_prefix(  # type: ignore[override]
        self,
        target_len: int,
        prefix: str,
        **gen_cfg: Any,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Call load() before generation.")

        if target_len <= 0:
            return ""

        prefix = prefix.replace(" ", "")
        if len(prefix) == 0:
            return self.generate_uncond(target_len, **gen_cfg)

        if len(prefix) > target_len:
            prefix = prefix[:target_len]

        cfg = self._build_gen_cfg(gen_cfg)

        input_tokens = self._initialize_generation(
            num_seqs=1,
            length=target_len,
            cond_seq=[prefix],
            cond_position=[f"0-{len(prefix) - 1}"],
        )
        partial_mask = input_tokens.ne(self.model.mask_id)

        with torch.no_grad():
            samples = self.model.generate(
                input_tokens=input_tokens,
                tokenizer=self.tokenizer,
                max_iter=int(cfg.max_iter),
                temperature=cfg.temperature,
                sampling_strategy=cfg.sampling_strategy,
                partial_masks=partial_mask,
                disable_resample=cfg.disable_resample,
                resample_ratio=cfg.resample_ratio,
            )

        text = self.tokenizer.batch_decode(
            samples,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return "".join(text.split(" "))
