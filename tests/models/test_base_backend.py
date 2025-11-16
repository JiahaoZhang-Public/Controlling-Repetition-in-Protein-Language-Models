# tests/models/test_base_backend.py
from __future__ import annotations

from collections.abc import Sequence

import torch

from replm.config import BackendConfig
from replm.models.base import ModelBackend


class DummyBackend(ModelBackend):
    """Minimal backend to exercise ModelBackend pooling logic."""

    def __init__(
        self,
        *,
        backend_cfg: BackendConfig | None = None,
        task_type: str | None = None,
        dim: int = 3,
    ):
        cfg_kwargs = {}
        if backend_cfg is None:
            if task_type is None:
                raise ValueError("task_type is required for DummyBackend.")
            cfg_kwargs = {"task_type": task_type, "device": "cpu"}
        super().__init__(backend_cfg=backend_cfg, **cfg_kwargs)
        self.dim = dim
        self.vocab = {ch: i + 1 for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

    def load(self) -> None:  # pragma: no cover - nothing to load for dummy backend
        return None

    def tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        max_len = max(len(seq) for seq in sequences)
        tokens = torch.zeros(len(sequences), max_len, dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            ids = torch.tensor([self.vocab[ch] for ch in seq], dtype=torch.long)
            tokens[i, : len(ids)] = ids
            mask[i, : len(ids)] = True
        return {"tokens": tokens, "mask": mask}

    def detokenize(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        id_to_char = {v: k for k, v in self.vocab.items()}
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(id_to_char[i] for i in token_ids if i in id_to_char)

    def _forward_hidden_batch(
        self,
        token_batch: dict[str, torch.Tensor],
        layers: Sequence[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        tokens = token_batch["tokens"].float()
        base = tokens.unsqueeze(-1).repeat(1, 1, self.dim)
        hidden_list = []
        for layer in layers:
            hidden_list.append(base + float(layer))
        attn_mask = token_batch["mask"]
        return hidden_list, attn_mask

    def generate_uncond(self, length: int, **gen_cfg) -> str:
        raise NotImplementedError

    def generate_with_prefix(self, target_len: int, prefix: str, **gen_cfg) -> str:
        raise NotImplementedError


def test_model_backend_mean_pooling_handles_variable_length_sequences():
    backend = DummyBackend(task_type="mlm", dim=2)
    sequences = ["ABC", "AD"]
    layers = [0, 1]

    acts = backend.activations(sequences, layers, batch_size=2)
    assert acts.shape == (2, len(layers), backend.dim)

    expected_means = torch.tensor(
        [
            [(1 + 2 + 3) / 3, (1 + 2 + 3) / 3 + 1],
            [(1 + 4) / 2, (1 + 4) / 2 + 1],
        ],
        dtype=torch.float32,
    ).unsqueeze(-1).repeat(1, 1, backend.dim)
    assert torch.allclose(acts, expected_means)


def test_model_backend_last_nonpad_pooling_uses_final_token():
    backend = DummyBackend(task_type="causal", dim=4)
    sequences = ["BAC", "DD"]
    layers = [0, 2]

    acts = backend.activations(sequences, layers, batch_size=1)
    assert acts.shape == (2, len(layers), backend.dim)

    expected_last = torch.tensor(
        [
            [3, 3 + 2],  # last token of "BAC" is C -> 3
            [4, 4 + 2],  # last token of "DD" is D -> 4
        ],
        dtype=torch.float32,
    ).unsqueeze(-1).repeat(1, 1, backend.dim)
    assert torch.allclose(acts, expected_last)


def test_model_backend_accepts_shared_backend_config_instance():
    cfg = BackendConfig(task_type="mlm", device="cpu", dtype=None)
    backend = DummyBackend(backend_cfg=cfg, dim=1)
    assert backend.cfg is cfg
