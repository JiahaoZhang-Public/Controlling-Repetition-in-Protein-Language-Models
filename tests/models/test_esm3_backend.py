# tests/models/test_esm3_backend.py
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from replm.models.masked.esm3_backend import ESM3Backend
from replm.models.utils import get_special_ids


class FakeTokenizer:
    pad_token_id = 0
    cls_token_id = 101
    eos_token_id = 102
    special_token_ids = (pad_token_id, cls_token_id, eos_token_id)

    def encode(self, seq: str, add_special_tokens: bool = True) -> list[int]:
        content = [ord(ch) - 64 for ch in seq]
        if not add_special_tokens:
            return content
        return [self.cls_token_id, *content, self.eos_token_id]

    def decode(self, token_ids):  # pragma: no cover - unused but required by backend
        return "".join(chr(i + 64) for i in token_ids if i > 0)


class FakeBlock(nn.Module):
    def __init__(self, layer_idx: int, hidden_dim: int):
        super().__init__()
        self.register_buffer("shift", torch.tensor(float(layer_idx + 1)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.shift


class FakeESM3Model(nn.Module):
    def __init__(self, hidden_dim: int = 3, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList(
            FakeBlock(i, hidden_dim) for i in range(num_layers)
        )
        self.transformer.norm = nn.Identity()
        self.tokenizers = type("Toks", (), {"sequence": FakeTokenizer()})()

    def forward(self, *, sequence_tokens: torch.Tensor) -> torch.Tensor:
        hidden = sequence_tokens.float().unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        for block in self.transformer.blocks:
            hidden = block(hidden)
        hidden = self.transformer.norm(hidden)
        return hidden

    def generate(self, *args, **kwargs):  # pragma: no cover - not needed for the test
        raise NotImplementedError


def _manual_layer_embeddings(
    backend: ESM3Backend,
    sequences: Sequence[str],
    layers: Sequence[int],
) -> torch.Tensor:
    token_batch = backend.tokenize(sequences)
    manual_cache: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in layers:
        module = backend._module_for_layer(layer_idx)

        def _hook(_module, _inp, output, lid=layer_idx):
            manual_cache[lid] = output.detach().clone()

        handles.append(module.register_forward_hook(_hook))

    backend._run_model(token_batch["tokens"])
    for handle in handles:
        handle.remove()

    pooled = []
    mask = token_batch["mask"]
    pool_mode = backend.cfg.resolved_pooling()
    for layer_idx in layers:
        pooled.append(backend._pool_hidden(manual_cache[layer_idx], mask, mode=pool_mode))
    return torch.stack(pooled, dim=1)


def test_esm3_backend_matches_manual_hooked_embeddings():
    backend = ESM3Backend(
        task_type="mlm",
        device="cpu",
        torch_autocast=False,
        include_final_norm=False,
    )

    fake_model = FakeESM3Model(hidden_dim=4, num_layers=2)
    backend.model = fake_model
    backend.tokenizer = fake_model.tokenizers.sequence
    backend._blocks = list(fake_model.transformer.blocks)
    backend._final_norm = None
    backend._special_ids, backend._pad_id = get_special_ids(backend.tokenizer)

    sequences = ["AC", "B"]
    layers = [0, 1]

    backend_out = backend.activations(sequences, layers, batch_size=2)
    manual_out = _manual_layer_embeddings(backend, sequences, layers)

    assert torch.allclose(backend_out, manual_out)

