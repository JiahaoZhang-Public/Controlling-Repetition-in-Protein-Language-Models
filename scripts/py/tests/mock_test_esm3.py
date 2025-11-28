import importlib
import sys
import types

import torch
from torch import nn

from replm.config import BackendConfig
from replm.models.masked.esm3_backend import ESM3Backend
from replm.models.masked.esm3_config import ESM3InitConfig
from replm.models.utils import get_special_ids


def ensure_fake_esm_sdk() -> None:
    """
    Provide a lightweight shim for esm.sdk.api if the real dependency is absent.
    The backend only needs GenerationConfig and ESMProtein for test-time stubs.
    """
    try:
        importlib.import_module("esm.sdk.api")
        return
    except Exception:
        pass

    esm_mod = types.ModuleType("esm")
    sdk_mod = types.ModuleType("esm.sdk")
    api_mod = types.ModuleType("esm.sdk.api")

    class GenerationConfig:
        def __init__(self, track: str, num_steps: int):
            self.track = track
            self.num_steps = num_steps

    class ESMProtein:
        def __init__(self, sequence: str):
            self.sequence = sequence

    api_mod.GenerationConfig = GenerationConfig
    api_mod.ESMProtein = ESMProtein
    sdk_mod.api = api_mod
    esm_mod.sdk = sdk_mod
    sys.modules["esm"] = esm_mod
    sys.modules["esm.sdk"] = sdk_mod
    sys.modules["esm.sdk.api"] = api_mod


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

    def decode(self, token_ids):
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
        self.transformer.blocks = nn.ModuleList(FakeBlock(i, hidden_dim) for i in range(num_layers))
        self.transformer.norm = nn.Identity()
        self.tokenizers = type("Toks", (), {"sequence": FakeTokenizer()})()

    def forward(self, *, sequence_tokens: torch.Tensor) -> torch.Tensor:
        hidden = sequence_tokens.float().unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        for block in self.transformer.blocks:
            hidden = block(hidden)
        hidden = self.transformer.norm(hidden)
        return hidden

    def generate(self, *_args, **_kwargs):
        raise NotImplementedError


class DummyESM3Generator:
    def __init__(self):
        self.last_cfg = None
        self.last_protein = None

    def generate(self, protein, cfg):
        self.last_cfg = cfg
        self.last_protein = protein
        length = len(getattr(protein, "sequence", ""))
        sequence = "M" * length
        return type("Out", (), {"sequence": sequence})()


def _manual_layer_embeddings(backend: ESM3Backend, sequences, layers) -> torch.Tensor:
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


def run_activation_parity_test() -> None:
    backend = ESM3Backend(
        backend_cfg=BackendConfig(task_type="mlm"),
        init_cfg=ESM3InitConfig(
            model_name="mock",
            torch_autocast=False,
            include_final_norm=False,
            exclude_special_tokens=True,
        ),
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
    torch.testing.assert_close(backend_out, manual_out)
    print("✔ Activation extraction matches raw forward-hook captures.")


def run_generation_test() -> None:
    backend = ESM3Backend(
        backend_cfg=BackendConfig(task_type="mlm"),
        init_cfg=ESM3InitConfig(model_name="mock", torch_autocast=False),
    )
    backend.model = DummyESM3Generator()
    seq = backend.generate_uncond(length=100)
    assert len(seq) == 100, "Generated sequence should match requested length."

    cfg = backend.model.last_cfg
    protein = backend.model.last_protein
    assert protein.sequence == "_" * 100, "Unconditional prompt should be masked underscores."
    assert cfg.track == "sequence"
    assert cfg.num_steps == 20, "Default num_steps should remain at the GenerationConfig default."
    print("✔ Unconditional generation works with default config for length 100.")


def main():
    ensure_fake_esm_sdk()
    run_activation_parity_test()
    run_generation_test()


if __name__ == "__main__":
    main()
