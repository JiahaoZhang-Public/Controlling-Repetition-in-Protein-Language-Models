from __future__ import annotations

import torch
from torch import nn

from replm.steer.ops import AffineEdit
from replm.steer.steerer import Steerer


class IdentityBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyTransformer(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList(IdentityBlock(hidden) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.transformer.blocks:
            out = block(out)
        return out


def test_steerer_applies_dense_affine_edits():
    model = DummyTransformer(hidden=4, layers=2)
    x = torch.ones(2, 4)
    baseline = model(x)

    edit = AffineEdit(layer=0, mul=torch.full((4,), 2.0), add=torch.ones(4))
    with Steerer(model, specs=[edit]):
        steered = model(x)

    expected = baseline * 2 + 1
    assert torch.allclose(steered, expected)


def test_steerer_applies_sparse_edits_with_token_mask():
    model = DummyTransformer(hidden=4, layers=2)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]])
    baseline = model(x)

    mask = torch.tensor([1.0, 0.0, 1.0])
    edit = AffineEdit(
        layer=1,
        dims=torch.tensor([2], dtype=torch.long),
        mul=torch.tensor([0.0]),
        token_mask=mask,
    )

    with Steerer(model, specs=[edit]):
        steered = model(x)

    expected = baseline.clone()
    expected[:, [0, 2], 2] = 0.0
    assert torch.allclose(steered, expected)
