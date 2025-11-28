from __future__ import annotations

from pathlib import Path

import torch

from replm.steer.io import (
    affine_edit_from_dict,
    affine_edit_to_dict,
    load_steer_result,
    save_steer_result,
)
from replm.steer.methods.base import SteerResult
from replm.steer.ops import AffineEdit


def test_affine_edit_serialization_roundtrip(tmp_path: Path):
    edit = AffineEdit(
        layer=1, dims=torch.tensor([0, 2]), mul=torch.tensor([0.5, 1.5]), add=torch.tensor([1.0, -1.0])
    )  # noqa: E501
    payload = affine_edit_to_dict(edit)
    restored = affine_edit_from_dict(payload)
    assert torch.allclose(restored.mul, edit.mul)
    assert torch.allclose(restored.add, edit.add)
    assert torch.equal(restored.dims, edit.dims)


def test_steer_result_save_load(tmp_path: Path):
    edits = {0: [AffineEdit(layer=0, dims=None, add=torch.tensor([1.0, 2.0]))]}
    result = SteerResult(by_layer=edits, meta={"foo": "bar"})
    path = tmp_path / "steer.json"
    save_steer_result(result, path)
    loaded = load_steer_result(path)
    assert loaded.meta == result.meta
    assert loaded.by_layer.keys() == result.by_layer.keys()
    restored_edit = loaded.by_layer[0][0]
    assert torch.allclose(restored_edit.add, torch.tensor([1.0, 2.0]))
