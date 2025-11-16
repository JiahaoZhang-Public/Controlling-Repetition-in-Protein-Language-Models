from __future__ import annotations

import torch

from replm.steer.methods.base import ActivationBatch
from replm.steer.methods.contrastive_layer import ContrastiveLayer
from replm.steer.methods.control import ControlNoOp
from replm.steer.methods.helpers import dense_add, sparse_add, sparse_replace
from replm.steer.methods.neuron_topk import NeuronTopK
from replm.steer.methods.probe import ProbeNaiveBinary


def _base_batch(layer_tensor: torch.Tensor, pos, neg) -> ActivationBatch:
    return ActivationBatch(
        by_layer={0: layer_tensor},
        positive_idx=torch.tensor(pos, dtype=torch.long),
        negative_idx=torch.tensor(neg, dtype=torch.long),
    )


def test_control_noop_emits_empty_edits():
    batch = _base_batch(torch.zeros(4, 3), [0, 1], [2, 3])
    method = ControlNoOp()
    result = method.fit(batch)
    assert result.by_layer == {}
    assert result.meta["kind"] == "control"


def test_contrastive_layer_returns_mean_difference_direction():
    layer = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ]
    )
    batch = _base_batch(layer, [0, 1], [2, 3])
    method = ContrastiveLayer(layer=0, normalize=False, var_scale=False, alpha=1.0)
    result = method.fit(batch)
    edit = result.by_layer[0][0]
    expected = torch.tensor([1.5, -1.5])  # mean(pos) - mean(neg)
    assert torch.allclose(edit.add, expected, atol=1e-5)


def test_dense_and_sparse_helpers_build_expected_affine_edits():
    vec = torch.tensor([3.0, 4.0])
    edit_dense = dense_add(layer=1, vec=vec, normalize=True, alpha=2.0)
    assert torch.allclose(edit_dense.add, torch.tensor([1.2, 1.6]))

    dims = torch.tensor([0, 2])
    values = torch.tensor([5.0, -1.0])
    edit_sparse = sparse_add(layer=2, dims=dims, values=values)
    assert edit_sparse.dims.tolist() == [0, 2]
    assert torch.allclose(edit_sparse.add, values)

    replace = sparse_replace(layer=3, dims=dims, values=values)
    assert torch.allclose(replace.mul, torch.zeros_like(values))
    assert torch.allclose(replace.add, values)


def test_probe_naive_binary_learns_direction():
    # Positives differ from negatives only along the first dimension
    layer = torch.tensor(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ]
    )
    batch = ActivationBatch(
        by_layer={0: layer},
        positive_idx=torch.tensor([0, 1], dtype=torch.long),
        negative_idx=torch.tensor([2, 3], dtype=torch.long),
    )
    method = ProbeNaiveBinary(layer=0, normalize=True, alpha=1.0, epochs=100, lr=0.05)
    result = method.fit(batch)
    edit = result.by_layer[0][0]
    # Direction should point along first dimension after normalization
    assert torch.allclose(edit.add[0], torch.tensor(1.0), atol=1e-2)
    assert abs(float(edit.add[1])) < 0.2


def test_neuron_topk_global_ranking_sets_mul_to_zero():
    layer0 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    layer1 = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])

    batch = ActivationBatch(by_layer={0: layer0, 1: layer1}, y=y)
    method = NeuronTopK(topk=2)
    result = method.fit(batch)

    assert set(result.by_layer.keys()) == {0, 1}
    # layer0 should have neuron index 0 zeroed, layer1 index 1
    edit0 = result.by_layer[0][0]
    assert torch.all(edit0.mul == torch.tensor([0.0], dtype=torch.float32))
    assert edit0.dims.tolist() == [0]

    edit1 = result.by_layer[1][0]
    assert edit1.dims.tolist() == [1]


def test_neuron_topk_handles_token_averaging():
    layer = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )  # (N=2, T=2, D=2)
    y = torch.tensor([1.0, 0.0])

    batch = ActivationBatch(by_layer={0: layer}, y=y)
    method = NeuronTopK(topk=1)
    result = method.fit(batch)

    edit = result.by_layer[0][0]
    assert edit.dims.tolist() == [0]
