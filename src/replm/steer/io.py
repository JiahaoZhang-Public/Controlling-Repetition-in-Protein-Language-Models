"""Serialization helpers for steering results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from .methods.base import SteerResult
from .ops import AffineEdit


def _tensor_to_list(value, *, numeric_type: str | None = None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().tolist()
    else:
        data = value
    if numeric_type == "int" and isinstance(data, Iterable):
        return [int(x) for x in data]
    return data


def _list_to_tensor(value, *, dtype: torch.dtype) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.tensor(value, dtype=dtype)


def affine_edit_to_dict(edit: AffineEdit) -> dict:
    return {
        "layer": int(edit.layer),
        "dims": _tensor_to_list(edit.dims, numeric_type="int"),
        "mul": _tensor_to_list(edit.mul),
        "add": _tensor_to_list(edit.add),
        "token_mask": _tensor_to_list(edit.token_mask),
    }


def affine_edit_from_dict(data: dict[str, Any]) -> AffineEdit:
    return AffineEdit(
        layer=int(data["layer"]),
        dims=_list_to_tensor(data.get("dims"), dtype=torch.long),
        mul=_list_to_tensor(data.get("mul"), dtype=torch.float32),
        add=_list_to_tensor(data.get("add"), dtype=torch.float32),
        token_mask=_list_to_tensor(data.get("token_mask"), dtype=torch.float32),
    )


def steer_result_to_dict(result: SteerResult) -> dict:
    by_layer = {
        str(layer): [affine_edit_to_dict(edit) for edit in edits]
        for layer, edits in (result.by_layer or {}).items()
    }
    return {"by_layer": by_layer, "meta": result.meta or {}}


def steer_result_from_dict(payload: dict[str, Any]) -> SteerResult:
    by_layer = {
        int(layer): [affine_edit_from_dict(edit) for edit in edits]
        for layer, edits in payload.get("by_layer", {}).items()
    }
    return SteerResult(by_layer=by_layer, meta=payload.get("meta", {}))


def save_steer_result(result: SteerResult, path: str | Path) -> None:
    payload = steer_result_to_dict(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_steer_result(path: str | Path) -> SteerResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return steer_result_from_dict(data)


__all__ = [
    "affine_edit_from_dict",
    "affine_edit_to_dict",
    "load_steer_result",
    "save_steer_result",
    "steer_result_from_dict",
    "steer_result_to_dict",
]
