# src/replm/steer/methods/control.py
"""
Control (no-op) steering method.

Use this as a baseline: it produces **no edits**, so running the model under
`Steerer(model, specs=[])` is equivalent to the unsteered forward pass.

Config example:

methods:
  name: control
  kwargs: {}

This deliberately satisfies the unified API:
- uses ActivationBatch for signature compatibility
- returns SteerResult with an empty `by_layer`
"""
from __future__ import annotations

from dataclasses import dataclass

from . import register_method
from .base import ActivationBatch, InputSpec, SteerMethod, SteerResult


@register_method("control")
@dataclass
class ControlNoOp(SteerMethod):
    """No-op steering: emits no edits (true control baseline)."""

    def requires(self) -> InputSpec:
        # No specific requirements; keep signature consistent.
        return InputSpec()

    def fit(self, data: ActivationBatch) -> SteerResult:
        # Return an empty by_layer mapping and minimal meta.
        return SteerResult(
            by_layer={},
            meta={
                "kind": "control",
                "note": "no-op baseline (no edits produced)",
            },
        )
