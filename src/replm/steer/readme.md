# Unified Affine Steering

All steering operations in `replm` reduce to per-neuron affine transforms. Every edit is built from two elementwise operations over the hidden dimension:

- `mul`: multiplicative scaling (defaults to ones)
- `add`: additive bias (defaults to zeros)

This abstraction covers dense directional edits, sparse neuron suppression, replacements, and token-selective gating in a single, composable API.

## Public API

- `AffineEdit`: dataclass that stores layer index, optional dims, `mul`, `add`, and `token_mask`.
- `Steerer(model, specs)`: context manager that registers forward hooks, compiles edits per layer, and applies them in deterministic order (dense mul → dense add → sparse mul → sparse add).
- `SteerMethod.fit(...) -> SteerResult`: steering methods return ready-to-run `AffineEdit` objects (see `src/replm/steer/methods` and helpers).


Steering reduces to affine edits in neuron space. By composing `AffineEdit`s and compiling them into `LayerProgram`s, we unify directional, neuron-level, and replacement strategies while keeping execution deterministic and easy to reason about.
