# Scripts Guide

This folder hosts runnable utilities grouped by language:

- `py/run/` — main experiment and evaluation entry points (Hydra-based).
- `py/run/sweeps/` — sweep spec generators; use `--index` for a single config or `--json` to dump all.
- `py/pre_analysis/` & `py/post_analysis/` — offline data/metric analysis helpers.
- `sh/test/` — lightweight smoke tests to validate pipelines locally.

## Quick smoke tests

- All-model UCCS sanity checks: `bash scripts/sh/test/local_uccs_smoke.sh`  
  Set `DEVICE=cuda` for GPU
  
## Sweep helpers

Generate sweep specs without launching jobs (all mirrored in the paper):

- ESM3 (masked): `python scripts/py/run/sweeps/sweep_esm3.py --json > sweep_esm3.json`
- ESM2 (masked): `python scripts/py/run/sweeps/sweep_esm2.py --json > sweep_esm2.json`
- DPLM (diffusion): `python scripts/py/run/sweeps/sweep_dplm.py --json > sweep_dplm.json`
- ProGen2-Base (autoregressive): `python scripts/py/run/sweeps/sweep_progen2_base.py --json > sweep_progen2_base.json`
- Decoding ablations (ESM3 + ProGen2): `python scripts/py/run/sweeps/sweep_ablation_decoding.py --json > sweep_ablation_decoding.json`

Each entry prints `id::override1,override2,...`, ready for your launcher or direct use with `main_experiment.py --config-path configs`.
