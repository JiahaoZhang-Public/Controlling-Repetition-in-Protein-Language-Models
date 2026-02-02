# Scripts Guide

This folder hosts runnable utilities grouped by language:

- `py/run/` — main experiment and evaluation entry points (Hydra-based).
- `py/run/sweeps/` — sweep spec generators; use `--index` for a single config or `--json` to dump all.
- `py/pre_analysis/` & `py/post_analysis/` — offline data/metric analysis helpers.
- `sh/test/` — lightweight smoke tests to validate pipelines locally.

## Quick smoke tests

- All-model UCCS sanity checks: `bash scripts/sh/test/local_uccs_smoke.sh`  
  Set `DEVICE=cuda` for GPU or `SKIP_DPLM=1` to skip the DPLM download.
- ProGen2 sweep subset: `bash scripts/sh/test/local_progen2_smoke.sh`
- DPLM / decoding ablation / other targeted smokes: see additional scripts under `sh/test`.

## Sweep helpers

Generate sweep specs without launching jobs:

```bash
python scripts/py/run/sweeps/sweep_dplm.py --json > sweep_dplm.json
python scripts/py/run/sweeps/sweep_progen2_base.py --index 0
```

Each entry outputs `id::override1,override2,...` ready to pass to your launcher or to `main_experiment.py`.
