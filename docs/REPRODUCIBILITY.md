# Reproducibility Guide

This checklist shows how to recreate the paper’s main steering runs and evaluation numbers in a fresh environment.

## 1) Environment

- Python 3.10 (minimum) and the pinned dependencies in `requirements/runtime.txt`.
- For GPU builds of PyTorch, install the wheel that matches your CUDA version, e.g.:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121  # adjust cu version
pip install -r requirements/runtime.txt
pip install -e .
```

Run a quick check:

```bash
PYTHONPATH=src pytest -q
```

## 2) Single-run reproduction (paper default)

The command below mirrors the configuration reported in the paper (UCCS/contrastive-layer steering on PROGEN2-Base):

```bash
python scripts/py/run/main_experiment.py \
  exp.id=paper_uccs_progen2 \
  runtime.device=cuda \
  dataset=posneg \
  models=progen2_base \
  methods=contrastive_layer \
  methods.layer=0 \
  generation.uncond.n=100 generation.prefix.n=100 \
  split.train=100 split.test=100 \
  dataset.opt.target_per_side=1000
```

Artifacts will be written under `outputs/paper_uucs_progen2/run_<timestamp>/`.

## 3) Evaluate generated sequences

Compute repetition, diversity, and structure metrics for any FASTA (e.g., unconditional generations from the run above):

```bash
python scripts/py/run/evaluate_sequences.py outputs/paper_uucs_progen2/run_*/uncond.steer.fasta \
  --structure-model esm3 --structure-device cuda
```

This writes `<fasta>.metrics.csv` and `<fasta>.summary.json`, which can be aggregated across runs.

## 4) Full sweeps used in the paper

Use the sweep spec generators below to enumerate every config (layer, seed, dataset, decoding method). Each entry prints `id::override1,override2,...` and can be consumed by your launcher or SLURM array.

- **ESM3 (masked)**  
  `python scripts/py/run/sweeps/sweep_esm3.py --json > sweep_esm3.json`
- **ESM2 (masked)**  
  `python scripts/py/run/sweeps/sweep_esm2.py --json > sweep_esm2.json`
- **DPLM (diffusion)**  
  `python scripts/py/run/sweeps/sweep_dplm.py --json > sweep_dplm.json`
- **ProGen2-Base (autoregressive)**  
  `python scripts/py/run/sweeps/sweep_progen2_base.py --json > sweep_progen2_base.json`
- **ProtGPT2 (autoregressive)**  
  `python scripts/py/run/sweeps/sweep_protgpt2.py --json > sweep_protgpt2.json`
- **Cross-model decoding ablations (ESM3 + ProGen2)**  
  `python scripts/py/run/sweeps/sweep_ablation_decoding.py --json > sweep_ablation_decoding.json`

Local smoke jobs (CPU-friendly) mirror CI and are useful to sanity-check dependencies:

```bash
scripts/sh/test/local_uccs_smoke.sh            # tiny UCCS run across all shipped models
scripts/sh/test/local_ablation_decoding_smoke.sh
```

## 5) Refreshing datasets (optional)

See `data/README.md` for regenerating negative pools and recomputing metrics. Plug the new file paths into Hydra via `dataset.pos_fasta`, `dataset.neg_fasta`, etc.

## 6) Tips for determinism

- Hydra/torch/NumPy seeds are all tied to `runtime.seed` (default 42).
- Generation length is stochastic; set `generation.uncond.length_min/length_max` and keep seeds fixed to match reported runs.
- When using GPUs, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` if you need strict determinism / memory stability.
