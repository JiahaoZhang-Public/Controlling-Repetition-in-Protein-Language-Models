# Controlling-Repetition-in-Protein-Language-Models

｜ Official implementation of “Controlling Repetition in Protein Language Models” (ICLR 2026)

## Overview

Protein Language Models (PLMs) have achieved major breakthroughs in structure prediction and de novo design. However, they frequently collapse into pathological repetition — generating redundant motifs or long homopolymer stretches.
Unlike in text generation, where repetition reduces readability, in proteins it undermines structural stability and biological functionality.

This repository provides the official implementation and reproducible experiments for our paper, which introduces the first systematic framework for understanding and controlling repetition in PLMs.

## Environment Setup

We recommend Python 3.10 or 3.11. Create an isolated environment (e.g., with `venv` or `conda`) and install runtime dependencies via `requirements/runtime.txt` (or `requirements/dev.txt` for contributors) before installing the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m pip install --upgrade pip
pip install -r requirements/dev.txt      # or requirements/runtime.txt for users
pip install -e .
```

The `dev` requirements install the testing/linting toolchain (pytest, ruff, mypy). Run the test suite to verify your setup:

```bash
pytest
```

Configurations are managed via Hydra; override modules by selecting configs under `configs/{datasets,models,methods}` when running CLI entry points.
