#!/usr/bin/env python
# scripts/py/run/generate_esm3_steps.py
"""
Generate fixed-length sequences using the ESM3 backend with a specified
generation num_steps override, and write a FASTA file.

Example:
    python scripts/py/run/generate_esm3_steps.py \
      --length 200 --num-seqs 100 --steps 40 \
      --output data/neg/esm3_len200_steps40.fasta \
      --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from replm.config import BackendConfig
from replm.models import get_model_class
from replm.utils.io import write_fasta

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate fixed-length sequences using ESM3 with a num_steps override.",
    )
    p.add_argument("--length", type=int, required=True, help="Target sequence length.")
    p.add_argument("--num-seqs", type=int, default=100, help="Number of sequences to generate.")
    p.add_argument("--steps", type=int, required=True, help="Generation num_steps override.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output FASTA path (defaults to data/neg/esm3_len{L}_steps{S}.fasta)",
    )
    p.add_argument("--device", type=str, default="cpu", help="Device override (cpu, cuda, cuda:0, ...)")
    return p


def _load_cfg(device: str) -> DictConfig:
    overrides = [
        f"+runtime.device={device}",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="models/esm3", overrides=overrides)


def _instantiate_backend(model_cfg: DictConfig):
    # Resolve backend block safely (supports struct mode)
    backend_node = OmegaConf.select(model_cfg, "backend") or OmegaConf.select(model_cfg, "models.backend")
    if backend_node is None:
        raise KeyError("Model config is missing 'backend' block")
    backend_raw = OmegaConf.to_container(backend_node, resolve=True)
    backend_cfg = BackendConfig(**backend_raw)  # type: ignore[arg-type]

    init_node = OmegaConf.select(model_cfg, "init")
    generation_node = OmegaConf.select(model_cfg, "generation")
    init_cfg = instantiate(init_node) if init_node is not None else None
    gen_cfg = instantiate(generation_node) if generation_node is not None else None

    resolved_name = OmegaConf.select(model_cfg, "name") or "esm3"
    backend_cls = get_model_class(str(resolved_name))
    return backend_cls(backend_cfg=backend_cfg, init_cfg=init_cfg, gen_cfg=gen_cfg)


def main() -> None:
    args = _build_parser().parse_args()
    if args.length <= 0:
        raise ValueError("--length must be positive")
    if args.num_seqs <= 0:
        raise ValueError("--num-seqs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    out_path = args.output or Path(f"data/neg/esm3_len{args.length}_steps{args.steps}.fasta")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(args.device)
    backend = _instantiate_backend(cfg)
    backend.load()

    records: list[tuple[str, str]] = []
    for i in range(args.num_seqs):
        seq = backend.generate_uncond(length=int(args.length), num_steps=int(args.steps))
        header = f"esm3_{i:05d}_len{args.length}_steps{args.steps}"
        records.append((header, seq))

    write_fasta(records, out_path)
    print(f"[INFO] Wrote {len(records)} sequences to {out_path}")


if __name__ == "__main__":
    main()
