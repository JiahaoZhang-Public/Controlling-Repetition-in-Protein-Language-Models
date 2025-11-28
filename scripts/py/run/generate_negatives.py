#!/usr/bin/env python
# scripts/py/run/generate_negatives.py
"""
Generate raw negative FASTA samples using one or more language models.
Example:
    python scripts/py/run/generate_negatives.py \
    --models esm3 protgpt2 progen2_small progen2_base \
    --samples-per-model 10000 \
    --min-len 50 \
    --max-len 1024 \
    --output-dir data/neg \
    --seed 42 \
    --device cpu
"""
from __future__ import annotations

import argparse
import inspect
import random
from collections.abc import Iterable
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from replm.config import BackendConfig
from replm.models import get_model_class
from replm.utils.io import write_fasta

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - best-effort fallback if tqdm is missing

    class tqdm:  # minimal no-op shim
        def __init__(self, *_, **__):
            pass

        def update(self, *_: int) -> None:  # noqa: D401
            """No-op update"""
            return None

        def close(self) -> None:  # noqa: D401
            """No-op close"""
            return None


# Adjust this relative depth to your repo layout
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate raw negative FASTA samples using one or more language models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["esm3", "protgpt2", "progen2_base"],
        help="Model config names under configs/models (e.g., esm3, protgpt2, progen2_small, progen2_base).",
    )
    parser.add_argument(
        "--samples-per-model",
        type=int,
        default=10000,
        help="Number of sequences to generate per model.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=50,
        help="Minimum sequence length (inclusive).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum sequence length (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/neg"),
        help="Directory where per-model FASTA files will be written (defaults to data/neg).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling lengths.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device override for ${runtime.device} (e.g., cpu, cuda:0).",
    )
    return parser


def _load_model_config(name: str, device: str) -> DictConfig:
    overrides = [f"+runtime.device={device}"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=f"models/{name}", overrides=overrides)
    return cfg


def _instantiate_backend(model_cfg: DictConfig, model_name: str):
    # Safely pull nested blocks regardless of struct mode
    backend_node = OmegaConf.select(model_cfg, "backend") or OmegaConf.select(model_cfg, "models.backend")
    if backend_node is None:
        raise KeyError(
            f"Config for model '{OmegaConf.select(model_cfg, 'name') or model_name}' is missing a backend block."  # noqa: E501
        )
    backend_raw = OmegaConf.to_container(backend_node, resolve=True)
    backend_cfg = BackendConfig(**backend_raw)

    # Params (optional)
    params_source = OmegaConf.select(model_cfg, "params") or OmegaConf.select(model_cfg, "models.params")
    params_cfg = OmegaConf.to_container(params_source, resolve=True) if params_source is not None else {}
    if not isinstance(params_cfg, dict):
        params_cfg = {}

    # Optional init/generation configs (instantiate if provided)
    init_node = OmegaConf.select(model_cfg, "init")
    generation_node = OmegaConf.select(model_cfg, "generation")
    init_cfg = instantiate(init_node) if init_node is not None else None
    generation_cfg = instantiate(generation_node) if generation_node is not None else None

    # Resolve class to use, preferring config's 'name' if available
    resolved_name = (
        OmegaConf.select(model_cfg, "name") or OmegaConf.select(model_cfg, "models.name") or model_name
    )
    backend_cls = get_model_class(resolved_name)

    # Filter params to match backend __init__ signature unless it accepts **kwargs
    sig = inspect.signature(backend_cls.__init__)
    accepted = set(sig.parameters)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    params = dict(params_cfg) if has_var_kwargs else {k: v for k, v in params_cfg.items() if k in accepted}

    if init_cfg is not None and "init_cfg" in accepted:
        params["init_cfg"] = init_cfg
    if generation_cfg is not None and "gen_cfg" in accepted:
        params["gen_cfg"] = generation_cfg

    return backend_cls(backend_cfg=backend_cfg, **params)


def _iter_lengths(rng: random.Random, count: int, lo: int, hi: int) -> Iterable[int]:
    for _ in range(count):
        yield rng.randint(lo, hi)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.min_len <= 0 or args.max_len < args.min_len:
        raise ValueError("Length bounds must satisfy 0 < min_len <= max_len.")
    if args.samples_per_model <= 0:
        raise ValueError("samples-per-model must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for model_name in args.models:
        print(f"[INFO] Loading model config '{model_name}' on device '{args.device}'")
        model_cfg = _load_model_config(model_name, args.device)

        backend = _instantiate_backend(model_cfg, model_name)
        backend.load()

        print(f"[INFO] Generating {args.samples_per_model} sequences with {model_name}")
        records: list[tuple[str, str]] = []
        pbar = tqdm(total=args.samples_per_model, desc=f"Generating[{model_name}]", unit="seq")
        try:
            for idx, length in enumerate(
                _iter_lengths(rng, args.samples_per_model, args.min_len, args.max_len)
            ):
                seq = backend.generate_uncond(length=int(length))
                header = f"{model_name}_{idx:05d}_len{length}"
                records.append((header, seq))
                pbar.update(1)
        finally:
            pbar.close()

        out_path = args.output_dir / f"{model_name}_neg.fasta"
        write_fasta(records, out_path)
        print(f"[INFO] Wrote {len(records)} sequences to {out_path}")


if __name__ == "__main__":
    main()
