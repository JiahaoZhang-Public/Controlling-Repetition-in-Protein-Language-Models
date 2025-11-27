"""Layer-wise probe analysis for repetition signals.

This script builds a balanced positive/negative set using
``PosNegProvider`` and trains a simple binary probe on pooled layer
activations to predict high-vs-low repetition. It plots layer index
against probe accuracy and AUC.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from replm.config import BackendConfig
from replm.datasets.posneg_provider import PosNegProvider
from replm.models import get_model_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a layer-wise binary probe on model activations to predict "
            "repetition (high-entropy positives vs low-entropy negatives)."
        )
    )

    parser.add_argument(
        "--pos-fasta",
        type=Path,
        default=Path("data/pos/cath.fa"),
        help="Positive FASTA file (high-entropy candidates).",
    )
    parser.add_argument(
        "--pos-metrics",
        type=Path,
        default=Path("data/pos/cath.metrics.csv"),
        help="Metrics CSV for positives (must include seq_id and sequence).",
    )
    parser.add_argument(
        "--neg-fasta",
        type=Path,
        default=Path("data/neg/esm3_neg.fasta"),
        help="Negative FASTA file (low-entropy candidates).",
    )
    parser.add_argument(
        "--neg-metrics",
        type=Path,
        default=Path("data/neg/esm3_neg.metrics.csv"),
        help="Metrics CSV for negatives (must include seq_id and sequence).",
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/esm3.yaml"),
        help="Model config YAML used to instantiate the backend (e.g., configs/models/esm2.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visualization/probe"),
        help="Directory to store plots and CSV summaries.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for activation extraction.")
    parser.add_argument(
        "--backend-dtype",
        type=str,
        default=None,
        help="Optional dtype override for backend (e.g., float32, float16, bfloat16).",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable backend autocast if the init config supports 'torch_autocast'.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "last_nonpad"],
        default=None,
        help="Optional override for backend pooling mode (defaults to backend config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for activation extraction.",
    )

    parser.add_argument(
        "--entropy-hi",
        type=float,
        default=0.90,
        help="Minimum entropy_norm for positives.",
    )
    parser.add_argument(
        "--entropy-lo",
        type=float,
        default=0.80,
        help="Maximum entropy_norm for negatives.",
    )
    parser.add_argument(
        "--plddt-high",
        type=float,
        default=85.0,
        help="Minimum pLDDT filter for both sides (85.0 by default to mirror dataset cfg).",
    )
    parser.add_argument("--min-len", type=int, default=50, help="Minimum sequence length to keep.")
    parser.add_argument(
        "--max-len", type=int, default=1024, help="Maximum sequence length to keep."
    )
    parser.add_argument(
        "--max-per-side",
        type=int,
        default=1000,
        help="Optional cap on examples per class after filtering/balancing.",
    )
    parser.add_argument(
        "--opt-method",
        type=str,
        default="pareto",
        choices=["simple", "random", "composite", "pareto"],
        help="PosNegProvider opt.method (default pareto, matching provided dataset config).",
    )
    parser.add_argument(
        "--target-per-side",
        type=int,
        default=1000,
        help="Target examples per side for provider balancing (opt.target_per_side).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Train/validation split fraction for the probe (per layer).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and splits.")

    parser.add_argument("--probe-epochs", type=int, default=200, help="Probe training epochs per layer.")
    parser.add_argument("--probe-lr", type=float, default=1e-2, help="Probe learning rate.")
    parser.add_argument("--probe-weight-decay", type=float, default=0.0, help="Probe weight decay.")
    parser.add_argument(
        "--probe-device",
        type=str,
        default="cuda",
        help="Device for the lightweight probe (CPU recommended).",
    )

    return parser.parse_args()


def _sample_pairs(pairs: list[tuple[str, str]], k: int, seed: int) -> list[tuple[str, str]]:
    if k is None or k <= 0 or len(pairs) <= k:
        return pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=k, replace=False)
    return [pairs[i] for i in idx]


def _build_posneg_dataset(args: argparse.Namespace, cache_dir: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    cfg = {
        "pos_fasta": args.pos_fasta,
        "pos_metrics": args.pos_metrics,
        "neg_fasta": args.neg_fasta,
        "neg_metrics": args.neg_metrics,
        "seed": args.seed,
        "filter": {
            "ent_hi": args.entropy_hi,
            "ent_lo": args.entropy_lo,
            "plddt_hi": args.plddt_high,
            "min_len": args.min_len,
            "max_len": args.max_len,
        },
        "opt": {
            "method": args.opt_method,
            "target_per_side": args.target_per_side,
        },
        "metrics": {
            "repetition": ("entropy_norm",),
            "utility": ("ptm",),
            "entropy": ("entropy_norm", "H_norm"),
            "plddt": ("plddt_mean_0_100", "plddt", "plddt_mean_01"),
        },
    }

    provider = PosNegProvider(**cfg)
    provider.build(cache_dir)

    pos_pairs = list(provider.iter_pos())
    neg_pairs = list(provider.iter_neg())

    pos_sel = _sample_pairs(pos_pairs, args.max_per_side, args.seed)
    neg_sel = _sample_pairs(neg_pairs, args.max_per_side, args.seed + 1)
    if not pos_sel or not neg_sel:
        raise RuntimeError("Filtered dataset is empty; adjust thresholds or input paths.")
    return pos_sel, neg_sel


def _load_backend(
    cfg_path: Path,
    device: str,
    pooling_override: str | None,
    dtype_override: str | None,
    disable_autocast: bool,
):
    cfg = OmegaConf.load(cfg_path)
    if "backend" not in cfg or "name" not in cfg:
        raise ValueError(f"Model config must define 'name' and 'backend': {cfg_path}")

    # Override runtime-dependent fields for standalone use.
    cfg.backend.device = device
    if pooling_override is not None:
        cfg.backend.default_pooling = pooling_override
    if dtype_override is not None:
        cfg.backend.dtype = dtype_override

    backend_cfg = BackendConfig(**OmegaConf.to_container(cfg.backend, resolve=True))

    params = OmegaConf.to_container(cfg.get("params", {}), resolve=True) or {}
    def _strip_hydra_keys(obj):
        if not isinstance(obj, dict):
            return obj
        return {k: v for k, v in obj.items() if not str(k).startswith("_")}

    def _safe_to_container(node):
        if node is None:
            return {}
        try:
            return OmegaConf.to_container(node, resolve=True) or {}
        except ValueError:
            return {}

    init_cfg = _strip_hydra_keys(_safe_to_container(cfg.get("init")))
    gen_cfg = _strip_hydra_keys(_safe_to_container(cfg.get("generation")))

    backend_cls = get_model_class(str(cfg.name))
    sig = inspect.signature(backend_cls.__init__)
    accepted = set(sig.parameters)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    kwargs = {k: v for k, v in params.items() if k in accepted}
    if accepts_var_kwargs:
        for k, v in params.items():
            if k not in kwargs:
                kwargs[k] = v
    if disable_autocast and "torch_autocast" in init_cfg:
        init_cfg["torch_autocast"] = False
    if "init_cfg" in accepted and init_cfg:
        kwargs["init_cfg"] = init_cfg
    if "gen_cfg" in accepted and gen_cfg:
        kwargs["gen_cfg"] = gen_cfg

    backend = backend_cls(backend_cfg=backend_cfg, **kwargs)
    backend.load()
    return backend


def _activations(
    backend,
    sequences: Sequence[str],
    layers: Sequence[int],
    *,
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(sequences), batch_size),
        desc="Extracting activations",
        unit="seq",
    ):
        end = min(len(sequences), start + batch_size)
        chunk = sequences[start:end]
        hidden = backend.activations(chunk, layers=layers, batch_size=batch_size)
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach().cpu().numpy()
        batches.append(np.asarray(hidden, dtype=np.float32))
    return np.concatenate(batches, axis=0) if batches else np.empty((0, len(layers), 0), dtype=np.float32)


def _train_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> tuple[float, float]:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {X.shape}")
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to evaluate a probe.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_n = int(max(1, min(n - 1, round(train_frac * n))))
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]

    x_train = torch.from_numpy(X[train_idx]).float().to(device)
    y_train = torch.from_numpy(y[train_idx]).float().to(device)
    x_test = torch.from_numpy(X[test_idx]).float().to(device)
    y_test = torch.from_numpy(y[test_idx]).float().to(device)

    model = torch.nn.Linear(X.shape[1], 1, bias=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(-1)
        loss = criterion(logits, y_train)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y_test.cpu().numpy()

    acc = float(((probs >= 0.5).astype(float) == y_true).mean())
    auc = _fast_auc(y_true, probs)
    return acc, auc


def _fast_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0.0 or neg == 0.0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    rank_sum = float(np.sum(ranks[y == 1]))
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _evaluate_layers(
    hidden: np.ndarray,
    labels: np.ndarray,
    *,
    train_frac: float,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> list[dict[str, float]]:
    num_layers = hidden.shape[1]
    results: list[dict[str, float]] = []
    for layer in tqdm(range(num_layers), desc="Evaluating layers"):
        feats = hidden[:, layer, :]
        acc, auc = _train_probe(
            feats,
            labels,
            train_frac=train_frac,
            seed=seed + layer,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        results.append({"layer": int(layer), "accuracy": float(acc), "auc": float(auc)})
    return results


def _write_csv(rows: Iterable[dict], out_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(results: list[dict[str, float]], out_path: Path, out_pdf: Path | None = None) -> None:
    layers = [r["layer"] for r in results]
    acc = [r["accuracy"] for r in results]
    auc = [r["auc"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, acc, marker="o", label="Accuracy")
    ax.plot(layers, auc, marker="s", label="AUC")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Probe performance")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache = args.output_dir / "posneg_dataset"
    dataset_cache.mkdir(parents=True, exist_ok=True)

    pos_pairs, neg_pairs = _build_posneg_dataset(args, dataset_cache)
    print(f"pos_pairs: {len(pos_pairs)}")
    print(f"neg_pairs: {len(neg_pairs)}")
    sequences = [s for _, s in pos_pairs] + [s for _, s in neg_pairs]
    labels = np.concatenate([
        np.ones(len(pos_pairs), dtype=np.float32),
        np.zeros(len(neg_pairs), dtype=np.float32),
    ])

    backend = _load_backend(
        args.model_config,
        args.device,
        args.pooling,
        args.backend_dtype,
        bool(args.disable_autocast),
    )
    print(backend.model)
    layers = list(range(len(backend.layers)))
    hidden = _activations(
        backend,
        sequences,
        layers=layers,
        batch_size=int(args.batch_size),
    )

    results = _evaluate_layers(
        hidden,
        labels,
        train_frac=float(args.train_frac),
        seed=int(args.seed),
        epochs=int(args.probe_epochs),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
        device=str(args.probe_device),
    )

    summary = {
        "num_layers": len(results),
        "num_pos": len(pos_pairs),
        "num_neg": len(neg_pairs),
        "model_config": str(args.model_config),
        "batch_size": int(args.batch_size),
    }

    metrics_csv = args.output_dir / "layer_probe_metrics.csv"
    plot_path = args.output_dir / "layer_probe_metrics.png"
    plot_pdf_path = plot_path.with_suffix(".pdf")
    summary_path = args.output_dir / "probe_run_summary.json"

    _write_csv(results, metrics_csv)
    _plot(results, plot_path, plot_pdf_path)
    summary["results_path"] = str(metrics_csv)
    summary["plot_path"] = str(plot_path)
    summary["plot_pdf_path"] = str(plot_pdf_path)
    summary["layers"] = [r["layer"] for r in results]
    summary["accuracy"] = [r["accuracy"] for r in results]
    summary["auc"] = [r["auc"] for r in results]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
