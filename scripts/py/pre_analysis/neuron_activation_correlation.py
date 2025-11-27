"""Neuron-level correlation between activations and repetition metrics.

This script mirrors the dataset/activation plumbing from
``mechanistic_probe.py`` but instead of training probes, it computes a
Pearson correlation coefficient for every neuron (per layer dimension)
against a repetition-focused metric (``rep_metric`` by default). It
emits CSV summaries and lightweight plots to highlight which layers and
neurons track repetition the most, including per-layer correlation
histograms to compare distributional shifts across depth.
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
            "Compute neuron-wise Pearson correlations between pooled layer "
            "activations and a repetition metric (rep_metric by default)."
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
        help=(
            "Model config YAML used to instantiate the backend "
            "(e.g., configs/models/esm2.yaml)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visualization/neuron_correlation"),
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and splits.")

    parser.add_argument(
        "--metric-key",
        type=str,
        default="rep_metric",
        help="Record key to correlate against (e.g., rep_metric or entropy).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Number of strongest neurons to visualize in the bar plot.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=40,
        help="Number of bins for per-layer correlation histograms.",
    )
    parser.add_argument(
        "--hist-cols",
        type=int,
        default=6,
        help="Number of subplot columns for per-layer histograms (default 6).",
    )

    return parser.parse_args()


def _sample_items(items: list[dict], k: int, seed: int) -> list[dict]:
    if k is None or k <= 0 or len(items) <= k:
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in idx]


def _build_posneg_dataset(args: argparse.Namespace, cache_dir: Path) -> list[dict]:
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

    pos_items = [r for r in provider.items() if r.get("source") == "pos"]
    neg_items = [r for r in provider.items() if r.get("source") == "neg"]

    pos_sel = _sample_items(pos_items, args.max_per_side, args.seed)
    neg_sel = _sample_items(neg_items, args.max_per_side, args.seed + 1)
    if not pos_sel or not neg_sel:
        raise RuntimeError("Filtered dataset is empty; adjust thresholds or input paths.")
    return pos_sel + neg_sel


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


def _extract_targets(
    records: Sequence[dict],
    metric_key: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    sequences: list[str] = []
    metric_values: list[float] = []
    labels: list[float] = []

    for rec in records:
        seq = rec.get("sequence")
        if not seq:
            continue
        raw_val = rec.get(metric_key)
        if raw_val is None:
            alt = rec.get("rep_metric") or rec.get("entropy")
            raw_val = alt
        try:
            val = float(raw_val)
        except Exception:
            continue
        if not np.isfinite(val):
            continue

        sequences.append(seq)
        metric_values.append(val)
        labels.append(1.0 if rec.get("source") == "pos" else 0.0)

    if not sequences:
        raise RuntimeError(f"No usable sequences with metric '{metric_key}'.")

    return sequences, np.asarray(metric_values, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _compute_correlations(hidden: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden shape (N, L, D); got {hidden.shape}")
    if hidden.shape[0] != scores.shape[0]:
        raise ValueError("Hidden activations and scores must align along axis 0.")

    flat_hidden = hidden.reshape(hidden.shape[0], -1).astype(np.float64)
    centered_hidden = flat_hidden - flat_hidden.mean(axis=0)

    centered_scores = scores.astype(np.float64) - float(scores.mean())
    score_power = float(np.sum(centered_scores ** 2))
    if score_power <= 0:
        raise ValueError("Scores have zero variance; correlation is undefined.")

    numerator = np.sum(centered_hidden * centered_scores[:, None], axis=0)
    denom = np.sqrt(np.sum(centered_hidden ** 2, axis=0) * score_power)
    denom = np.where(denom == 0.0, np.nan, denom)
    corr = numerator / denom
    return corr.reshape(hidden.shape[1], hidden.shape[2])


def _top_neurons(corr: np.ndarray, k: int) -> list[dict[str, float]]:
    flat = corr.reshape(-1)
    abs_flat = np.abs(flat)
    abs_flat = np.nan_to_num(abs_flat, nan=-np.inf)
    k = min(k, flat.size)
    top_idx = np.argsort(-abs_flat)[:k]

    rows: list[dict[str, float]] = []
    width = corr.shape[1]
    for idx in top_idx:
        layer = int(idx // width)
        neuron = int(idx % width)
        value = float(flat[idx])
        rows.append(
            {
                "layer": layer,
                "neuron": neuron,
                "correlation": value,
                "abs_correlation": abs(value),
            }
        )
    return rows


def _layer_stats(corr: np.ndarray) -> list[dict[str, float]]:
    stats: list[dict[str, float]] = []
    for layer in range(corr.shape[0]):
        vals = corr[layer]
        stats.append(
            {
                "layer": int(layer),
                "mean_abs_corr": float(np.nanmean(np.abs(vals))),
                "max_abs_corr": float(np.nanmax(np.abs(vals))),
            }
        )
    return stats


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


def _plot_layer_stats(
    layer_rows: list[dict[str, float]],
    out_path: Path,
    out_pdf: Path | None = None,
) -> None:
    layers = [r["layer"] for r in layer_rows]
    mean_abs = [r["mean_abs_corr"] for r in layer_rows]
    max_abs = [r["max_abs_corr"] for r in layer_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, mean_abs, marker="o", label="Mean |r|")
    ax.plot(layers, max_abs, marker="s", label="Max |r|")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Correlation with repetition")
    ax.set_ylim(0.0, max(1.0, max(max_abs) * 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def _plot_top_neurons(
    rows: list[dict[str, float]],
    out_path: Path,
    out_pdf: Path | None = None,
) -> None:
    if not rows:
        return
    labels = [f"L{r['layer']} N{r['neuron']}" for r in rows]
    values = [r["correlation"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.25)))
    y_pos = np.arange(len(rows))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in values]
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Correlation with repetition")
    ax.set_title(f"Top {len(rows)} neurons by |correlation|")
    ax.axvline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def _plot_layer_distributions(
    corr: np.ndarray,
    out_path: Path,
    *,
    bins: int = 40,
    cols: int = 5,
    out_pdf: Path | None = None,
) -> None:
    if corr.ndim != 2:
        return

    num_layers, width = corr.shape
    cols = max(1, cols)
    rows = int(np.ceil(num_layers / cols))
    bin_edges = np.linspace(-1.0, 1.0, bins + 1)

    fig_width = 3.6 * cols
    fig_height = 2.8 * rows
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)

    for layer in range(num_layers):
        r = layer // cols
        c = layer % cols
        ax = axes[r, c]
        vals = corr[layer]
        ax.hist(vals, bins=bin_edges, color="tab:blue", alpha=0.7)
        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlim(-1.0, 1.0)
        ax.tick_params(labelsize=10)
    # Hide any unused subplots.
    for extra in range(num_layers, rows * cols):
        r = extra // cols
        c = extra % cols
        axes[r, c].axis("off")

    fig.text(0.5, 0.04, "Correlation with repetition", ha="center", fontsize=22)
    fig.text(0.04, 0.5, "Neuron count", va="center", rotation="vertical", fontsize=22)
    fig.tight_layout(rect=(0.06, 0.06, 1.0, 1.0))
    fig.savefig(out_path, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache = args.output_dir / "posneg_dataset"
    dataset_cache.mkdir(parents=True, exist_ok=True)

    records = _build_posneg_dataset(args, dataset_cache)
    print(f"pos_records: {sum(1 for r in records if r.get('source') == 'pos')}")
    print(f"neg_records: {sum(1 for r in records if r.get('source') == 'neg')}")

    sequences, scores, labels = _extract_targets(records, args.metric_key)

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

    corr = _compute_correlations(hidden, scores)
    layer_rows = _layer_stats(corr)
    top_rows = _top_neurons(corr, int(args.top_k))

    correlation_csv = args.output_dir / "neuron_correlation.csv"
    layer_csv = args.output_dir / "layer_correlation_summary.csv"
    top_csv = args.output_dir / "top_neurons.csv"
    layer_plot = args.output_dir / "layer_correlation.png"
    top_plot = args.output_dir / "top_neurons.png"
    hist_plot = args.output_dir / "layer_correlation_hist.png"
    layer_plot_pdf = layer_plot.with_suffix(".pdf")
    top_plot_pdf = top_plot.with_suffix(".pdf")
    hist_plot_pdf = hist_plot.with_suffix(".pdf")
    summary_path = args.output_dir / "neuron_correlation_summary.json"
    corr_npy = args.output_dir / "neuron_correlation.npy"

    _write_csv(
        (
            {
                "layer": int(i // corr.shape[1]),
                "neuron": int(i % corr.shape[1]),
                "correlation": float(val),
                "abs_correlation": float(abs(val)),
            }
            for i, val in enumerate(corr.reshape(-1))
        ),
        correlation_csv,
    )
    _write_csv(layer_rows, layer_csv)
    _write_csv(top_rows, top_csv)

    np.save(corr_npy, corr)
    _plot_layer_stats(layer_rows, layer_plot, layer_plot_pdf)
    _plot_top_neurons(top_rows, top_plot, top_plot_pdf)
    _plot_layer_distributions(
        corr,
        hist_plot,
        bins=int(args.hist_bins),
        cols=int(args.hist_cols),
        out_pdf=hist_plot_pdf,
    )

    summary = {
        "num_layers": corr.shape[0],
        "hidden_size": corr.shape[1] if corr.ndim == 2 else None,
        "num_sequences": len(sequences),
        "num_pos": int(labels.sum()),
        "num_neg": int(len(labels) - labels.sum()),
        "metric_key": args.metric_key,
        "output_csv": str(correlation_csv),
        "layer_summary": str(layer_csv),
        "top_neurons": str(top_csv),
        "layer_plot": str(layer_plot),
        "top_plot": str(top_plot),
        "layer_plot_pdf": str(layer_plot_pdf),
        "top_plot_pdf": str(top_plot_pdf),
        "layer_hist_plot_pdf": str(hist_plot_pdf),
        "layer_hist_plot": str(hist_plot),
        "correlation_npy": str(corr_npy),
        "model_config": str(args.model_config),
        "batch_size": int(args.batch_size),
        "pooling": args.pooling,
        "backend_dtype": args.backend_dtype,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
