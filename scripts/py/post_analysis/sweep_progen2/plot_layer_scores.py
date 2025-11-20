#!/usr/bin/env python3
"""Visualize layer sweeps for Probe/UCCS steering on ProGen2."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import re

RUN_PATTERN = re.compile(
    r"seed_(?P<seed>\d+)_method_(?P<method>.+)_dataset_(?P<dataset>.+)$"
)
TARGET_METHODS = ("probe_layer", "uccs_layer")
METHOD_LABELS = {
    "probe_layer": "Probe Steering",
    "uccs_layer": "UCCS",
}
CONDITION_CHOICES = ("both", "conditional", "unconditional")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot layer-level repetition/utility scores for Probe & UCCS sweeps."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing seed_*_method_*_dataset_* experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output figures (default: <root>/figures).",
    )
    parser.add_argument(
        "--condition",
        choices=CONDITION_CHOICES,
        default="both",
        help="Which generation condition(s) to aggregate (default: both).",
    )
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def harmonic_mean(rep: float | None, util: float | None) -> float:
    if rep is None or util is None:
        return float("nan")
    if not (math.isfinite(rep) and math.isfinite(util)):
        return float("nan")
    if rep <= 0 or util <= 0:
        return float("nan")
    return 2 * rep * util / (rep + util)


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [val for val in values if val == val]
    if not vals:
        return float("nan")
    return statistics.fmean(vals)


def parse_method_layer(method: str) -> tuple[str, int] | None:
    for base in TARGET_METHODS:
        if method.startswith(base):
            suffix = method[len(base) :]
            try:
                return base, int(suffix)
            except ValueError:
                return None
    return None


def gather_metrics(root: Path, condition: str):
    metrics_store: dict[str, defaultdict[int, dict[str, list[float]]]] = {
        method: defaultdict(lambda: {"R": [], "U": [], "H": []})
        for method in TARGET_METHODS
    }

    run_dirs = sorted(root.glob("seed_*_method_*_dataset_*"))
    if not run_dirs:
        raise SystemExit(f"No run directories found under {root}")

    condition_map = {
        "conditional": ("conditional",),
        "unconditional": ("unconditional",),
        "both": ("conditional", "unconditional"),
    }[condition]

    for run_dir in run_dirs:
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if not match:
            continue
        parsed = parse_method_layer(match.group("method"))
        if parsed is None:
            continue
        method_base, layer_idx = parsed

        files = {
            "conditional": run_dir / "prefix.steer.summary.json",
            "unconditional": run_dir / "uncond.steer.summary.json",
        }
        for cond in condition_map:
            summary = read_summary(files[cond])
            rep = summary.get("repetition_score")
            util = summary.get("utility_score")
            if rep is None or util is None:
                continue
            entry = metrics_store[method_base][layer_idx]
            entry["R"].append(float(rep))
            entry["U"].append(float(util))
            entry["H"].append(harmonic_mean(float(rep), float(util)))

    return metrics_store


def plot_metric(
    metrics_store,
    metric_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    for method in TARGET_METHODS:
        layer_map = metrics_store.get(method, {})
        if not layer_map:
            continue
        xs = sorted(layer_map)
        ys = [mean_or_nan(layer_map[layer][metric_key]) for layer in xs]
        if not xs or all(math.isnan(val) for val in ys):
            continue
        ax.plot(xs, ys, marker="o", label=METHOD_LABELS.get(method, method))
        plotted = True
    if not plotted:
        print(f"[warn] No data for {ylabel}, skipping plot.")
        plt.close(fig)
        return
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.root / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_store = gather_metrics(args.root, args.condition)

    plot_metric(
        metrics_store,
        "R",
        "Repetition Score",
        output_dir / "layers_vs_repetition.png",
    )
    plot_metric(
        metrics_store,
        "U",
        "Biological Utility",
        output_dir / "layers_vs_utility.png",
    )
    plot_metric(
        metrics_store,
        "H",
        "Harmonic Mean",
        output_dir / "layers_vs_hmean.png",
    )
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
