"""Visualize KDEs for repetition metrics comparing natural proteins vs PLM generations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Source CSVs for each dataset.
DATASET_SOURCES: dict[str, Path] = {
    "CATH": Path("data/pos/cath.metrics.csv"),
    "UniRef50": Path("data/pos/uniprot.metrics.csv"),
    "SCOP": Path("data/pos/scop.metrics.csv"),
    "ESM3": Path("data/neg/esm3_neg.metrics.csv"),
    "ProtGPT2": Path("data/neg/protgpt2_neg.metrics.csv"),
}

# Columns corresponding to metrics of interest.
METRICS = {
    "Hnorm": "entropy_norm",
    "Distinct-2": "distinct2",
    "Distinct-3": "distinct3",
    "Rhpoly": "H_poly_k4",
    # "pLDDT": "plddt",
    # "pTM": "ptm",
}

NATURAL_DATASETS = ("CATH", "UniRef50", "SCOP")
PLM_DATASETS = ("ESM3", "ProtGPT2")

DATASET_COLORS = {
    "CATH": "#0a0a0a",
    "UniRef50": "#4a4a4a",
    "SCOP": "#8b8b8b",
    "ESM3": "#ac1b1b",
    "ProtGPT2": "#d94f4f",
}

LINE_STYLES = {
    "CATH": "-",
    "UniRef50": "-",
    "SCOP": "-",
    "ESM3": "--",
    "ProtGPT2": "-.",
}

GRID_POINTS = 256
JITTER_SCALE = 1e-3
BANDWIDTH_SCALE = 1.2
FILL_ALPHA = 0.18
LINE_WIDTH = 1.4
MEDIAN_LINESTYLE = ":"
MEDIAN_LINEWIDTH = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize kernel density estimates for repetition-oriented metrics comparing "
            "natural protein datasets (CATH, UniRef50, SCOP) against PLM generations "
            "(ESM3, ProtGPT2). Relative densities are scaled to max=1."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/visualization/kde_metric_comparison.png"),
        help="Path to save the resulting figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Resolution for the saved figure.",
    )
    return parser.parse_args()


def load_metric_values() -> dict[str, dict[str, np.ndarray]]:
    dataset_metrics: dict[str, dict[str, np.ndarray]] = {}

    for dataset, csv_path in DATASET_SOURCES.items():
        resolved = csv_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{dataset} CSV not found: {resolved}")

        with resolved.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{resolved} is missing a header row.")

            missing = [column for column in METRICS.values() if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"{dataset} CSV missing required columns: {', '.join(missing)}")

            collected: dict[str, list[float]] = {metric: [] for metric in METRICS}
            for row in reader:
                for metric_name, column in METRICS.items():
                    raw_value = row.get(column, "").strip()
                    if raw_value == "":
                        continue
                    try:
                        collected[metric_name].append(float(raw_value))
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric value for {column} in {dataset}: {raw_value}"
                        ) from exc

        dataset_metrics[dataset] = {}
        for metric_name, values in collected.items():
            if not values:
                raise ValueError(f"{dataset} has no valid values for {METRICS[metric_name]}.")
            dataset_metrics[dataset][metric_name] = np.asarray(values, dtype=float)

    return dataset_metrics


def _ensure_variance(values: np.ndarray) -> np.ndarray:
    """Add structured jitter when variance is zero to keep gaussian_kde stable."""

    if np.allclose(values, values[0]):
        jitter = np.linspace(-0.5, 0.5, num=values.size, dtype=float)
        scale = max(abs(values[0]), 1.0) * JITTER_SCALE
        values = values + jitter * scale
    return values


def compute_kde_curves(
    dataset_metrics: dict[str, dict[str, np.ndarray]],
    metric_name: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a shared grid and per-dataset relative KDE densities."""

    values_per_dataset = {ds: dataset_metrics[ds][metric_name] for ds in DATASET_SOURCES}
    global_min = min(values.min() for values in values_per_dataset.values())
    global_max = max(values.max() for values in values_per_dataset.values())
    span = global_max - global_min
    padding = 0.05 * span if span > 0 else 1.0
    grid = np.linspace(global_min - padding, global_max + padding, GRID_POINTS)

    raw_densities: dict[str, np.ndarray] = {}

    def _scaled_bw(kde_obj):
        return BANDWIDTH_SCALE * kde_obj.scotts_factor()

    for dataset, values in values_per_dataset.items():
        prepared = _ensure_variance(values)
        kde = gaussian_kde(prepared, bw_method=_scaled_bw)
        raw_densities[dataset] = kde(grid)

    scaled: dict[str, np.ndarray] = {}
    for dataset, density in raw_densities.items():
        max_density = float(density.max())
        scaled[dataset] = density / max_density if max_density > 0 else density

    return grid, scaled


def apply_nature_style():
    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "font.size": 8,
            "axes.edgecolor": "#0f0f0f",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#0f0f0f",
            "xtick.color": "#0f0f0f",
            "ytick.color": "#0f0f0f",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_kdes(
    dataset_metrics: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    dpi: int,
) -> None:
    apply_nature_style()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8.5, 4.2),
        sharey=True,
        constrained_layout=False,
    )

    axes = axes.ravel()
    legend_order = list(NATURAL_DATASETS) + list(PLM_DATASETS)
    legend_handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            lw=LINE_WIDTH,
            linestyle=LINE_STYLES[dataset],
            label=dataset,
        )
        for dataset in legend_order
    ]

    for idx, (metric_label, _) in enumerate(METRICS.items()):
        ax = axes[idx]
        grid, densities = compute_kde_curves(dataset_metrics, metric_label)

        for dataset in NATURAL_DATASETS + PLM_DATASETS:
            color = DATASET_COLORS[dataset]
            ax.plot(
                grid,
                densities[dataset],
                color=color,
                linewidth=LINE_WIDTH,
                linestyle=LINE_STYLES[dataset],
            )
            ax.fill_between(
                grid,
                densities[dataset],
                color=color,
                alpha=FILL_ALPHA,
                linewidth=0,
            )
            median_value = float(np.median(dataset_metrics[dataset][metric_label]))
            ax.axvline(
                median_value,
                color=color,
                linestyle=MEDIAN_LINESTYLE,
                linewidth=MEDIAN_LINEWIDTH,
                alpha=0.9,
            )

        ax.set_title(metric_label, loc="left", fontweight="bold", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis="both", length=3, width=0.8, pad=2)

        if idx // 3 == 1:
            ax.set_xlabel(METRICS[metric_label])
        if idx % 3 == 0:
            ax.set_ylabel("Relative density")

    fig.text(0.5, 0.97, "KDE of repetition metrics (max density = 1)", ha="center", fontsize=10)
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )
    legend._legend_box.align = "left"  # type: ignore[attr-defined]
    fig.text(
        0.5,
        0.93,
        "Natural proteins (black/gray) vs PLM generations (red)",
        ha="center",
        va="center",
        fontsize=7,
        color="#2b2b2b",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_metrics = load_metric_values()
    plot_kdes(dataset_metrics, args.output, args.dpi)


if __name__ == "__main__":
    main()
