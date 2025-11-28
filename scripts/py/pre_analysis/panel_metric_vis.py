"""Produce Nature/Science style panels for KDE + JS divergence for key metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

DATASET_SOURCES: dict[str, Path] = {
    "CATH": Path("data/pos/cath.metrics.csv"),
    "UniRef50": Path("data/pos/uniprot.metrics.csv"),
    "SCOP": Path("data/pos/scop.metrics.csv"),
    "ESM3": Path("data/neg/esm3_neg.metrics.csv"),
    "ProtGPT2": Path("data/neg/protgpt2_neg.metrics.csv"),
}

METRICS = {
    "Hnorm": "entropy_norm",
    "Distinct-2": "distinct2",
    "Distinct-3": "distinct3",
    "Rhpoly": "H_poly_k4",
    "pLDDT": "plddt",
    "pTM": "ptm",
}

REPETITION_PANEL = ["Hnorm", "Distinct-2", "Distinct-3", "Rhpoly"]
STRUCTURE_PANEL = ["pLDDT", "pTM"]

NATURAL_DATASETS = ("CATH", "UniRef50", "SCOP")
PLM_DATASETS = ("ESM3", "ProtGPT2")
DATASET_ORDER = NATURAL_DATASETS + PLM_DATASETS

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
LINE_WIDTH = 1.65
HISTOGRAM_BINS = 64
HISTOGRAM_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two-panel figures: KDE (row 1) and JS divergence heatmaps (row 2) "
            "for repetition and structure-related metrics."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visualization"),
        help="Directory to store the generated PDF panels.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Resolution used when rasterizing to PDF.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Global scaling factor for text (e.g., 1.2 for larger fonts).",
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
                raise ValueError(f"{resolved} missing header row.")

            missing = [column for column in METRICS.values() if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"{dataset} CSV missing required columns: {', '.join(missing)}")

            collected: dict[str, list[float]] = {metric: [] for metric in METRICS}
            for row in reader:
                for metric_label, column in METRICS.items():
                    raw_value = row.get(column, "").strip()
                    if not raw_value:
                        continue
                    try:
                        collected[metric_label].append(float(raw_value))
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric value for {column} in {dataset}: {raw_value}"
                        ) from exc

        dataset_metrics[dataset] = {}
        for metric_label, values in collected.items():
            if not values:
                raise ValueError(f"{dataset} has no valid values for {METRICS[metric_label]}.")
            dataset_metrics[dataset][metric_label] = np.asarray(values, dtype=float)

    return dataset_metrics


def _ensure_variance(values: np.ndarray) -> np.ndarray:
    if np.allclose(values, values[0]):
        jitter = np.linspace(-0.5, 0.5, num=values.size, dtype=float)
        scale = max(abs(values[0]), 1.0) * JITTER_SCALE
        values = values + jitter * scale
    return values


def compute_kde_curves(
    dataset_metrics: dict[str, dict[str, np.ndarray]], metric_label: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values_per_dataset = {ds: dataset_metrics[ds][metric_label] for ds in DATASET_ORDER}
    global_min = min(values.min() for values in values_per_dataset.values())
    global_max = max(values.max() for values in values_per_dataset.values())
    span = global_max - global_min
    padding = 0.05 * span if span > 0 else 1.0
    grid = np.linspace(global_min - padding, global_max + padding, GRID_POINTS)

    def _scaled_bw(kde_obj):
        return BANDWIDTH_SCALE * kde_obj.scotts_factor()

    densities: dict[str, np.ndarray] = {}
    for dataset, values in values_per_dataset.items():
        prepared = _ensure_variance(values)
        kde = gaussian_kde(prepared, bw_method=_scaled_bw)
        density = kde(grid)
        max_density = float(density.max())
        if max_density > 0:
            density /= max_density
        densities[dataset] = density

    return grid, densities


def compute_js_matrix(dataset_metrics: dict[str, dict[str, np.ndarray]], metric_label: str) -> np.ndarray:
    n = len(DATASET_ORDER)
    matrix = np.zeros((n, n), dtype=float)
    for i, dataset_a in enumerate(DATASET_ORDER):
        values_a = dataset_metrics[dataset_a][metric_label]
        for j, dataset_b in enumerate(DATASET_ORDER):
            if j < i:
                matrix[i, j] = matrix[j, i]
                continue
            if i == j:
                matrix[i, j] = 0.0
                continue
            js_val = _js_divergence(values_a, dataset_metrics[dataset_b][metric_label])
            matrix[i, j] = js_val
            matrix[j, i] = js_val
    return matrix


def _js_divergence(values_a: np.ndarray, values_b: np.ndarray) -> float:
    combined = np.concatenate([values_a, values_b])
    bin_edges = np.histogram_bin_edges(combined, bins=HISTOGRAM_BINS)
    hist_a, _ = np.histogram(values_a, bins=bin_edges)
    hist_b, _ = np.histogram(values_b, bins=bin_edges)

    p = hist_a.astype(float) + HISTOGRAM_EPS
    q = hist_b.astype(float) + HISTOGRAM_EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    js_div = 0.5 * (float(np.sum(p * np.log(p / m))) + float(np.sum(q * np.log(q / m))))
    return js_div


def apply_style(font_scale: float):
    base_font = 8 * font_scale
    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "font.size": base_font,
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titlesize": 9 * font_scale,
            "axes.labelsize": 8 * font_scale,
            "xtick.labelsize": 7 * font_scale,
            "ytick.labelsize": 7 * font_scale,
        }
    )


def plot_panel(
    dataset_metrics: dict[str, dict[str, np.ndarray]],
    metric_labels: list[str],
    output_path: Path,
    title: str,
    dpi: int,
    font_scale: float,
) -> None:
    apply_style(font_scale)

    n_cols = len(metric_labels)
    fig = plt.figure(figsize=(3.0 * n_cols, 4.8), dpi=dpi)
    gs = GridSpec(
        2,
        n_cols,
        figure=fig,
        height_ratios=[1.3, 1.35],
        hspace=0.35,
        wspace=0.25,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            lw=LINE_WIDTH,
            linestyle=LINE_STYLES[dataset],
            label=dataset,
        )
        for dataset in DATASET_ORDER
    ]

    js_matrices = {metric: compute_js_matrix(dataset_metrics, metric) for metric in metric_labels}
    js_max = max(float(matrix.max()) for matrix in js_matrices.values())
    js_max = js_max if js_max > 0 else 1.0
    heatmaps = []
    heat_axes = []

    axis_label_size = 8 * font_scale
    tick_label_size = 7 * font_scale
    legend_size = 7 * font_scale
    suptitle_size = 11 * font_scale
    cbar_label_size = 8 * font_scale

    for col, metric_label in enumerate(metric_labels):
        ax_kde = fig.add_subplot(gs[0, col])
        grid, densities = compute_kde_curves(dataset_metrics, metric_label)

        for dataset in DATASET_ORDER:
            color = DATASET_COLORS[dataset]
            ax_kde.plot(
                grid,
                densities[dataset],
                color=color,
                linewidth=LINE_WIDTH,
                linestyle=LINE_STYLES[dataset],
            )
            ax_kde.fill_between(grid, densities[dataset], color=color, alpha=FILL_ALPHA, linewidth=0)

        ax_kde.set_title("")
        ax_kde.set_ylim(0, 1.05)
        ax_kde.set_xlim(grid[0], grid[-1])
        ax_kde.set_yticks([0, 0.5, 1.0])
        ax_kde.tick_params(
            axis="both",
            length=3,
            width=0.8,
            pad=2 * font_scale,
            labelsize=tick_label_size,
        )
        if col == 0:
            ax_kde.set_ylabel("Relative density", fontsize=axis_label_size)
        else:
            ax_kde.set_yticklabels([])
        ax_kde.set_xlabel(metric_label, fontsize=axis_label_size)

        ax_heat = fig.add_subplot(gs[1, col])
        matrix = js_matrices[metric_label]
        im = ax_heat.imshow(matrix, vmin=0, vmax=js_max, cmap="viridis")
        heatmaps.append(im)
        heat_axes.append(ax_heat)
        ax_heat.set_xticks(range(len(DATASET_ORDER)))
        ax_heat.set_yticks(range(len(DATASET_ORDER)))
        if col == 0:
            ax_heat.set_yticklabels(DATASET_ORDER, fontsize=tick_label_size)
            ax_heat.set_ylabel("Dataset A", labelpad=10 * font_scale, fontsize=axis_label_size)
        else:
            ax_heat.set_yticklabels([])
        ax_heat.set_xticklabels(
            DATASET_ORDER,
            rotation=35,
            ha="right",
            fontsize=tick_label_size,
        )
        ax_heat.tick_params(
            axis="both",
            length=2.5,
            width=0.6,
            labelsize=tick_label_size,
            pad=2 * font_scale,
        )
        ax_heat.set_xlabel("Dataset B", labelpad=4 * font_scale, fontsize=axis_label_size)

        for spine in ax_heat.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#111111")

    divider = make_axes_locatable(heat_axes[-1])
    cax = divider.append_axes("right", size="3.5%", pad=0.12)

    cbar = fig.colorbar(heatmaps[-1], cax=cax, orientation="vertical")
    cbar.ax.set_ylabel("Jensen-Shannon divergence", fontsize=cbar_label_size, labelpad=6 * font_scale)
    cbar.ax.tick_params(length=2.0, width=0.6, labelsize=tick_label_size)

    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(DATASET_ORDER),
        frameon=False,
        fontsize=legend_size,
        bbox_to_anchor=(0.5, 0.98),
    )
    legend._legend_box.align = "left"  # type: ignore[attr-defined]

    fig.suptitle(title, fontsize=suptitle_size, fontweight="bold", y=0.995)

    # 调整总体布局参数
    fig.subplots_adjust(left=0.07, right=0.93, top=0.9, bottom=0.08, wspace=0.3, hspace=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_metrics = load_metric_values()
    output_dir = args.output_dir.expanduser().resolve()

    repetition_path = output_dir / "repetition_metrics_panel.pdf"
    structure_path = output_dir / "structure_metrics_panel.pdf"

    plot_panel(
        dataset_metrics,
        REPETITION_PANEL,
        repetition_path,
        title="Repetition metrics: KDE + JS divergence",
        dpi=args.dpi,
        font_scale=args.font_scale,
    )
    plot_panel(
        dataset_metrics,
        STRUCTURE_PANEL,
        structure_path,
        title="Structure utilities: KDE + JS divergence",
        dpi=args.dpi,
        font_scale=args.font_scale,
    )


if __name__ == "__main__":
    main()
