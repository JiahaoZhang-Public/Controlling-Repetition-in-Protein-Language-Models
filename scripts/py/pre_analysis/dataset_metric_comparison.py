"""Compute dataset-wise repetition metric separability statistics and heatmaps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
from scipy.stats import ks_2samp, kurtosis, wasserstein_distance

DATASET_SOURCES: dict[str, Path] = {
    "CATH": Path("data/pos/cath.metrics.csv"),
    "UniRef50": Path("data/pos/uniprot.metrics.csv"),
    "SCOP": Path("data/pos/scop.metrics.csv"),
    "ESM3": Path("data/neg/esm3_neg.metrics.csv"),
    "PROGPT2": Path("data/neg/protgpt2_neg.metrics.csv"),
}

# Columns corresponding to repetition-focused metrics of interest.
METRICS = {
    "Hnorm": "entropy_norm",
    "Distinct-2": "distinct2",
    "Distinct-3": "distinct3",
    "Rhpoly": "H_poly_k4",
}

TAIL_QUANTILE = 0.95
QUANTILE_FOR_SEPARATION = 0.95
HISTOGRAM_BINS = 64
HISTOGRAM_EPS = 1e-12

COMMON_HEATMAP_STATS = [
    "wasserstein_distance",
    "tail_wasserstein_distance",
    "kl_divergence",
    "js_divergence",
    "excess_kurtosis_diff",
    "q95_diff",
]

METRIC_EXTRA_STATS = {
    "Rhpoly": ["roc_auc"],
}


def _stat_keys_for_metric(metric_name: str) -> list[str]:
    return COMMON_HEATMAP_STATS + METRIC_EXTRA_STATS.get(metric_name, [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-wise separability metrics (Wasserstein, KL/JSD, "
            "tail-weighted distances, kurtosis, quantile gaps) for repetition features "
            "and store tabular summaries and heatmaps."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visulization/metrics"),
        help="Directory to store CSV summaries and heatmaps.",
    )
    return parser.parse_args()


def load_metric_values() -> dict[str, dict[str, np.ndarray]]:
    """Load the relevant columns for each dataset."""

    dataset_metrics: dict[str, dict[str, np.ndarray]] = {}

    for dataset, csv_path in DATASET_SOURCES.items():
        resolved = csv_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{dataset} CSV not found: {resolved}")

        with resolved.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{resolved} is missing a header row.")

            missing = [col for col in METRICS.values() if col not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"{dataset} CSV missing required columns: {', '.join(missing)}"
                )

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
            array = np.asarray(values, dtype=float)
            if array.size == 0:
                raise ValueError(
                    f"{dataset} does not contain non-null values for {METRICS[metric_name]}."
                )
            dataset_metrics[dataset][metric_name] = array

    return dataset_metrics


def compute_statistics(
    dataset_metrics: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, float]], dict[str, dict[str, np.ndarray]], list[str]]:
    """Compute pairwise statistics for each metric."""

    dataset_names = list(DATASET_SOURCES.keys())
    index_lookup = {name: idx for idx, name in enumerate(dataset_names)}
    records: list[dict[str, float]] = []
    matrices = {
        metric_name: {
            stat_key: np.zeros((len(dataset_names), len(dataset_names)), dtype=float)
            for stat_key in _stat_keys_for_metric(metric_name)
        }
        for metric_name in METRICS
    }

    for metric_name in METRICS:
        stat_keys = _stat_keys_for_metric(metric_name)
        for dataset_a in dataset_names:
            values_a = dataset_metrics[dataset_a][metric_name]
            kurt_a = _safe_kurtosis(values_a)
            q95_a = float(np.quantile(values_a, QUANTILE_FOR_SEPARATION))
            for dataset_b in dataset_names:
                values_b = dataset_metrics[dataset_b][metric_name]
                kurt_b = _safe_kurtosis(values_b)
                q95_b = float(np.quantile(values_b, QUANTILE_FOR_SEPARATION))

                roc_auc = None
                youden_threshold = None
                youden_sensitivity = None
                youden_specificity = None
                youden_j = None

                if dataset_a == dataset_b:
                    wasserstein = 0.0
                    tail_wass = 0.0
                    ks_stat = 0.0
                    p_value = 1.0
                    kl_div = 0.0
                    js_div = 0.0
                    kurtosis_diff = 0.0
                    q95_diff = 0.0
                    if "roc_auc" in stat_keys:
                        roc_auc = 0.5
                else:
                    wasserstein = float(wasserstein_distance(values_a, values_b))
                    tail_wass = float(
                        wasserstein_distance(
                            _tail_values(values_a, TAIL_QUANTILE),
                            _tail_values(values_b, TAIL_QUANTILE),
                        )
                    )
                    ks_stat, p_value = ks_2samp(values_a, values_b, alternative="two-sided")
                    kl_div, js_div = _symmetric_divergences(values_a, values_b)
                    kurtosis_diff = float(abs(kurt_a - kurt_b))
                    q95_diff = float(abs(q95_a - q95_b))
                    if "roc_auc" in stat_keys:
                        roc = _compute_roc_metrics(values_a, values_b)
                        (
                            roc_auc,
                            youden_threshold,
                            youden_sensitivity,
                            youden_specificity,
                            youden_j,
                        ) = roc

                i = index_lookup[dataset_a]
                j = index_lookup[dataset_b]

                matrices[metric_name]["wasserstein_distance"][i, j] = wasserstein
                matrices[metric_name]["tail_wasserstein_distance"][i, j] = tail_wass
                matrices[metric_name]["kl_divergence"][i, j] = kl_div
                matrices[metric_name]["js_divergence"][i, j] = js_div
                matrices[metric_name]["excess_kurtosis_diff"][i, j] = kurtosis_diff
                matrices[metric_name]["q95_diff"][i, j] = q95_diff
                if "roc_auc" in stat_keys:
                    matrices[metric_name]["roc_auc"][i, j] = (
                        roc_auc if roc_auc is not None else np.nan
                    )

                records.append(
                    {
                        "metric": metric_name,
                        "dataset_a": dataset_a,
                        "dataset_b": dataset_b,
                        "wasserstein_distance": wasserstein,
                        "tail_wasserstein_distance": tail_wass,
                        "kl_divergence": kl_div,
                        "js_divergence": js_div,
                        "excess_kurtosis_diff": kurtosis_diff,
                        "q95_diff": q95_diff,
                        "ks_statistic": float(ks_stat),
                        "ks_pvalue": float(p_value),
                        "roc_auc": roc_auc,
                        "youden_threshold": youden_threshold,
                        "youden_sensitivity": youden_sensitivity,
                        "youden_specificity": youden_specificity,
                        "youden_j": youden_j,
                    }
                )

    return records, matrices, dataset_names


def save_summary(records: list[dict[str, float]], output_dir: Path) -> Path:
    csv_path = output_dir / "dataset_metric_comparison.csv"
    fieldnames = [
        "metric",
        "dataset_a",
        "dataset_b",
        "wasserstein_distance",
        "tail_wasserstein_distance",
        "kl_divergence",
        "js_divergence",
        "excess_kurtosis_diff",
        "q95_diff",
        "ks_statistic",
        "ks_pvalue",
        "roc_auc",
        "youden_threshold",
        "youden_sensitivity",
        "youden_specificity",
        "youden_j",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def plot_heatmaps(
    matrices: dict[str, dict[str, np.ndarray]],
    dataset_names: list[str],
    output_dir: Path,
) -> list[Path]:
    saved_paths: list[Path] = []

    _apply_nature_style()
    n = len(dataset_names)
    for metric_name, stat_matrices in matrices.items():
        for stat_name, matrix in stat_matrices.items():
            values = np.asarray(matrix, dtype=float)
            if np.all(np.isnan(values)):
                continue
            plot_values = np.nan_to_num(values, nan=0.0)

            fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
            im = ax.imshow(plot_values, cmap="viridis")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(dataset_names, rotation=45, ha="right")
            ax.set_yticklabels(dataset_names)
            ax.set_title(f"{metric_name} {stat_name.replace('_', ' ').title()}")
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("black")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(length=2.5, width=0.6)
            cbar.set_label(stat_name.replace("_", " "), rotation=90, labelpad=8)

            mean_val = float(np.nanmean(values))
            for i in range(n):
                for j in range(n):
                    value = values[i, j]
                    if np.isnan(value):
                        continue
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value > mean_val else "black",
                        fontsize=8,
                    )

            metric_slug = metric_name.lower().replace(" ", "_").replace("-", "_")
            stat_slug = stat_name.lower()
            heatmap_path = output_dir / f"{metric_slug}_{stat_slug}_heatmap.pdf"
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=300, format="pdf")
            plt.close(fig)
            saved_paths.append(heatmap_path)

    return saved_paths


def _tail_values(values: np.ndarray, quantile: float) -> np.ndarray:
    threshold = np.quantile(values, quantile)
    tail = values[values >= threshold]
    if tail.size == 0:
        tail = np.asarray([values.max()], dtype=float)
    return tail


def _compute_roc_metrics(
    positive_scores: np.ndarray, negative_scores: np.ndarray
) -> tuple[float, float, float, float, float]:
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    scores = np.concatenate([pos, neg])
    labels = np.concatenate(
        [np.ones(pos.shape[0], dtype=int), np.zeros(neg.shape[0], dtype=int)]
    )

    desc_indices = np.argsort(scores)[::-1]
    scores = scores[desc_indices]
    labels = labels[desc_indices]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, labels.size - 1]

    tps = np.cumsum(labels)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[scores[threshold_idxs], scores[threshold_idxs][-1] - 1e-12]

    total_pos = float(pos.size)
    total_neg = float(neg.size)
    tpr = tps / total_pos
    fpr = fps / total_neg

    roc_auc = float(np.trapezoid(tpr, fpr))
    youden = tpr - fpr
    best_idx = int(np.nanargmax(youden))

    youden_threshold = float(thresholds[best_idx])
    youden_sensitivity = float(tpr[best_idx])
    youden_specificity = float(1.0 - fpr[best_idx])
    youden_j = float(youden[best_idx])

    return (
        roc_auc,
        youden_threshold,
        youden_sensitivity,
        youden_specificity,
        youden_j,
    )


def _symmetric_divergences(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([values_a, values_b])
    bin_edges = np.histogram_bin_edges(combined, bins=HISTOGRAM_BINS)
    hist_a, _ = np.histogram(values_a, bins=bin_edges)
    hist_b, _ = np.histogram(values_b, bins=bin_edges)

    p = hist_a.astype(float) + HISTOGRAM_EPS
    q = hist_b.astype(float) + HISTOGRAM_EPS
    p /= p.sum()
    q /= q.sum()

    kl_ab = float(np.sum(p * np.log(p / q)))
    kl_ba = float(np.sum(q * np.log(q / p)))
    kl_sym = 0.5 * (kl_ab + kl_ba)

    m = 0.5 * (p + q)
    js_div = 0.5 * (
        float(np.sum(p * np.log(p / m))) + float(np.sum(q * np.log(q / m)))
    )

    return kl_sym, js_div


def _safe_kurtosis(values: np.ndarray) -> float:
    val = float(kurtosis(values, fisher=True, bias=False))
    if np.isnan(val):
        return 0.0
    return val


def _apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_metrics = load_metric_values()
    records, matrices, dataset_names = compute_statistics(dataset_metrics)
    save_summary(records, output_dir)
    plot_heatmaps(matrices, dataset_names, output_dir)


if __name__ == "__main__":
    main()
