#!/usr/bin/env python3
"""Summarize decoding/steering sweeps into per-model tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

MODEL_CONFIG = {
    "esm3": {
        "label": "ESM3",
        "method_order": ["temperature", "top_p", "entropy"],
        "methods": {
            "temperature": {"label": "Temperature + UCCS", "parameter": "T=1.3"},
            "top_p": {"label": "Top-p + UCCS", "parameter": "p=0.95"},
            "entropy": {"label": "Entropy Sampling + UCCS", "parameter": "-"},
        },
    },
    "protgpt2": {
        "label": "ProtGPT2",
        "method_order": [
            "temperature",
            "top_p",
            "repetition_penalty",
            "no_repeat_ngram",
        ],
        "methods": {
            "temperature": {"label": "Temperature + UCCS", "parameter": "T=1.3"},
            "top_p": {"label": "Top-p + UCCS", "parameter": "p=0.98"},
            "repetition_penalty": {"label": "Repetition Penalty + UCCS", "parameter": "1.2"},
            "no_repeat_ngram": {"label": "No Repeat N-gram + UCCS", "parameter": "N=3"},
        },
    },
}

DATASET_LABELS = {
    "cath": "CATH",
    "uniref50": "UniRef50",
    "scop": "SCOP",
}
DATASET_ORDER = ["cath", "uniref50", "scop"]

METRIC_FIELDS = [
    ("plddt", "pLDDT"),
    ("ptm", "pTM"),
    ("entropy_norm", "Hnorm"),
    ("distinct2", "Distinct-2"),
    ("distinct3", "Distinct-3"),
    ("H_poly_k4", "Rhpoly"),
]

SUMMARY_FIELDS = [
    ("repetition_score", "Repetition Score"),
    ("utility_score", "Biological Utility"),
    ("diversity_pid", "Diversity (PID)"),
]

CONDITION_LABELS = {"unconditional": "Unconditional", "conditional": "Conditional"}

RUN_PATTERN = re.compile(
    r"seed_(?P<seed>\d+)_method_(?P<method>.+)_(?P<model>esm3|protgpt2)_dataset_(?P<dataset>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate sweep outputs into Markdown tables. "
            "Provide the directory containing seed_* folders."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to sweep outputs (e.g. sweep-abalation-decoding-105)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the generated tables (default: stdout)",
    )
    return parser.parse_args()


def read_metrics_file(path: Path) -> dict[str, float]:
    """Compute per-run averages for the requested metrics."""
    values = {field: [] for field, _ in METRIC_FIELDS}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in values:
                if row.get(field) is None:
                    continue
                try:
                    values[field].append(float(row[field]))
                except ValueError:
                    continue
    return {
        field: (sum(vals) / len(vals) if vals else float("nan"))
        for field, vals in values.items()
    }


def read_summary_file(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {field: float(data[field]) for field, _ in SUMMARY_FIELDS if field in data}


def structure_utility_score(plddt: float, ptm: float) -> float:
    """Match src.replm.metrics.structure.structure_utility_score without imports."""

    plddt = plddt / 100.0 if plddt > 1.0 else plddt
    return (plddt + ptm) / 2.0


def compute_utility_score_from_metrics(metrics: dict[str, float] | None) -> float:
    """Compute biological utility via structure_utility_score."""

    if not metrics:
        return float("nan")
    plddt = metrics.get("plddt")
    ptm = metrics.get("ptm")
    if plddt is None or ptm is None:
        return float("nan")
    if not (math.isfinite(plddt) and math.isfinite(ptm)):
        return float("nan")
    return structure_utility_score(plddt, ptm)


def aggregate_store(
    store: defaultdict[str, defaultdict[str, defaultdict[str, defaultdict[str, list[dict[str, float]]]]]],
    fields: Iterable[tuple[str, str]],
) -> dict[str, dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]]]:
    aggregated: dict[str, dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]]] = {}
    for model, condition_map in store.items():
        aggregated[model] = {}
        for condition, method_map in condition_map.items():
            aggregated[model][condition] = {}
            for method, dataset_map in method_map.items():
                aggregated[model][condition][method] = {}
                for dataset, runs in dataset_map.items():
                    aggregated[model][condition][method][dataset] = compute_stats(
                        runs, fields
                    )
    return aggregated


def compute_stats(
        runs: Iterable[dict[str, float]],
    fields: Iterable[tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for field, _ in fields:
        vals = [run[field] for run in runs if field in run and run[field] == run[field]]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        result[field] = (mean, std)
    return result


def format_stat(value: tuple[float, float] | None) -> str:
    if not value:
        return "-"
    mean, std = value
    return f"{mean:.3f}±{std:.3f}"


def render_table(
    out_handle,
        header: list[str],
    rows: list[list[str]],
) -> None:
    if not rows:
        out_handle.write("_(no data)_\n\n")
        return
    out_handle.write("| " + " | ".join(header) + " |\n")
    out_handle.write("| " + " | ".join(["---"] * len(header)) + " |\n")
    for row in rows:
        out_handle.write("| " + " | ".join(row) + " |\n")
    out_handle.write("\n")


def build_rows(
    model_key: str,
    condition: str,
    aggregated_metrics: dict[str, dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]]],
    fields: Iterable[tuple[str, str]],
) -> list[list[str]]:
    model_cfg = MODEL_CONFIG[model_key]
    rows: list[list[str]] = []
    for dataset in DATASET_ORDER:
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        for method in model_cfg["method_order"]:
            method_cfg = model_cfg["methods"][method]
            stats = (
                aggregated_metrics.get(model_key, {})
                .get(condition, {})
                .get(method, {})
                .get(dataset)
            )
            if not stats:
                continue
            row = [dataset_label, method_cfg["label"], method_cfg["parameter"]]
            for field, _ in fields:
                row.append(format_stat(stats.get(field)))
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    run_dirs = sorted(args.root.glob("seed_*_method_*_dataset_*"))
    if not run_dirs:
        raise SystemExit(f"No run directories found under {args.root}")

    metrics_store: defaultdict[
        str,
        defaultdict[str, defaultdict[str, defaultdict[str, list[dict[str, float]]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    summary_store: defaultdict[
        str,
        defaultdict[str, defaultdict[str, defaultdict[str, list[dict[str, float]]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for run_dir in run_dirs:
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if not match:
            print(f"Skipping unrecognized directory name: {run_dir.name}", file=sys.stderr)
            continue
        model = match.group("model")
        method = match.group("method")
        dataset = match.group("dataset")

        if model not in MODEL_CONFIG:
            print(f"Unknown model '{model}' in {run_dir.name}", file=sys.stderr)
            continue
        if method not in MODEL_CONFIG[model]["methods"]:
            print(f"Unknown method '{method}' for {model} in {run_dir.name}", file=sys.stderr)
            continue
        if dataset not in DATASET_LABELS:
            print(f"Unknown dataset '{dataset}' in {run_dir.name}", file=sys.stderr)
            continue

        prefix_metrics = run_dir / "prefix.steer.metrics.csv"
        uncond_metrics = run_dir / "uncond.steer.metrics.csv"
        prefix_summary = run_dir / "prefix.steer.summary.json"
        uncond_summary = run_dir / "uncond.steer.summary.json"

        prefix_metrics_data = None
        if prefix_metrics.exists():
            prefix_metrics_data = read_metrics_file(prefix_metrics)
            metrics_store[model]["conditional"][method][dataset].append(
                prefix_metrics_data
            )
        else:
            print(f"Missing {prefix_metrics}", file=sys.stderr)

        uncond_metrics_data = None
        if uncond_metrics.exists():
            uncond_metrics_data = read_metrics_file(uncond_metrics)
            metrics_store[model]["unconditional"][method][dataset].append(
                uncond_metrics_data
            )
        else:
            print(f"Missing {uncond_metrics}", file=sys.stderr)

        prefix_summary_data: dict[str, float] = {}
        if prefix_summary.exists():
            prefix_summary_data = read_summary_file(prefix_summary)
        else:
            print(f"Missing {prefix_summary}", file=sys.stderr)
        prefix_utility = compute_utility_score_from_metrics(prefix_metrics_data)
        if prefix_utility == prefix_utility:
            prefix_summary_data["utility_score"] = prefix_utility
        else:
            prefix_summary_data.pop("utility_score", None)
        if prefix_summary_data:
            summary_store[model]["conditional"][method][dataset].append(
                prefix_summary_data
            )

        uncond_summary_data: dict[str, float] = {}
        if uncond_summary.exists():
            uncond_summary_data = read_summary_file(uncond_summary)
        else:
            print(f"Missing {uncond_summary}", file=sys.stderr)
        uncond_utility = compute_utility_score_from_metrics(uncond_metrics_data)
        if uncond_utility == uncond_utility:
            uncond_summary_data["utility_score"] = uncond_utility
        else:
            uncond_summary_data.pop("utility_score", None)
        if uncond_summary_data:
            summary_store[model]["unconditional"][method][dataset].append(
                uncond_summary_data
            )

    aggregated_metrics = aggregate_store(metrics_store, METRIC_FIELDS)
    aggregated_summary = aggregate_store(summary_store, SUMMARY_FIELDS)

    out_handle = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for model_key in ("esm3", "protgpt2"):
            if model_key not in aggregated_metrics:
                continue
            model_label = MODEL_CONFIG[model_key]["label"]
            out_handle.write(f"## {model_label}\n\n")

            out_handle.write("### Detailed Metrics\n\n")
            for condition in ("unconditional", "conditional"):
                out_handle.write(f"**{CONDITION_LABELS[condition]}**\n\n")
                header = ["Dataset", "Method", "Parameter"] + [
                    label for _, label in METRIC_FIELDS
                ]
                rows = build_rows(model_key, condition, aggregated_metrics, METRIC_FIELDS)
                render_table(out_handle, header, rows)

            out_handle.write("### Summary Scores\n\n")
            for condition in ("unconditional", "conditional"):
                out_handle.write(f"**{CONDITION_LABELS[condition]}**\n\n")
                header = ["Dataset", "Method", "Parameter"] + [
                    label for _, label in SUMMARY_FIELDS
                ]
                rows = build_rows(model_key, condition, aggregated_summary, SUMMARY_FIELDS)
                render_table(out_handle, header, rows)

            out_handle.write("\n")
    finally:
        if args.output:
            out_handle.close()


if __name__ == "__main__":
    main()
