#!/usr/bin/env python3
"""Summarize ProGen2-style sweeps, selecting the best parameter per method."""

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

DATASET_LABELS = {
    "cath": "CATH",
    "uniref50": "UniRef50",
    "scop": "SCOP",
}
DATASET_ORDER = ["cath", "uniref50", "scop"]

SUMMARY_FIELDS = [
    ("repetition_score", "Repetition Score"),
    ("utility_score", "Biological Utility"),
    ("diversity_pid", "Diversity PID"),
]
CONDITION_LABELS = {"unconditional": "Unconditional", "conditional": "Conditional"}
CONDITIONS = ("unconditional", "conditional")

METHOD_CONFIG = {
    "control": {"label": "Original Model"},
    "temperature": {"label": "Temperature"},
    "top_p": {"label": "Top-p Sampling"},
    "no_repeat_ngram": {"label": "No Repeat N-gram"},
    "repetition_penalty": {"label": "Repetition Penalty"},
    "neuron_deactivation": {"label": "Neuron Deactivation"},
    "probe_layer": {"label": "Probe Steering"},
    "uccs_layer": {"label": "UCCS"},
}
METHOD_ORDER = [
    "control",
    "temperature",
    "top_p",
    "no_repeat_ngram",
    "repetition_penalty",
    "neuron_deactivation",
    "probe_layer",
    "uccs_layer",
]

RUN_PATTERN = re.compile(r"seed_(?P<seed>\d+)_method_(?P<method>.+)_dataset_(?P<dataset>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ProGen2 sweep outputs, selecting the best parameter per method."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing seed_*_method_*_dataset_* experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file (Markdown). Defaults to stdout.",
    )
    parser.add_argument(
        "--probe-layer",
        type=int,
        default=None,
        help="Force Probe Steering rows to report this specific layer.",
    )
    parser.add_argument(
        "--uccs-layer",
        type=int,
        default=None,
        help="Force UCCS rows to report this specific layer.",
    )
    return parser.parse_args()


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _token_to_float(token: str) -> float | None:
    token = token.replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def parse_method_name(name: str) -> tuple[str, str]:
    """Return (method_base, parameter_label)."""
    if name.startswith("control"):
        return "control", "-"
    patterns = [
        (r"temperature_(.+)", "temperature", lambda val: f"T={_format_float(val)}"),
        (r"top_p_(.+)", "top_p", lambda val: f"p={_format_float(val)}"),
        (
            r"no_repeat_ngram_(\d+)",
            "no_repeat_ngram",
            lambda val: f"N={int(val)}",
        ),
        (
            r"repetition_penalty_(.+)",
            "repetition_penalty",
            lambda val: f"Penalty={_format_float(val)}",
        ),
        (
            r"neuron_deactivation_(\d+)",
            "neuron_deactivation",
            lambda val: f"TopK={int(val)}",
        ),
        (
            r"probe_layer(\d+)",
            "probe_layer",
            lambda val: f"Layer={int(val)}",
        ),
        (
            r"uccs_layer(\d+)",
            "uccs_layer",
            lambda val: f"Layer={int(val)}",
        ),
    ]
    for pattern, base, formatter in patterns:
        match = re.fullmatch(pattern, name)
        if not match:
            continue
        token = match.group(1)
        if token is None:
            return base, name
        num = _token_to_float(token)
        if num is None:
            return base, token
        return base, formatter(num)
    return name, "-"


def read_metrics_file(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = {
        "plddt": [],
        "ptm": [],
    }
    if not path.exists():
        return {}
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
    return {field: (sum(vals) / len(vals) if vals else float("nan")) for field, vals in values.items()}


def read_summary_file(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_utility_score(metrics: dict[str, float] | None) -> float:
    if not metrics:
        return float("nan")
    vals = []
    if metrics.get("plddt") is not None:
        vals.append(metrics["plddt"] / 100.0)
    if metrics.get("ptm") is not None:
        vals.append(metrics["ptm"])
    if not vals:
        return float("nan")
    return statistics.fmean(vals)


def compute_stats(
    runs: Iterable[dict[str, float]],
    fields: Iterable[tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for field, _ in fields:
        vals = [run[field] for run in runs if field in run and run[field] == run[field]]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[field] = (mean, std)
    return stats


def harmonic_mean(stats: dict[str, tuple[float, float]] | None) -> float:
    if not stats:
        return float("nan")
    rep = stats.get("repetition_score")
    util = stats.get("utility_score")
    if not rep or not util:
        return float("nan")
    a = rep[0]
    b = util[0]
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    if a <= 0 or b <= 0:
        return float("nan")
    return 2 * a * b / (a + b)


def format_stat(value: tuple[float, float] | None) -> str:
    if not value:
        return "-"
    mean, std = value
    return f"{mean:.3f}±{std:.3f}"


def format_hmean(stats: dict[str, tuple[float, float]] | None) -> str:
    h = harmonic_mean(stats)
    if not math.isfinite(h):
        return "-"
    return f"{h:.3f}"


def aggregate_summary(
    store: dict[str, dict[str, defaultdict[str, list[dict[str, float]]]]],
) -> dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]]:
    aggregated: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]] = {}
    for condition, dataset_map in store.items():
        aggregated[condition] = {}
        for dataset, method_map in dataset_map.items():
            aggregated[condition][dataset] = {}
            for method, runs in method_map.items():
                aggregated[condition][dataset][method] = compute_stats(runs, SUMMARY_FIELDS)
    return aggregated


def compute_best_parameters(
    aggregated: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]],
    method_meta: dict[str, tuple[str, str]],
    *,
    forced_layers: dict[str, int | None] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Return best parameter per (condition, dataset, method)."""

    best: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    forced_layers = forced_layers or {}

    def _layer_matches(method_name: str, method_base: str, target: int | None) -> bool:
        if target is None:
            return False
        suffix = method_name[len(method_base) :]
        try:
            layer_idx = int(suffix)
        except ValueError:
            return False
        return layer_idx == target

    for condition in CONDITIONS:
        best.setdefault(condition, {})
        cond_data = aggregated.get(condition, {})
        for dataset in DATASET_ORDER:
            best[condition].setdefault(dataset, {})
            dataset_stats = cond_data.get(dataset, {})
            for method_base in METHOD_ORDER:
                forced_layer = forced_layers.get(method_base)
                if forced_layer is not None:
                    forced_name = next(
                        (
                            method_name
                            for method_name, meta in method_meta.items()
                            if meta[0] == method_base
                            and _layer_matches(method_name, method_base, forced_layer)
                        ),
                        None,
                    )
                    stats = dataset_stats.get(forced_name) if forced_name else None
                    if not stats:
                        continue
                    best[condition][dataset][method_base] = {
                        "method_name": forced_name,
                        "parameter_label": method_meta[forced_name][1],
                    }
                    continue

                best_score = float("-inf")
                best_name = None
                for method_name, meta in method_meta.items():
                    base, _ = meta
                    if base != method_base:
                        continue
                    stats = dataset_stats.get(method_name)
                    if not stats:
                        continue
                    score = harmonic_mean(stats)
                    if not math.isfinite(score):
                        continue
                    if score > best_score:
                        best_score = score
                        best_name = method_name

                if best_name is not None:
                    best[condition][dataset][method_base] = {
                        "method_name": best_name,
                        "parameter_label": method_meta[best_name][1],
                    }

    return best


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
    condition: str,
    aggregated: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]],
    best_map: dict[str, dict[str, dict[str, dict[str, str]]]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for dataset in DATASET_ORDER:
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        for method_base in METHOD_ORDER:
            best_info = best_map.get(condition, {}).get(dataset, {}).get(method_base)
            if not best_info:
                continue
            method_label = METHOD_CONFIG.get(method_base, {}).get("label", method_base)
            stats = aggregated.get(condition, {}).get(dataset, {}).get(best_info["method_name"])
            rows.append(
                [
                    dataset_label,
                    method_label,
                    best_info["parameter_label"],
                    format_stat(stats.get("repetition_score") if stats else None),
                    format_stat(stats.get("utility_score") if stats else None),
                    format_stat(stats.get("diversity_pid") if stats else None),
                    format_hmean(stats),
                ]
            )
    return rows


def main() -> None:
    args = parse_args()
    run_dirs = sorted(args.root.glob("seed_*_method_*_dataset_*"))
    if not run_dirs:
        raise SystemExit(f"No run directories found under {args.root}")

    summary_store: dict[
        str,
        dict[str, defaultdict[str, list[dict[str, float]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    method_meta: dict[str, tuple[str, str]] = {}

    for run_dir in run_dirs:
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if not match:
            print(f"Skipping directory with unexpected name: {run_dir.name}", file=sys.stderr)
            continue
        method_name = match.group("method")
        dataset = match.group("dataset")
        if dataset not in DATASET_LABELS:
            print(f"Unknown dataset '{dataset}' in {run_dir.name}", file=sys.stderr)
            continue
        method_meta.setdefault(method_name, parse_method_name(method_name))

        files = {
            "conditional": {
                "metrics": run_dir / "prefix.steer.metrics.csv",
                "summary": run_dir / "prefix.steer.summary.json",
            },
            "unconditional": {
                "metrics": run_dir / "uncond.steer.metrics.csv",
                "summary": run_dir / "uncond.steer.summary.json",
            },
        }

        for condition, paths in files.items():
            metrics_data = read_metrics_file(paths["metrics"])
            summary_data = read_summary_file(paths["summary"])
            utility = compute_utility_score(metrics_data)
            if math.isfinite(utility):
                summary_data["utility_score"] = utility
            elif "utility_score" in summary_data:
                summary_data.pop("utility_score")
            if summary_data:
                summary_store[condition][dataset][method_name].append(summary_data)

    aggregated = aggregate_summary(summary_store)
    forced = {
        "probe_layer": args.probe_layer,
        "uccs_layer": args.uccs_layer,
    }
    # remove None entries to avoid forcing when unspecified
    forced = {k: v for k, v in forced.items() if v is not None}
    best_map = compute_best_parameters(
        aggregated,
        method_meta,
        forced_layers=forced,
    )

    out_handle = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        out_handle.write("## ProGen2 Sweep Summary\n\n")
        for condition in CONDITIONS:
            header = [
                "Dataset",
                "Method",
                "Best Parameter",
                "Repetition Score",
                "Biological Utility",
                "Diversity PID",
                "Harmonic Mean(Repetition Score, Biological Utility)",
            ]
            out_handle.write(f"### {CONDITION_LABELS[condition]}\n\n")
            rows = build_rows(condition, aggregated, best_map)
            render_table(out_handle, header, rows)
    finally:
        if args.output:
            out_handle.close()


if __name__ == "__main__":
    main()
