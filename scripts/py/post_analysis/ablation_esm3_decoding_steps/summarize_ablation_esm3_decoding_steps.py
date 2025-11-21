#!/usr/bin/env python3
"""Summarize ESM3 decoding-step ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STEP_PATTERN = re.compile(r"steps(?P<steps>\d+)")

METRIC_FIELDS = [
    ("plddt", "pLDDT"),
    ("ptm", "pTM"),
    ("entropy_norm", "Hnorm"),
    ("distinct2", "Distinct-2"),
    ("distinct3", "Distinct-3"),
    ("H_poly_k4", "Rhpoly"),
]


def structure_utility_score(plddt: float, ptm: float) -> float:
    """Match src.replm.metrics.structure.structure_utility_score."""

    plddt = plddt / 100.0 if plddt > 1.0 else plddt
    return (plddt + ptm) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize decoding-step ablations for ESM3."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing esm3_len*_steps*.summary.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file (Markdown). Defaults to stdout.",
    )
    return parser.parse_args()


def collect_runs(root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*.summary.json")):
        name = summary_path.name
        if not name.endswith(".summary.json"):
            continue
        prefix = name[: -len(".summary.json")]
        match = STEP_PATTERN.search(prefix)
        if not match:
            print(f"Skipping file without steps indicator: {name}", file=sys.stderr)
            continue
        steps = int(match.group("steps"))
        metrics_path = summary_path.with_name(f"{prefix}.metrics.csv")
        runs.append(
            {
                "steps": steps,
                "summary": summary_path,
                "metrics": metrics_path,
            }
        )
    runs.sort(key=lambda item: item["steps"])
    return runs


def read_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_metrics(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field, _ in METRIC_FIELDS:
                raw = row.get(field)
                if raw is None or raw == "":
                    continue
                try:
                    values[field].append(float(raw))
                except ValueError:
                    continue
    return values


def compute_stats(values: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for field, _ in METRIC_FIELDS:
        vals = values.get(field, [])
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[field] = (mean, std)
    return stats


def compute_biological_utility(stats: dict[str, tuple[float, float]]) -> float:
    plddt = stats.get("plddt")
    ptm = stats.get("ptm")
    if not plddt or not ptm:
        return float("nan")
    return structure_utility_score(plddt[0], ptm[0])


def format_stat(value: tuple[float, float] | None) -> str:
    if not value:
        return "-"
    mean, std = value
    return f"{mean:.3f}±{std:.3f}"


def format_value(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    text = f"{value:.{digits}f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def render_table(out_handle, header: list[str], rows: list[list[str]]) -> None:
    if not rows:
        out_handle.write("_(no data)_\n\n")
        return
    out_handle.write("| " + " | ".join(header) + " |\n")
    out_handle.write("| " + " | ".join(["---"] * len(header)) + " |\n")
    for row in rows:
        out_handle.write("| " + " | ".join(row) + " |\n")
    out_handle.write("\n")


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.root)
    if not runs:
        raise SystemExit(f"No summary files found under {args.root}")

    metric_rows: list[list[str]] = []
    score_rows: list[list[str]] = []

    for run in runs:
        steps = run["steps"]
        summary_path = run["summary"]
        metrics_path = run["metrics"]

        summary_data = read_summary(summary_path)
        metric_values = read_metrics(metrics_path)
        stats = compute_stats(metric_values)
        bio_util = compute_biological_utility(stats)

        metric_rows.append(
            [
                str(steps),
                *[format_stat(stats.get(field)) for field, _ in METRIC_FIELDS],
            ]
        )

        score_rows.append(
            [
                str(steps),
                format_value(summary_data.get("repetition_score")),
                format_value(bio_util),
                format_value(summary_data.get("diversity_pid")),
            ]
        )

    out_handle = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        out_handle.write("## ESM3 Decoding-Step Ablation Summary\n\n")

        header_metrics = ["Decoding Steps"] + [label for _, label in METRIC_FIELDS]
        render_table(out_handle, header_metrics, metric_rows)

        out_handle.write("---\n\n")

        header_scores = [
            "Decoding Steps",
            "Repetition Score",
            "Biological Utility",
            "Diversity PID",
        ]
        render_table(out_handle, header_scores, score_rows)
    finally:
        if args.output:
            out_handle.close()


if __name__ == "__main__":
    main()
