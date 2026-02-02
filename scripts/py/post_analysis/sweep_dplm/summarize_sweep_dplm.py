#!/usr/bin/env python3
"""Summarize DPLM-style sweeps, selecting the best parameter per method."""

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
from typing import NamedTuple

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

DEFAULT_LATEX_CAPTION = "DPLM sweep results for two tasks with utility constraint."
DEFAULT_LATEX_LABEL = "tab:dplm-sweep"
DEFAULT_LATEX_NOTES = (
    r"\footnotesize \emph{Notes.} $R$: repetition score; $U$: biological utility. Cells marked with \good{} satisfy the utility constraint relative to the Original Model."
)

METHOD_CONFIG = {
    "control": {"label": "Original Model"},
    "temperature": {"label": "Temperature"},
    "sampling": {"label": "Sampling Strategy"},
    "disable_resample": {"label": "Disable Resample"},
    "resample_ratio": {"label": "Resample Ratio"},
    "neuron_deactivation": {"label": "Neuron Deactivation"},
    "probe_layer": {"label": "Probe Steering"},
    "uucs_layer": {"label": "UUCS"},
}
METHOD_ORDER = [
    "control",
    "temperature",
    "sampling",
    "disable_resample",
    "resample_ratio",
    "neuron_deactivation",
    "probe_layer",
    "uucs_layer",
]

RUN_PATTERN = re.compile(r"seed_(?P<seed>\d+)_method_(?P<method>.+)_dplm_dataset_(?P<dataset>.+)$")
RUN_GLOB = "seed_*_method_*_dplm_dataset_*"


class RunReference(NamedTuple):
    path: Path
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DPLM sweep outputs, selecting the best parameter per method."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing seed_*_method_*_dplm_dataset_* experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "latex"),
        default="markdown",
        help="Output format. Defaults to Markdown.",
    )
    parser.add_argument(
        "--latex-caption",
        type=str,
        default=None,
        help="Caption text when using LaTeX output.",
    )
    parser.add_argument(
        "--latex-label",
        type=str,
        default=None,
        help="Label identifier when using LaTeX output.",
    )
    parser.add_argument(
        "--latex-notes",
        type=str,
        default=None,
        help="Optional caption* contents (include size commands) for LaTeX output.",
    )
    parser.add_argument(
        "--probe-layer",
        type=int,
        default=None,
        help="Force Probe Steering rows to report this specific layer.",
    )
    parser.add_argument(
        "--uucs-layer",
        type=int,
        default=None,
        help="Force UUCS rows to report this specific layer.",
    )
    parser.add_argument(
        "--latest-per-experiment",
        action="store_true",
        help=(
            "Treat root as containing experiment subfolders (<experiment>/run_<timestamp>) and "
            "summarize only the newest run per experiment."
        ),
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


def _format_strategy(token: str) -> str:
    parts = token.split("_")
    if not parts:
        return token
    return " ".join(word.capitalize() for word in parts)


def parse_method_name(name: str) -> tuple[str, str]:
    """Return (method_base, parameter_label)."""
    if name.startswith("control"):
        return "control", "-"
    if name == "disable_resample":
        return "disable_resample", "-"
    patterns = [
        (r"temperature_(.+)", "temperature", lambda val: f"T={_format_float(val)}"),
        (
            r"sampling_(.+)",
            "sampling",
            lambda val: f"Strategy={_format_strategy(val)}",
        ),
        (
            r"resample_ratio_(.+)",
            "resample_ratio",
            lambda val: f"Ratio={_format_float(val)}",
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
            r"uucs_layer(\d+)",
            "uucs_layer",
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
            return base, formatter(token)
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


def compute_utility_baselines(
    aggregated: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]],
    best_map: dict[str, dict[str, dict[str, dict[str, str]]]],
) -> dict[str, dict[str, float | None]]:
    baselines: dict[str, dict[str, float | None]] = {}
    for condition in CONDITIONS:
        baselines[condition] = {}
        for dataset in DATASET_ORDER:
            value: float | None = None
            method_info = best_map.get(condition, {}).get(dataset, {}).get("control")
            if method_info:
                stats = aggregated.get(condition, {}).get(dataset, {}).get(method_info["method_name"])
                util = stats.get("utility_score") if stats else None
                if util and math.isfinite(util[0]):
                    value = util[0]
            baselines[condition][dataset] = value
    return baselines


def format_stat_latex(value: tuple[float, float] | None, *, annotate_good: bool = False) -> str:
    if not value:
        return "--"
    mean, std = value
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "--"
    text = f"${mean:.3f}\\pm{std:.3f}$"
    if annotate_good:
        return f"\\good{{{text}}}"
    return text


def build_latex_rows(
    condition: str,
    aggregated: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]],
    best_map: dict[str, dict[str, dict[str, dict[str, str]]]],
    baselines: dict[str, dict[str, float | None]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for method_base in METHOD_ORDER:
        method_label = METHOD_CONFIG.get(method_base, {}).get("label", method_base)
        row = [method_label]
        has_data = False
        for dataset in DATASET_ORDER:
            best_info = best_map.get(condition, {}).get(dataset, {}).get(method_base)
            stats = aggregated.get(condition, {}).get(dataset, {}).get(best_info["method_name"]) if best_info else None
            rep = stats.get("repetition_score") if stats else None
            util = stats.get("utility_score") if stats else None
            row.append(format_stat_latex(rep))
            baseline = baselines.get(condition, {}).get(dataset)
            annotate = bool(
                util and baseline is not None and math.isfinite(util[0]) and util[0] >= baseline
            )
            row.append(format_stat_latex(util, annotate_good=annotate))
            if stats:
                has_data = True
        if has_data:
            rows.append(row)
    return rows


def render_latex_summary(
    out_handle,
    aggregated: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]],
    best_map: dict[str, dict[str, dict[str, dict[str, str]]]],
    *,
    caption: str,
    label: str,
    notes: str | None,
) -> None:
    baselines = compute_utility_baselines(aggregated, best_map)
    dataset_count = len(DATASET_ORDER)
    column_spec = f"l *{{{dataset_count}}}{{r r}}"
    total_columns = 1 + dataset_count * 2

    out_handle.write("\\begin{table}[t]\n")
    out_handle.write("\\centering\n")
    out_handle.write("\\footnotesize\n")
    out_handle.write("\\setlength{\\tabcolsep}{5.5pt}\n")
    out_handle.write("\\renewcommand{\\arraystretch}{1.15}\n")
    out_handle.write(f"\\caption{{{caption}}}\n")
    out_handle.write(f"\\label{{{label}}}\\vspace{{-5pt}}\n")
    out_handle.write("\\resizebox{\\linewidth}{!}{%\n")
    out_handle.write(f"\\begin{{tabular}}{{{column_spec}}}\n")
    out_handle.write("\\toprule\n")

    header = "\\multirow{2}{*}{\\textbf{Method}}"
    for dataset in DATASET_ORDER:
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        header += f"  &  \\multicolumn{{2}}{{c}}{{\\textbf{{{dataset_label}}}}}"
    out_handle.write(header + " \\\\n")

    cmidrules = []
    for idx in range(dataset_count):
        start = 2 + idx * 2
        end = start + 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    out_handle.write("".join(cmidrules) + "\n")

    subheader_cells = ["{$R \\uparrow$} & {$U \\uparrow$}" for _ in DATASET_ORDER]
    out_handle.write(" & " + " & ".join(subheader_cells) + " \\\\n")
    out_handle.write("\\midrule\n")

    for idx, condition in enumerate(CONDITIONS):
        cond_label = f"{CONDITION_LABELS.get(condition, condition)} generation"
        letter = chr(ord("a") + idx)
        out_handle.write(f"\\textbf{{({letter}) {cond_label}}}\\\\\n")
        rows = build_latex_rows(condition, aggregated, best_map, baselines)
        if rows:
            for row in rows:
                out_handle.write(" & ".join(row) + " \\\\n")
        else:
            out_handle.write(f"\\multicolumn{{{total_columns}}}{{c}}{{No data}} \\\\n")
        out_handle.write("\\addlinespace[3pt]\n")
        if idx < len(CONDITIONS) - 1:
            out_handle.write("\\midrule\n")

    out_handle.write("\\bottomrule\n")
    out_handle.write("\\end{tabular}\n")
    out_handle.write("}%\n")
    if notes:
        out_handle.write(f"\\caption*{{{notes}}}\n")
    out_handle.write("\\vspace{-12pt}\n")
    out_handle.write("\\end{table}\n")


def discover_run_dirs(root: Path, latest_per_experiment: bool) -> list[RunReference]:
    if not latest_per_experiment:
        return [RunReference(path=path, label=path.name) for path in sorted(root.glob(RUN_GLOB))]

    run_dirs: list[RunReference] = []
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        run_subdirs = sorted(
            (
                path
                for path in experiment_dir.iterdir()
                if path.is_dir() and path.name.startswith("run_")
            ),
            key=lambda path: path.name,
        )
        if not run_subdirs:
            continue
        latest_run = run_subdirs[-1]
        run_dirs.append(RunReference(path=latest_run, label=experiment_dir.name))
    return run_dirs


def main() -> None:
    args = parse_args()
    run_refs = discover_run_dirs(args.root, args.latest_per_experiment)
    if not run_refs:
        if args.latest_per_experiment:
            raise SystemExit(
                f"No run directories found under {args.root} when using --latest-per-experiment; "
                "expected <root>/<experiment>/<run_*> structure."
            )
        raise SystemExit(f"No run directories found under {args.root}")

    summary_store: dict[
        str,
        dict[str, defaultdict[str, list[dict[str, float]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    method_meta: dict[str, tuple[str, str]] = {}

    for run_ref in run_refs:
        match = RUN_PATTERN.fullmatch(run_ref.label)
        if not match:
            print(f"Skipping directory with unexpected name: {run_ref.label}", file=sys.stderr)
            continue
        method_name = match.group("method")
        dataset = match.group("dataset")
        if dataset not in DATASET_LABELS:
            print(f"Unknown dataset '{dataset}' in {run_ref.label}", file=sys.stderr)
            continue
        method_meta.setdefault(method_name, parse_method_name(method_name))

        files = {
            "conditional": {
                "metrics": run_ref.path / "prefix.steer.metrics.csv",
                "summary": run_ref.path / "prefix.steer.summary.json",
            },
            "unconditional": {
                "metrics": run_ref.path / "uncond.steer.metrics.csv",
                "summary": run_ref.path / "uncond.steer.summary.json",
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
        "uucs_layer": args.uucs_layer,
    }
    forced = {k: v for k, v in forced.items() if v is not None}
    best_map = compute_best_parameters(
        aggregated,
        method_meta,
        forced_layers=forced,
    )

    out_handle = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        if args.format == "latex":
            caption = args.latex_caption or DEFAULT_LATEX_CAPTION
            label = args.latex_label or DEFAULT_LATEX_LABEL
            notes = args.latex_notes if args.latex_notes is not None else DEFAULT_LATEX_NOTES
            render_latex_summary(out_handle, aggregated, best_map, caption=caption, label=label, notes=notes)
        else:
            out_handle.write("## DPLM Sweep Summary\n\n")
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
