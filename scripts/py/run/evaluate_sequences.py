#!/usr/bin/env python
# scripts/py/run/evaluate_sequences.py
"""
Evaluate repetition/structure/diversity metrics for a FASTA dataset.

Example:
    python scripts/py/run/evaluate_sequences.py data/example/example.fasta \
    --structure-model esm3 \
    --structure-device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from replm.metrics.diversity import pairwise_percent_identity
from replm.metrics.repetition import repetition_metrics, repetition_score
from replm.metrics.structure import StructureProxyModel, get_structure_model
from replm.utils.io import read_fasta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate repetition/structure/diversity metrics for a FASTA dataset.",
    )
    parser.add_argument("fasta", type=Path, help="Input FASTA file.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to write the per-sequence CSV. Defaults to <fasta>.metrics.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to write dataset-level summary JSON. Defaults to <fasta>.summary.json",
    )
    parser.add_argument(
        "--structure-model",
        type=str,
        default="esm3",
        help="Structure proxy registered under replm.metrics.structure (default: esm3).",
    )
    parser.add_argument(
        "--structure-device",
        type=str,
        default=None,
        help="Optional device override passed to the structure proxy.",
    )
    parser.add_argument(
        "--skip-structure",
        action="store_true",
        help="Skip structure evaluation (ptm/plddt columns will be NaN).",
    )
    return parser


def _instantiate_structure_proxy(name: str, device: str | None) -> StructureProxyModel:
    cfg: dict[str, Any] = {}
    if device is not None:
        cfg["device"] = device
    return get_structure_model(name, **cfg)


def _coerce_plddt(val: float | None) -> float:
    if val is None or not math.isfinite(val):
        return float("nan")
    if 0.0 <= val <= 1.0:
        return val * 100.0
    return val


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    fasta_path = args.fasta
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    seqs = read_fasta(fasta_path)
    if not seqs:
        raise ValueError(f"No sequences found in {fasta_path}")

    out_csv = args.output_csv or fasta_path.with_suffix(".metrics.csv")
    summary_path = args.summary_json or fasta_path.with_suffix(".summary.json")

    structure_proxy: StructureProxyModel | None = None
    if not args.skip_structure:
        try:
            structure_proxy = _instantiate_structure_proxy(args.structure_model, args.structure_device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize structure proxy '{args.structure_model}'. "
                "Install the required dependencies or re-run with --skip-structure."
            ) from exc

    rows: list[dict[str, Any]] = []
    rep_scores: list[float] = []
    utility_scores: list[float] = []

    for header, sequence in seqs:
        rep = repetition_metrics(sequence)
        rep_score = repetition_score(sequence)
        length = len(sequence)
        distinct2 = rep["Distinct-2"]
        distinct3 = rep["Distinct-3"]
        distinct23 = (distinct2 + distinct3) / 2.0
        hpoly = rep["R_hpoly"]
        entropy = rep["H_norm"]

        ptm = float("nan")
        plddt = float("nan")
        if structure_proxy is not None:
            result = structure_proxy.evaluate(sequence)
            metrics = result.metrics
            ptm_val = metrics.get("ptm", float("nan"))
            ptm = float(ptm_val) if ptm_val is not None else float("nan")
            raw_plddt = (
                metrics.get("plddt_mean_0_100")
                or metrics.get("plddt")
                or metrics.get("plddt_mean_01")
            )
            raw_plddt_f = float(raw_plddt) if raw_plddt is not None else float("nan")
            plddt = _coerce_plddt(raw_plddt_f)

        rows.append(
            {
                "seq_id": header,
                "sequence": sequence,
                "length": length,
                "entropy_norm": entropy,
                "distinct2": distinct2,
                "distinct3": distinct3,
                "distinct23_avg": distinct23,
                "H_poly_k4": hpoly,
                "ptm": ptm,
                "plddt": plddt,
            }
        )
        rep_scores.append(rep_score)
        if math.isfinite(ptm):
            utility_scores.append(ptm)

    with out_csv.open("w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "seq_id",
                "sequence",
                "length",
                "entropy_norm",
                "distinct2",
                "distinct3",
                "distinct23_avg",
                "H_poly_k4",
                "ptm",
                "plddt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    pid_score: float | None = None
    if len(rows) >= 2:
        pid_score = float(
            pairwise_percent_identity([row["sequence"] for row in rows], return_matrix=False)
        )

    summary = {
        "num_sequences": len(rows),
        "repetition_score": float(np.mean(rep_scores)) if rep_scores else float("nan"),
        "utility_score": float(np.mean(utility_scores)) if utility_scores else float("nan"),
        "diversity_pid": pid_score if pid_score is not None else float("nan"),
    }

    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[INFO] Wrote per-sequence metrics to {out_csv}")
    print(f"[INFO] Dataset summary:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
