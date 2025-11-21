"""Summarize pairwise percent identity for collections of FASTA files."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

from replm.metrics.diversity import pairwise_percent_identity
from replm.utils.io import read_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-level pairwise percent identity (PID) for each FASTA file "
            "and store results in a CSV table."
        )
    )
    parser.add_argument(
        "fastas",
        nargs="+",
        type=Path,
        help="Paths to FASTA files to analyze.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/post_analysis/fasta_pid_summary.csv"),
        help="Path to the CSV file that will store the PID summary.",
    )
    parser.add_argument(
        "--denominator-mode",
        choices=["over_alignment", "ignore_gaps", "over_longer", "over_shorter"],
        default="over_alignment",
        help="How to normalize matches when computing percent identity.",
    )
    parser.add_argument(
        "--local-alignment",
        action="store_true",
        help="Use local (Smith-Waterman) alignment instead of global (Needleman-Wunsch).",
    )
    return parser.parse_args()


def sequences_from_fasta(path: Path) -> list[str]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    records = read_fasta(path)
    sequences = [seq for _, seq in records if seq]
    if len(sequences) < 2:
        raise ValueError(f"{path} contains fewer than two sequences; cannot compute PID.")
    return sequences


def compute_pid(
    sequences: Iterable[str],
    *,
    denominator_mode: str,
    local_alignment: bool,
) -> float:
    return float(
        pairwise_percent_identity(
            list(sequences),
            return_matrix=False,
            denominator_mode=denominator_mode,
            local_alignment=local_alignment,
        )
    )


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str | float | int]] = []

    for fasta_path in args.fastas:
        seqs = sequences_from_fasta(fasta_path)
        pid = compute_pid(
            seqs,
            denominator_mode=args.denominator_mode,
            local_alignment=args.local_alignment,
        )
        rows.append(
            {
                "fasta_path": str(Path(fasta_path).expanduser().resolve()),
                "num_sequences": len(seqs),
                "mean_pid": pid,
            }
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fasta_path", "num_sequences", "mean_pid"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote PID summary for {len(rows)} FASTA files to {output_path}")


if __name__ == "__main__":
    main()
