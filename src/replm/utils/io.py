# replm/utils/io.py
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """
    Return (header, sequence) tuples from a FASTA file.
    - Header is the string after '>'
    - Sequence will have spaces/tabs removed
    """
    recs: list[tuple[str, str]] = []
    header: str | None = None
    seq_lines: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(seq_lines).replace(" ", "").replace("\t", "")))
                    seq_lines.clear()
                header = line[1:].strip()
            else:
                seq_lines.append(line.strip())
        if header is not None:
            recs.append((header, "".join(seq_lines).replace(" ", "").replace("\t", "")))
    return recs


def write_fasta(seqs: Iterable[tuple[str, str]], path: Path) -> None:
    """Write (header, seq) pairs to *path* in FASTA format. each row contains 60 characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for header, seq in seqs:
            fout.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fout.write(seq[i : i + 60] + "\n")
