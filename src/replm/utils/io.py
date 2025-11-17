# repsurf/utils/io.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """
    Return (header, sequence) tuples from a FASTA file.
    - Header is the string after '>'
    - Sequence will have spaces/tabs removed
    """
    recs: List[Tuple[str, str]] = []
    header: str | None = None
    seq_lines: List[str] = []
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


def write_fasta(seqs: Iterable[Tuple[str, str]], path: Path) -> None:
    """Write (header, seq) pairs to *path* in FASTA format. each row contains 60 characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for header, seq in seqs:
            fout.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fout.write(seq[i:i+60] + "\n")
                
                
