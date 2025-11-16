"""Sequence diversity helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

DenominatorMode = Literal["over_alignment", "ignore_gaps", "over_longer", "over_shorter"]


def _needleman_wunsch(
    seq_a: str,
    seq_b: str,
    *,
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    gap_penalty: float = -1.0,
) -> tuple[str, str]:
    a = seq_a.upper()
    b = seq_b.upper()
    n, m = len(a), len(b)

    score = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        score[i, 0] = gap_penalty * i
        trace[i, 0] = 1  # up
    for j in range(1, m + 1):
        score[0, j] = gap_penalty * j
        trace[0, j] = 2  # left

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match_score if a[i - 1] == b[j - 1] else mismatch_score)
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            best = max(diag, up, left)
            score[i, j] = best
            if best == diag:
                trace[i, j] = 0
            elif best == up:
                trace[i, j] = 1
            else:
                trace[i, j] = 2

    align_a: list[str] = []
    align_b: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        direction = trace[i, j]
        if direction == 0:
            align_a.append(a[i - 1])
            align_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif direction == 1:
            align_a.append(a[i - 1])
            align_b.append("-")
            i -= 1
        else:
            align_a.append("-")
            align_b.append(b[j - 1])
            j -= 1

    return "".join(reversed(align_a)), "".join(reversed(align_b))


def _smith_waterman(
    seq_a: str,
    seq_b: str,
    *,
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    gap_penalty: float = -1.0,
) -> tuple[str, str]:
    a = seq_a.upper()
    b = seq_b.upper()
    n, m = len(a), len(b)

    score = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)

    best_score = 0.0
    best_pos = (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match_score if a[i - 1] == b[j - 1] else mismatch_score)
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            best = max(0.0, diag, up, left)
            score[i, j] = best
            if best == 0.0:
                trace[i, j] = 3  # stop
            elif best == diag:
                trace[i, j] = 0
            elif best == up:
                trace[i, j] = 1
            else:
                trace[i, j] = 2

            if best > best_score:
                best_score = best
                best_pos = (i, j)

    if best_pos == (0, 0):
        # fallback to global alignment if no positive score
        return _needleman_wunsch(
            seq_a, seq_b, match_score=match_score, mismatch_score=mismatch_score, gap_penalty=gap_penalty
        )  # noqa: E501

    align_a: list[str] = []
    align_b: list[str] = []
    i, j = best_pos

    while i > 0 and j > 0 and score[i, j] > 0:
        direction = trace[i, j]
        if direction == 0:
            align_a.append(a[i - 1])
            align_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif direction == 1:
            align_a.append(a[i - 1])
            align_b.append("-")
            i -= 1
        elif direction == 2:
            align_a.append("-")
            align_b.append(b[j - 1])
            j -= 1
        else:
            break

    return "".join(reversed(align_a)), "".join(reversed(align_b))


def _percent_identity(
    seq_a: str,
    seq_b: str,
    *,
    local: bool = False,
    denominator_mode: DenominatorMode = "over_alignment",
) -> float:
    if denominator_mode not in {"over_alignment", "ignore_gaps", "over_longer", "over_shorter"}:
        raise ValueError(f"Unsupported denominator_mode '{denominator_mode}'.")

    align_a, align_b = _smith_waterman(seq_a, seq_b) if local else _needleman_wunsch(seq_a, seq_b)

    if not align_a and not align_b:
        return 0.0

    matches = 0
    aligned_columns = len(align_a)
    non_gap_columns = 0
    for ch_a, ch_b in zip(align_a, align_b):
        if ch_a != "-" and ch_b != "-":
            non_gap_columns += 1
            if ch_a == ch_b:
                matches += 1
        elif ch_a == ch_b:  # both gaps (shouldn't happen in standard alignments)
            matches += 1

    if denominator_mode == "over_alignment":
        denom = aligned_columns
    elif denominator_mode == "ignore_gaps":
        denom = non_gap_columns
    elif denominator_mode == "over_longer":
        denom = max(len(seq_a), len(seq_b))
    else:  # over_shorter
        denom = min(len(seq_a), len(seq_b))

    if denom <= 0:
        return 0.0
    return (matches / denom) * 100.0


def pairwise_percent_identity(
    sequences: Sequence[str],
    *,
    return_matrix: bool = False,
    denominator_mode: DenominatorMode = "over_alignment",
    local_alignment: bool = False,
) -> float | np.ndarray:
    """
    Compute pairwise percent identity (PID) across a collection of sequences.

    Args:
        sequences: Iterable of protein strings. At least two sequences required.
        return_matrix: When True, return the full NxN PID matrix (np.ndarray).
        denominator_mode: How to normalize matches (alignment length, ignore_gaps,
            over_longer, over_shorter).
        local_alignment: Use Smith-Waterman (local) instead of Needleman-Wunsch (global).

    Returns:
        Either the mean PID (float) or the symmetric PID matrix (np.ndarray).
    """

    seqs = [str(s or "") for s in sequences]
    n = len(seqs)
    if n < 2:
        raise ValueError("pairwise_percent_identity requires at least two sequences.")

    matrix = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(matrix, 100.0)

    pid_values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            pid = _percent_identity(
                seqs[i],
                seqs[j],
                local=local_alignment,
                denominator_mode=denominator_mode,
            )
            matrix[i, j] = matrix[j, i] = pid
            pid_values.append(pid)

    if return_matrix:
        return matrix
    return float(np.mean(pid_values))


__all__ = ["pairwise_percent_identity"]
