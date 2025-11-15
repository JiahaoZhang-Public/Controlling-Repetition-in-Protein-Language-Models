"""
src/repetition_control_plm/metrics/repetition.py

Official implementation of repetition evaluation metrics for protein sequences:
- Token-level entropy (H_norm)
- n-gram diversity (Distinct-n)
- Homopolymer diversity score (R_hpoly)

Formulas (from the paper excerpt in the prompt):

1) Token-level entropy (normalized):
   H_norm(x) = - [ sum_{a in A} p(a) log2 p(a) ] / log2 |A|,  with H_norm in [0, 1].

2) n-gram diversity:
   Distinct-n(x) = |uniq_ngrams_n(x)| / |ngrams_n(x)|.

3) Homopolymer diversity score:
   R_hpoly(x) = 1 - (1/T) * sum_i [ l_i * 1(l_i >= k) ],
   where l_i is the length of the i-th homopolymer run, k is a threshold (default k=4),
   and T is the sequence length. Higher is better (fewer long homopolymer runs).

Notes
-----
- All metrics are designed so that higher scores indicate *more* diversity / less collapse.
- Sequence `x` is a string over the amino-acid alphabet. Lower/upper case are both accepted.
- Any characters outside the alphabet are ignored for unigram probabilities; they still
  contribute to n-gram counting if present in the string. You can override this by passing
  an explicit `alphabet` argument.
"""
from __future__ import annotations

import math
import itertools
from collections import Counter
from typing import List, Sequence

from ..utils.constants import AA_LETTERS, normalize_sequence 


def token_level_entropy(seq: str, alphabet: Sequence[str] | None = None) -> float:
    """Compute normalized token-level Shannon entropy H_norm in [0, 1].

    Parameters
    ----------
    seq : str
        Amino-acid sequence.
    alphabet : Sequence[str], optional
        Alphabet A to use. Defaults to `AA_LETTERS` (20 canonical AAs).

    Returns
    -------
    float
        Normalized entropy H_norm(x).
    """
    if alphabet is None:
        alphabet = AA_LETTERS
    seq = normalize_sequence(seq)

    # Count only characters that are part of the alphabet for the unigram distribution.
    counts = Counter(ch for ch in seq if ch in alphabet)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    # Shannon entropy base 2.
    entropy = 0.0
    for a in alphabet:
        c = counts.get(a, 0)
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log(p, 2)

    # Normalize by log2(|A|).
    denom = math.log(len(alphabet), 2)
    return float(entropy / denom) if denom > 0 else 0.0


def _ngrams(seq: str, n: int) -> List[str]:
    return [seq[i : i + n] for i in range(0, len(seq) - n + 1)]


def distinct_n(seq: str, n: int) -> float:
    """Compute Distinct-n = |unique n-grams| / |all n-grams|.

    If the sequence is shorter than n, returns 0.0 by convention.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    seq = normalize_sequence(seq)
    grams = _ngrams(seq, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def homopolymer_diversity(seq: str, k: int = 4) -> float:
    """Compute the homopolymer diversity score R_hpoly as defined above.

    Parameters
    ----------
    seq : str
        Amino-acid sequence.
    k : int, default=4
        Threshold length for a homopolymer run to be considered 'long'.

    Returns
    -------
    float
        R_hpoly(x) in [0, 1].
    """
    if k <= 1:
        # If k<=1, the indicator triggers for all positions and score collapses to 0.
        # Guard here to keep the metric meaningful.
        k = 2

    seq = normalize_sequence(seq)
    T = len(seq)
    if T == 0:
        return 0.0

    long_run_tokens = 0
    for ch, group in itertools.groupby(seq):
        run_len = sum(1 for _ in group)
        if run_len >= k:
            long_run_tokens += run_len

    frac = long_run_tokens / T
    return float(1.0 - frac)


def repetition_metrics(seq: str, k: int = 4, *, alphabet: Sequence[str] | None = None) -> dict:
    """Convenience wrapper returning all three metrics.

    Returns
    -------
    dict with keys: 'H_norm', 'Distinct-2', 'Distinct-3', 'R_hpoly'
    """
    return {
        "H_norm": token_level_entropy(seq, alphabet=alphabet),
        "Distinct-2": distinct_n(seq, 2),
        "Distinct-3": distinct_n(seq, 3),
        "R_hpoly": homopolymer_diversity(seq, k=k),
    }


def repetition_score(seq: str, k: int = 4, *, alphabet: Sequence[str] | None = None) -> float:
    """Aggregate the three metrics into a single score.
    
    R(x) = (H_norm + (Distinct-2 + Distinct-3) / 2 + R_hpoly) / 3

    Returns
    -------
    float
        Aggregated repetition score.
    """
    m = repetition_metrics(seq, k=k, alphabet=alphabet)
    return (m["H_norm"] + (m["Distinct-2"] + m["Distinct-3"]) / 2.0 + m["R_hpoly"]) / 3.0
