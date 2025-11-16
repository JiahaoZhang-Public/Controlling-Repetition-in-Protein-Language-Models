from __future__ import annotations

import numpy as np

from replm.metrics.diversity import pairwise_percent_identity


def test_pairwise_pid_mean_default_global():
    seqs = ["AAAA", "AAAT", "TTTT"]
    # Expected: (75 + 0 + 25) / 3 = 100 / 3
    pid = pairwise_percent_identity(seqs)
    assert abs(pid - (100 / 3)) < 1e-6


def test_pairwise_pid_denominator_modes():
    seq_a = "ACGT"
    seq_b = "ACG"

    matrix = pairwise_percent_identity(
        [seq_a, seq_b], return_matrix=True, denominator_mode="over_alignment"
    )
    assert np.isclose(matrix[0, 1], 75.0)

    matrix = pairwise_percent_identity([seq_a, seq_b], return_matrix=True, denominator_mode="ignore_gaps")
    assert np.isclose(matrix[0, 1], 100.0)

    matrix = pairwise_percent_identity([seq_a, seq_b], return_matrix=True, denominator_mode="over_longer")
    assert np.isclose(matrix[0, 1], 75.0)

    matrix = pairwise_percent_identity([seq_a, seq_b], return_matrix=True, denominator_mode="over_shorter")
    assert np.isclose(matrix[0, 1], 100.0)


def test_pairwise_pid_local_alignment():
    seqs = ["NNNNAAA", "AAA"]
    matrix = pairwise_percent_identity(seqs, return_matrix=True, local_alignment=True)
    assert np.isclose(matrix[0, 1], 100.0)
