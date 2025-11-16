AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")  # 20 amino acids


def normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("`sequence` must be a non-empty string.")
    return sequence.replace(" ", "").replace("\n", "").upper()
