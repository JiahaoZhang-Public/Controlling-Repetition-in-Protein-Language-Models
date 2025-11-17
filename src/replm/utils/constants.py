AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")  # 20 amino acids


def normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("`sequence` must be a non-empty string.")
    # remove all non-AA letters
    sequence = "".join(ch for ch in sequence if ch in AA_LETTERS)
    return sequence.replace(" ", "").replace("\n", "").upper()
