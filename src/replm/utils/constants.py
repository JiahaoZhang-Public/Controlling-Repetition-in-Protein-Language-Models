AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")    # 20 amino acids

def normalize_sequence(seq: str) -> str:
    """Uppercase and strip whitespace."""
    return "".join(seq.split()).upper() 