# Toy Protein Dataset Provider
# will iterave some toy protein sequences and return them as a dataset
from collections.abc import Iterable
from pathlib import Path

from .base import DatasetProvider


class ToyProteinDataset(DatasetProvider):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def build(self, out_dir: Path) -> dict:
        return {}

    def iter_pos(self) -> Iterable[tuple[str, str]]:
        return [("pos_1", "MALWMRLLPLLALLALWGPDPAAA"), ("pos_2", "MALWMRLLPLLALLALWGPDPPPP")]

    def iter_neg(self) -> Iterable[tuple[str, str]]:
        return [("neg_1", "AAAAAAAAAA"), ("neg_2", "CCCCCCCCCC")]
