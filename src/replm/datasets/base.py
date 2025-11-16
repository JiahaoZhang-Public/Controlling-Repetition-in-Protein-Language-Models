# src/replm/datasets/base.py
"""
Base abstract class for dataset providers.

A dataset provider is responsible for building a dataset and providing iterators for
positive and negative examples.

The dataset is built into a manifest file that records the metadata of the dataset.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path


class DatasetProvider(ABC):
    def __init__(self, **cfg):
        self.cfg = cfg

    @abstractmethod
    def build(self, out_dir: Path) -> dict:
        """Generate manifest into out_dir and return stats dict."""

    @abstractmethod
    def iter_pos(self) -> Iterable[tuple[str, str]]: ...
    @abstractmethod
    def iter_neg(self) -> Iterable[tuple[str, str]]: ...
