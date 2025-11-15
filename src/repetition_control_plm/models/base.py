# src/repetition_control_plm/models/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ModelBackend(ABC):
    def __init__(self, **cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.device = cfg.get("device", "cpu")
        self.layer_attr_path = cfg.get("layer_attr_path", ("transformer", "blocks"))
        self.output_index = cfg.get("output_index", 0)

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def activations(self, sequences: List[str], layers: List[int] | None, *, batch_size: int = 8) -> Dict[str, Dict[int, Any]]:
        """Returns {seq_id: {layer: np.ndarray or torch.Tensor}}"""

    @abstractmethod
    def generate_uncond(self, length: int, **gen_cfg) -> str:
        """Returns a generated sequence of length `length`."""

    @abstractmethod
    def generate_with_prefix(self, target_len: int, prefix: str, **gen_cfg) -> str: 
        """Returns a generated sequence of length `target_len` with `prefix`."""