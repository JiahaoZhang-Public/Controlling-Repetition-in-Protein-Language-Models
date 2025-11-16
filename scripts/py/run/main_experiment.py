# scripts/py/run/main_experiment.py
#!/usr/bin/env python3
"""
Unified steering experiment pipeline (without evaluation metrics).

Pipeline:
  1) Load dataset (pos/neg sequences) via Hydra-configured provider.
  2) Split into train/test sets.
  3) Extract activations from the configured model backend.
  4) Fit the steering method to obtain AffineEdit objects.
  5) Generate steered sequences (unconditional + prefix-conditioned) and save artifacts.
"""
from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from replm.config import BackendConfig
from replm.metrics.repetition import token_level_entropy
from replm.models import get_model_class
from replm.steer.io import save_steer_result
from replm.steer.methods.base import ActivationBatch
from replm.steer.methods.base import InputSpec as MethodRequirements
from replm.steer.steerer import Steerer


@dataclass
class SequenceRecord:
    seq_id: str
    sequence: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_fasta(path: Path, sequences: Sequence[SequenceRecord], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in sequences:
            handle.write(f">{tag}_{rec.seq_id}\n")
            seq = rec.sequence.strip().upper()
            for i in range(0, len(seq), 60):
                handle.write(seq[i : i + 60] + "\n")


def _records_from_iter(items: Iterable) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            seq_id = str(item.get("seq_id", idx))
            seq = str(item["sequence"])
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            seq_id = str(item[0])
            seq = str(item[1])
        else:
            raise ValueError(f"Unsupported dataset item format: {type(item)!r}")
        records.append(SequenceRecord(seq_id=seq_id, sequence=seq))
    return records


def _split_records(
    records: list[SequenceRecord],
    n_train: int,
    n_test: int,
    *,
    rng: random.Random,
) -> tuple[list[SequenceRecord], list[SequenceRecord]]:
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n_tr = min(n_train, len(records))
    train_idx = idxs[:n_tr]
    remain = idxs[n_tr:]
    n_te = min(n_test, len(remain))
    test_idx = remain[:n_te]
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


def _activations_to_by_layer(
    acts: torch.Tensor,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    tensor = acts.detach().cpu()
    by_layer: dict[int, torch.Tensor] = {}
    for pos, layer_idx in enumerate(layers):
        by_layer[int(layer_idx)] = tensor[:, pos, :].to(torch.float32)
    return by_layer


def _build_supervision_vector(records: Sequence[SequenceRecord]) -> torch.Tensor:
    vals = [token_level_entropy(rec.sequence) for rec in records]
    return torch.tensor(vals, dtype=torch.float32)


def _summarize_edits(edits_by_layer) -> list[dict]:
    summary = []
    for layer, edits in (edits_by_layer or {}).items():
        total_sparse = sum(0 if edit.dims is None else int(edit.dims.numel()) for edit in edits)
        summary.append(
            {
                "layer": int(layer),
                "num_edits": len(edits),
                "sparse_dims": total_sparse,
                "has_dense": any(edit.dims is None for edit in edits),
            }
        )
    return summary


@hydra.main(config_path="../../../configs", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    seed = int(cfg.runtime.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    origin = Path(get_original_cwd())
    root = _ensure_dir(origin / cfg.io.root / cfg.exp.id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _ensure_dir(root / f"run_{timestamp}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "experiment.log", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("replm.main_experiment")
    logger.info("Artifacts: %s", run_dir)

    OmegaConf.save(cfg, run_dir / "config.yaml")

    # ----- dataset -----
    def _instantiate_dataset(cfg_dataset: DictConfig):
        if OmegaConf.is_config(cfg_dataset) and "_target_" in cfg_dataset:
            return instantiate(cfg_dataset)
        provider_cfg = cfg_dataset.get("provider")
        if provider_cfg is None:
            raise ValueError("dataset config must define either `_target_` or `provider`.")
        return instantiate(provider_cfg)

    dataset = _instantiate_dataset(cfg.dataset)
    dataset.build(run_dir / "dataset")
    pos_records = _records_from_iter(dataset.iter_pos())
    neg_records = _records_from_iter(dataset.iter_neg())
    logger.info("Loaded dataset with %d positive / %d negative sequences.", len(pos_records), len(neg_records)) # noqa: E501

    rng = random.Random(seed)
    pos_train, pos_test = _split_records(pos_records, cfg.split.train, cfg.split.test, rng=rng)
    neg_train, neg_test = _split_records(neg_records, cfg.split.train, cfg.split.test, rng=rng)

    if not pos_train or not neg_train:
        raise ValueError("Training split must contain at least one positive and one negative sequence.")

    _write_json(
        run_dir / "dataset_split.json",
        {
            "dataset_target": str(cfg.dataset.get("_target_", cfg.dataset.get("provider", "unknown"))),
            "pos_train": [rec.seq_id for rec in pos_train],
            "pos_test": [rec.seq_id for rec in pos_test],
            "neg_train": [rec.seq_id for rec in neg_train],
            "neg_test": [rec.seq_id for rec in neg_test],
        },
    )

    # ----- model backend -----
    backend_cfg = BackendConfig(**OmegaConf.to_container(cfg.models.backend, resolve=True))
    backend_cls = get_model_class(cfg.models.name)
    init_cfg = instantiate(cfg.models.init) if "init" in cfg.models else None
    gen_cfg = instantiate(cfg.models.generation) if "generation" in cfg.models else None
    backend = backend_cls(backend_cfg=backend_cfg, init_cfg=init_cfg, gen_cfg=gen_cfg)
    backend.load()
    layers = list(cfg.model.layers)
    batch_size = int(cfg.model.activation.batch_size)
    logger.info("Extracting activations for layers=%s", layers)

    pos_hidden = backend.activations([rec.sequence for rec in pos_train], layers=layers, batch_size=batch_size) # noqa: E501
    neg_hidden = backend.activations([rec.sequence for rec in neg_train], layers=layers, batch_size=batch_size) # noqa: E501
    by_layer = {}
    pos_dict = _activations_to_by_layer(pos_hidden, layers)
    neg_dict = _activations_to_by_layer(neg_hidden, layers)
    for layer in layers:
        by_layer[layer] = torch.cat([pos_dict[layer], neg_dict[layer]], dim=0)

    if bool(cfg.io.save_activations):
        np.savez(
            run_dir / "activations.npz",
            **{f"pos_layer_{layer}": pos_dict[layer].numpy() for layer in layers},
            **{f"neg_layer_{layer}": neg_dict[layer].numpy() for layer in layers},
        )

    pos_idx = torch.arange(0, len(pos_train), dtype=torch.long)
    neg_idx = torch.arange(len(pos_train), len(pos_train) + len(neg_train), dtype=torch.long)
    y_vec = torch.cat(
        [_build_supervision_vector(pos_train), _build_supervision_vector(neg_train)],
        dim=0,
    )
    batch = ActivationBatch(by_layer=by_layer, y=y_vec, positive_idx=pos_idx, negative_idx=neg_idx)

    # ----- steering method -----
    method = instantiate(cfg.methods)
    reqs: MethodRequirements = method.requires()
    if reqs.need_y and batch.y is None:
        raise ValueError("Selected steering method requires supervision (y) but none was provided.")
    logger.info("Fitting steering method: %s", cfg.methods._target_)
    result = method.fit(batch)
    edits = [edit for edits in (result.by_layer or {}).values() for edit in edits]
    logger.info("Produced %d affine edits across %d layers.", len(edits), len(result.by_layer or {}))
    _write_json(run_dir / "steer_summary.json", _summarize_edits(result.by_layer))
    save_steer_result(result, run_dir / "steer_result.json")

    # ----- generation -----
    layer_path = tuple(cfg.model.layer_attr_path)
    uncond_cfg = OmegaConf.to_container(cfg.generation.uncond.overrides, resolve=True)
    prefix_cfg = OmegaConf.to_container(cfg.generation.prefix.overrides, resolve=True)

    def _generate_uncond() -> list[SequenceRecord]:
        outputs: list[SequenceRecord] = []
        with torch.no_grad():
            with Steerer(backend.model, specs=edits, layer_attr_path=layer_path):
                for i in range(cfg.generation.uncond.n):
                    length = random.randint(cfg.generation.uncond.length_min, cfg.generation.uncond.length_max) # noqa: E501
                    seq = backend.generate_uncond(length=int(length), **(uncond_cfg or {}))
                    outputs.append(SequenceRecord(seq_id=f"uncond_{i}", sequence=seq))
        return outputs

    def _generate_prefix() -> list[tuple[SequenceRecord, SequenceRecord]]:
        count = min(cfg.generation.prefix.n, len(pos_test))
        subset = random.sample(pos_test, count) if count and len(pos_test) >= count else pos_test[:count]
        outputs: list[tuple[SequenceRecord, SequenceRecord]] = []
        with torch.no_grad():
            with Steerer(backend.model, specs=edits, layer_attr_path=layer_path):
                for src in subset:
                    L = len(src.sequence)
                    pref_len = max(1, int(cfg.generation.prefix.prefix_frac * L))
                    prefix = src.sequence[:pref_len]
                    generated = backend.generate_with_prefix(
                        target_len=L,
                        prefix=prefix,
                        **(prefix_cfg or {}),
                    )
                    outputs.append(
                        (
                            SequenceRecord(seq_id=f"prefix_{src.seq_id}", sequence=generated),
                            SequenceRecord(seq_id=src.seq_id, sequence=src.sequence),
                        )
                    )
        return outputs

    uncond = _generate_uncond()
    prefix_pairs = _generate_prefix()

    _write_fasta(run_dir / "uncond.steer.fasta", uncond, "uncond")
    prefix_outputs = [pair[0] for pair in prefix_pairs]
    _write_fasta(run_dir / "prefix.steer.fasta", prefix_outputs, "prefix")
    _write_fasta(run_dir / "prefix.sources.fasta", [pair[1] for pair in prefix_pairs], "source")

    logger.info("Generated %d unconditional and %d prefix sequences.", len(uncond), len(prefix_pairs))


if __name__ == "__main__":
    main()
