#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

# ===== seeds & datasets =====
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["cath", "uniref50", "scop"]

# ===== positive datasets for progen2_base =====
POSITIVE_DATASETS = {
    "cath": {
        "pos_fasta": "data/pos/cath.fa",
        "pos_metrics": "data/pos/cath.metrics.csv",
    },
    "uniref50": {
        "pos_fasta": "data/pos/uniprot.fa",
        "pos_metrics": "data/pos/uniprot.metrics.csv",
    },
    "scop": {
        "pos_fasta": "data/pos/scop.fa",
        "pos_metrics": "data/pos/scop.metrics.csv",
    },
}

# ===== negative dataset for progen2_base=====
NEGATIVE_DATA = {
    "neg_fasta": "data/neg/progen2_base_neg.fasta",
    "neg_metrics": "data/neg/progen2_base_neg.metrics.csv",
}

# ===== base generation setup =====
BASE_OVERRIDES = [
    "models=progen2_base",
    "generation.uncond.n=100",
    "generation.prefix.n=100",
    "split.train=100",
    "split.test=100",
    # should be larger than train+test since train and test are selected from the pool
    "dataset.opt.target_per_side=1000",
    "generation.uncond.length_min=50",
    "generation.uncond.length_max=512",
    "generation.prefix.prefix_frac=0.1",
]


def _float_label(val: float, digits: int = 2) -> str:
    text = f"{val:.{digits}f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text


def _normalize_overrides(overrides: Iterable[str]) -> list[str]:
    """If the key is *.overrides.* and does not start with '+', add '+' (Hydra's append semantic)."""
    processed: list[str] = []
    for ov in overrides:
        if ".overrides." in ov and not ov.startswith("+"):
            processed.append(f"+{ov}")
        else:
            processed.append(ov)
    return processed


def build_base_methods() -> list[tuple[str, list[str]]]:
    methods: list[tuple[str, list[str]]] = []

    # Control baseline
    methods.append(
        (
            "control_default",
            [
                "methods=control",
            ],
        )
    )

    # Temperature sweep
    for val in [0.7, 1.0, 1.3]:
        label = _float_label(val).replace(".", "p")
        methods.append(
            (
                f"temperature_{label}",
                [
                    "methods=control",
                    f"generation.uncond.overrides.temperature={val}",
                    f"generation.prefix.overrides.temperature={val}",
                ],
            )
        )

    # Top-p sampling sweep
    for val in [0.80, 0.85, 0.90, 0.95, 0.98, 1.00]:
        label = _float_label(val, digits=2).replace(".", "p")
        methods.append(
            (
                f"top_p_{label}",
                [
                    "methods=control",
                    f"generation.uncond.overrides.top_p={val}",
                    f"generation.prefix.overrides.top_p={val}",
                ],
            )
        )

    # No repeat n-gram constraint
    for size in [2, 3, 4]:
        methods.append(
            (
                f"no_repeat_ngram_{size}",
                [
                    "methods=control",
                    f"generation.uncond.overrides.no_repeat_ngram_size={size}",
                    f"generation.prefix.overrides.no_repeat_ngram_size={size}",
                ],
            )
        )

    # Repetition penalty sweep
    for val in [1.1, 1.2, 1.3]:
        label = _float_label(val, digits=2).replace(".", "p")
        methods.append(
            (
                f"repetition_penalty_{label}",
                [
                    "methods=control",
                    f"generation.uncond.overrides.repetition_penalty={val}",
                    f"generation.prefix.overrides.repetition_penalty={val}",
                ],
            )
        )

    # Neuron deactivation (NeuronTopK)
    for topk in [8, 64, 256, 1024, 4096]:
        methods.append(
            (
                f"neuron_deactivation_{topk}",
                [
                    "methods=neuron_topk",
                    f"methods.topk={topk}",
                ],
            )
        )

    # UUCS (contrastive layer) sweep over layers 0-26
    for layer in range(27):
        methods.append(
            (
                f"uccs_layer{layer:02d}",
                [
                    "methods=contrastive_layer",
                    f"methods.layer={layer}",
                ],
            )
        )

    # Probe steering sweep over layers 0-26
    for layer in range(27):
        methods.append(
            (
                f"probe_layer{layer:02d}",
                [
                    "methods=probe_layer",
                    f"methods.layer={layer}",
                ],
            )
        )

    return methods


def _dataset_overrides(dataset: str) -> list[str]:
    """Inject the positive and negative sample file paths into the dataset.* overrides."""
    if dataset not in POSITIVE_DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Known: {list(POSITIVE_DATASETS)}")

    pos = POSITIVE_DATASETS[dataset]
    neg = NEGATIVE_DATA
    return [
        f"dataset.pos_fasta={pos['pos_fasta']}",
        f"dataset.pos_metrics={pos['pos_metrics']}",
        f"dataset.neg_fasta={neg['neg_fasta']}",
        f"dataset.neg_metrics={neg['neg_metrics']}",
    ]


def build_experiments() -> list[dict[str, object]]:
    """
    Expand on (seed * dataset * method):
      id: seed_{seed}_method_{method}_dataset_{dataset}
      overrides: BASE_OVERRIDES + runtime + dataset_paths + method_overrides
    """
    exps: list[dict[str, object]] = []
    methods = build_base_methods()

    for dataset in DATASETS:
        for seed in SEEDS:
            for method_name, method_overrides in methods:
                exp_id = f"seed_{seed}_method_{method_name}_progen2_dataset_{dataset}"

                overrides = (
                    BASE_OVERRIDES
                    + [f"runtime.seed={seed}", "runtime.device=cuda"]
                    + _dataset_overrides(dataset)
                    + method_overrides
                )

                exps.append(
                    {
                        "id": exp_id,
                        "overrides": _normalize_overrides(overrides),
                    }
                )

    return exps


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate steering experiments.")
    parser.add_argument("--index", type=int, default=None, help="Return a single spec by index.")
    parser.add_argument("--json", action="store_true", help="Dump the entire sweep as JSON.")
    args = parser.parse_args()

    experiments = build_experiments()

    if args.json:
        print(json.dumps(experiments, indent=2))
        return

    if args.index is not None:
        if args.index < 0 or args.index >= len(experiments):
            raise IndexError(f"Index {args.index} out of range for {len(experiments)} experiments.")
        exp = experiments[args.index]
        overrides = ",".join(exp["overrides"])
        print(f"{exp['id']}::{overrides}")
        return

    for exp in experiments:
        overrides = ",".join(exp["overrides"])
        print(f"{exp['id']}::{overrides}")


if __name__ == "__main__":
    main()
