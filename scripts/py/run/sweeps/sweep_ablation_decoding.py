#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

# fixed random seeds
SEEDS = [0, 1, 2, 3, 4]

# ====== Base generation setup ======
BASE_OVERRIDES = [
    "generation.uncond.n=100",
    "generation.prefix.n=100",
    "split.train=100",
    "split.test=100",
    # this should be larger than train+test since train and test are selected from the pool
    "dataset.opt.target_per_side=1000",
    "generation.uncond.length_min=50",
    "generation.uncond.length_max=512",
    "generation.prefix.prefix_frac=0.1",
]

# ====== Dataset definitions ======
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

ESM3_NEGATIVE = {
    "neg_fasta": "data/neg/esm3_raw_neg.sequences.fasta",
    "neg_metrics": "data/neg/esm3_raw_neg.metrics.csv",
}

PROTGPT2_NEGATIVE = {
    "neg_fasta": "data/neg/gpt_neg.sequences.fasta",
    "neg_metrics": "data/neg/gpt_neg.metrics.csv",
}


def _with_override_prefix(overrides: Iterable[str]) -> list[str]:
    processed: list[str] = []
    for ov in overrides:
        if ".overrides." in ov and not ov.startswith("+"):
            processed.append(f"+{ov}")
        else:
            processed.append(ov)
    return processed


def build_experiments() -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = []

    def add_experiment(exp_id: str, overrides: Iterable[str]) -> None:
        experiments.append({"id": exp_id, "overrides": _with_override_prefix(overrides)})

    # ----- ESM3 -----
    esm3_methods: list[tuple[str, list[str]]] = [
        (
            "temperature_esm3",
            [
                "generation.uncond.overrides.temperature=1.3",
                "generation.prefix.overrides.temperature=1.3",
            ],
        ),
        (
            "top_p_esm3",
            [
                "generation.uncond.overrides.top_p=0.95",
                "generation.prefix.overrides.top_p=0.95",
            ],
        ),
        (
            "entropy_esm3",  # schedule = cosine, strategy = entropy, temperature annealing = False
            [
                "generation.uncond.overrides.strategy=entropy",
                "generation.prefix.overrides.strategy=entropy",
                "generation.uncond.overrides.temperature_annealing=False",
                "generation.prefix.overrides.temperature_annealing=False",
                "generation.uncond.overrides.schedule=cosine",
                "generation.prefix.overrides.schedule=cosine",
            ],
        ),
    ]

    for method_name, method_overrides in esm3_methods:
        for dataset_name, dataset_paths in POSITIVE_DATASETS.items():
            for seed in SEEDS:
                exp_id = f"seed_{seed}_method_{method_name}_dataset_{dataset_name}"
                overrides = (
                    [
                        "models=esm3",
                        "methods=contrastive_layer",
                        "runtime.device=cuda",
                        f"runtime.seed={seed}",
                        "steering.layer=47",
                        f"dataset.pos_fasta={dataset_paths['pos_fasta']}",
                        f"dataset.pos_metrics={dataset_paths['pos_metrics']}",
                        f"dataset.neg_fasta={ESM3_NEGATIVE['neg_fasta']}",
                        f"dataset.neg_metrics={ESM3_NEGATIVE['neg_metrics']}",
                    ]
                    + BASE_OVERRIDES
                    + method_overrides
                )
                add_experiment(exp_id, overrides)

    # ----- ProtGPT2 -----
    progen_methods: list[tuple[str, list[str]]] = [
        (
            "temperature_protgpt2",
            [
                "generation.uncond.overrides.temperature=1.3",
                "generation.prefix.overrides.temperature=1.3",
            ],
        ),
        (
            "top_p_protgpt2",
            [
                "generation.uncond.overrides.top_p=0.98",
                "generation.prefix.overrides.top_p=0.98",
            ],
        ),
        (
            "repetition_penalty_protgpt2",
            [
                "generation.uncond.overrides.repetition_penalty=1.2",
                "generation.prefix.overrides.repetition_penalty=1.2",
            ],
        ),
        (
            "no_repeat_ngram_protgpt2",
            [
                "generation.uncond.overrides.no_repeat_ngram_size=3",
                "generation.prefix.overrides.no_repeat_ngram_size=3",
            ],
        ),
    ]

    for method_name, method_overrides in progen_methods:
        for dataset_name, dataset_paths in POSITIVE_DATASETS.items():
            for seed in SEEDS:
                exp_id = f"seed_{seed}_method_{method_name}_dataset_{dataset_name}"
                overrides = (
                    [
                        "models=protgpt2",
                        "methods=contrastive_layer",
                        "runtime.device=cuda",
                        f"runtime.seed={seed}",
                        "steering.layer=30",
                        f"dataset.pos_fasta={dataset_paths['pos_fasta']}",
                        f"dataset.pos_metrics={dataset_paths['pos_metrics']}",
                        f"dataset.neg_fasta={PROTGPT2_NEGATIVE['neg_fasta']}",
                        f"dataset.neg_metrics={PROTGPT2_NEGATIVE['neg_metrics']}",
                    ]
                    + BASE_OVERRIDES
                    + method_overrides
                )
                add_experiment(exp_id, overrides)

    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate ablation decoding/steering experiments.")
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
