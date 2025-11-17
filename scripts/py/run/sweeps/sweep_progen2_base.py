#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable


def _float_label(val: float, digits: int = 2) -> str:
    text = f"{val:.{digits}f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text


def build_experiments() -> list[dict[str, object]]:
    exps: list[dict[str, object]] = []

    def add(exp_id: str, overrides: Iterable[str]) -> None:
        processed = []
        for ov in overrides:
            if ".overrides." in ov and not ov.startswith("+"):
                processed.append(f"+{ov}")
            else:
                processed.append(ov)
        exps.append({"id": exp_id, "overrides": processed})

    # Control baseline
    add("control_default", ["models=progen2_base", "methods=control", 
                            "generation.uncond.n=100", 
                            "generation.prefix.n=100",
                            "split.train=1000",
                            "split.test=10",
                            "generation.uncond.length_min=50", 
                            "generation.uncond.length_max=512", 
                            "generation.prefix.prefix_frac=0.1"])

    # Temperature sweep
    for val in [0.7, 1.0, 1.3]:
        label = _float_label(val).replace(".", "p")
        add(
            f"temperature_{label}",
            [
                "methods=control",
                f"generation.uncond.overrides.temperature={val}",
                f"generation.prefix.overrides.temperature={val}",
            ],
        )

    # Top-p sampling sweep
    for val in [0.80, 0.85, 0.90, 0.95, 0.98, 1.00]:
        label = _float_label(val, digits=2).replace(".", "p")
        add(
            f"top_p_{label}",
            [
                "methods=control",
                f"generation.uncond.overrides.top_p={val}",
                f"generation.prefix.overrides.top_p={val}",
            ],
        )

    # No repeat n-gram constraint
    for size in [2, 3, 4]:
        add(
            f"no_repeat_ngram_{size}",
            [
                "methods=control",
                f"generation.uncond.overrides.no_repeat_ngram_size={size}",
                f"generation.prefix.overrides.no_repeat_ngram_size={size}",
            ],
        )

    # Repetition penalty sweep
    for val in [1.1, 1.2, 1.3]:
        label = _float_label(val, digits=2).replace(".", "p")
        add(
            f"repetition_penalty_{label}",
            [
                "methods=control",
                f"generation.uncond.overrides.repetition_penalty={val}",
                f"generation.prefix.overrides.repetition_penalty={val}",
            ],
        )

    # Neuron deactivation (NeuronTopK)
    for topk in [8, 64, 256, 1024, 4096]:
        add(
            f"neuron_deactivation_{topk}",
            [
                "methods=neuron_topk",
                f"methods.topk={topk}",
            ],
        )

    # UUCS (contrastive layer) sweep over layers 0-26
    for layer in range(27):
        add(
            f"uccs_layer{layer:02d}",
            [
                "methods=contrastive_layer",
                f"methods.layer={layer}",
            ],
        )

    # Probe steering sweep over layers 0-26
    for layer in range(27):
        add(
            f"probe_layer{layer:02d}",
            [
                "methods=probe_layer",
                f"methods.layer={layer}",
            ],
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
