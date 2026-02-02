## What’s included

- Positive pools used in the paper: `pos/cath.fa`, `pos/scop.fa`, `pos/uniprot.fa` with matching `*.metrics.csv` files (entropy, pLDDT/PTM, length). `pos/combined.metrics.csv` concatenates the three tables.
- Negative pools sampled from generative LMs: `neg/{esm3,protgpt2,progen2,esm2}_neg.fasta` with precomputed metrics in the matching CSVs.
- A tiny smoke dataset under `example/` for quick sanity checks.

The FASTA/CSV pairs are already filtered the same way as in the ICLR 2026 paper, so you can run the main experiment scripts without any preprocessing.

## Regenerating or refreshing datasets

1) **Generate new negative pools** (optional):

```bash
python scripts/py/run/generate_negatives.py \
  --models esm3 protgpt2 progen2_small progen2_base \
  --samples-per-model 10000 \
  --min-len 50 --max-len 1024 \
  --output-dir data/neg \
  --seed 42 --device cuda
```

2) **Recompute metrics** for any FASTA (structure scoring uses ESM3; add `--skip-structure` if you only need repetition/diversity):

```bash
python scripts/py/run/evaluate_sequences.py data/neg/esm3_neg.fasta \
  --output-csv data/neg/esm3_neg.metrics.csv \
  --summary-json data/neg/esm3_neg.summary.json \
  --structure-model esm3 --structure-device cuda
```

3) **Wire new pools into Hydra configs** by overriding `dataset.pos_*` / `dataset.neg_*` in your run command (see `configs/dataset/posneg.yaml`).

## Provenance and licensing

- Positive sequences originate from public structure/classification resources (CATH, SCOP) and UniProt. Please respect the original database licenses/citation policies when redistributing.
- Negative pools are model-generated and carry no additional licensing restrictions beyond this repository’s Apache-2.0 license.
