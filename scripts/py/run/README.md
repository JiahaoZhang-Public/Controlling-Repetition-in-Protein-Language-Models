# Output Directory Layout

Each steering run stores its artifacts under a dedicated output directory. The
files are organized as follows:

```
├── config.yaml                     # Copy of the exact runtime configuration
├── dataset/
│   ├── balanced.fasta              # Combined positive/negative sequences used for sampling
│   ├── dataset_manifest.json       # Metadata about dataset sources and preprocessing
│   ├── neg.fasta                   # Negative class sequences
│   └── pos.fasta                   # Positive class sequences
├── dataset_split.json              # Train/validation/test split indices
├── experiment.log                  # Verbose log from the steering script
├── prefix.sources.fasta            # Input prefixes actually fed into the model
├── prefix.steer.fasta              # Generated sequences for prefix-conditioned steering
├── prefix.steer.metrics.csv        # Metrics computed for prefix-conditioned generations
├── prefix.steer.summary.json       # Aggregated statistics for the prefix-conditioned run
├── steer_result.json               # High-level outcome (success flags, config hashes, etc.)
├── steer_summary.json              # Consolidated summary across steering modes
├── uncond.steer.fasta              # Generated sequences for unconditional steering
├── uncond.steer.metrics.csv        # Metrics for unconditional generations
└── uncond.steer.summary.json       # Aggregated stats for unconditional runs
```