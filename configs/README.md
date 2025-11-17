Configuration layout:

- `dataset/`: Hydra config group for dataset providers. Add a new `<name>.yaml` with `_target_` pointing to a subclass of `DatasetProvider`, then launch with `dataset=<name>`. Providers such as `posneg_provider` accept a `metrics` block that maps dataset-specific column names (e.g., repetition/utility scores) to the canonical keys consumed by the builder.
- `backend/`: Shared backend defaults (e.g., causal HF LMs vs ESM-style MLMs) that model presets can import.
- `models/`: Model presets and generation settings (init/generation defaults, backend options).
- `methods/`: Steering method presets.
- `main.yaml`: Top-level defaults tying the above together for quick experiments.
