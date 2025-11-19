#!/usr/bin/env bash

# Local smoke test for the decoding+steering ablation sweep.
# Mirrors scripts/sh/test/local_progen2_smoke.sh but targets the new
# sweep defined in scripts/py/run/sweeps/sweep_ablation_decoding.py.

set -euo pipefail

SUBSET_INDICES=(
  0   # ESM3 temperature, cath, seed 0
  7   # ESM3 temperature, uniref50, seed 2
  22  # ESM3 top_p, uniref50, seed 2
  35  # ESM3 entropy, uniref50, seed 0
  45  # ProGen2 temperature, cath, seed 0
  60  # ProGen2 top_p, cath, seed 0
  75  # ProGen2 repetition penalty, cath, seed 0
  90  # ProGen2 no-repeat ngram, cath, seed 0
)

EXTRA_OVERRIDES=(
  "generation.uncond.n=1"
  "generation.prefix.n=1"
  "split.train=5"
  "split.test=5"
  "generation.uncond.length_min=10"
  "generation.uncond.length_max=20"
  "generation.prefix.prefix_frac=0.2"
)

echo "[INFO] Running ablation smoke test for ${#SUBSET_INDICES[@]} configs"

for idx in "${SUBSET_INDICES[@]}"; do
    SPEC=$(python scripts/py/run/sweeps/sweep_ablation_decoding.py --index "${idx}")
    EXP_ID=${SPEC%%::*}
    OVERRIDE_STR=${SPEC#*::}
    IFS=',' read -r -a RAW_OVERRIDES <<<"${OVERRIDE_STR}"
    OVERRIDES=()
    for ov in "${RAW_OVERRIDES[@]}"; do
        [[ -n "${ov}" ]] && OVERRIDES+=("${ov}")
    done

    CLEAN_OVERRIDES=()
    for ov in "${OVERRIDES[@]}"; do
        if [[ "${ov}" == runtime.device=* ]]; then
            continue
        fi
        CLEAN_OVERRIDES+=("${ov}")
    done
    CLEAN_OVERRIDES+=("runtime.device=cpu")

    RUN_ID="${EXP_ID}_smoke"
    CMD=(python scripts/py/run/main_experiment.py "exp.id=${RUN_ID}")
    for ov in "${CLEAN_OVERRIDES[@]}"; do
        CMD+=("${ov}")
    done
    for extra in "${EXTRA_OVERRIDES[@]}"; do
        CMD+=("${extra}")
    done

    echo "=== [RUN] ${RUN_ID} ==="
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"

    RUN_DIR=$(ls -td "outputs/${RUN_ID}"/run_* 2>/dev/null | head -n 1 || true)
    if [[ -z "${RUN_DIR}" ]]; then
        echo "[WARN] No run dir for ${RUN_ID}, skipping evaluation"
        continue
    fi

    for fasta in "${RUN_DIR}/uncond.steer.fasta" "${RUN_DIR}/prefix.steer.fasta"; do
        [[ -f "${fasta}" ]] && python scripts/py/run/evaluate_sequences.py "${fasta}"
    done
done

echo "[INFO] Ablation smoke test finished."
