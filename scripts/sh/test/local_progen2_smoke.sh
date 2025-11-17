#!/usr/bin/env bash

# Lightweight sanity check for the progen2_base sweep.
# Runs a reduced subset of parameter combinations with tiny generation counts
# so the full sweep can be validated locally before launching Slurm jobs.

set -euo pipefail

SUBSET_INDICES=(
  0   # control_default
  1   # temperature_0.70
  7   # top_p_0.80
  13  # no_repeat_ngram_2
  16  # repetition_penalty_1.10
  19  # neuron_deactivation_8
  24  # uccs_layer00
  51  # probe_layer00
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

echo "[INFO] Running local smoke test for ${#SUBSET_INDICES[@]} configs"

for idx in "${SUBSET_INDICES[@]}"; do
    SPEC=$(python scripts/py/run/sweeps/sweep_progen2_base.py --index "${idx}")
    IFS='::' read -r EXP_ID OVERRIDE_STR <<<"${SPEC}"
    IFS=',' read -r -a RAW_OVERRIDES <<<"${OVERRIDE_STR}"
    OVERRIDES=()
    for ov in "${RAW_OVERRIDES[@]}"; do
        if [[ -n "${ov}" ]]; then
            CLEAN="${ov##*:}"
            OVERRIDES+=("${CLEAN}")
        fi
    done

    RUN_ID="${EXP_ID}_smoke"
    CMD=(python scripts/py/run/main_experiment.py "exp.id=${RUN_ID}")
    for ov in "${OVERRIDES[@]}"; do
        [[ -n "${ov}" ]] && CMD+=("${ov}")
    done
    for extra in "${EXTRA_OVERRIDES[@]}"; do
        CMD+=("${extra}")
    done

    echo "=== [RUN] ${RUN_ID} ==="
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"

    RUN_DIR=$(ls -td "outputs/${RUN_ID}"/z* 2>/dev/null | head -n 1 || true)
    if [[ -z "${RUN_DIR}" ]]; then
        echo "[WARN] No run dir for ${RUN_ID}, skipping evaluation"
        continue
    fi

    for fasta in "${RUN_DIR}/uncond.steer.fasta" "${RUN_DIR}/prefix.steer.fasta"; do
        [[ -f "${fasta}" ]] && python scripts/py/run/evaluate_sequences.py "${fasta}"
    done
done

echo "[INFO] Local smoke test finished."
