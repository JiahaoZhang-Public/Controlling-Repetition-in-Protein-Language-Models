#!/usr/bin/env bash

# Local smoke test for the DPLM sweep (configs/models/dplm.yaml, airkingbd/dplm_650m).
# Runs a small subset of sweep configs with tiny generation settings to verify end-to-end plumbing.

set -euo pipefail

SUBSET_INDICES=(
  0   # cath, seed0, control_default
  3   # cath, seed0, temperature_1p3
  7   # cath, seed0, disable_resample
  16  # cath, seed1, sampling_gumbel_argmax
  61  # uniref50, seed0, temperature_0p7
  89  # uniref50, seed2, sampling_argmax
  145 # scop, seed2, temperature_0p7
  176 # scop, seed4, resample_ratio_0p10
)

EXTRA_OVERRIDES=(
  "generation.uncond.n=1"
  "generation.prefix.n=1"
  "split.train=5"
  "split.test=5"
  "generation.uncond.length_min=10"
  "generation.uncond.length_max=20"
  "generation.prefix.prefix_frac=0.2"
  "generation.uncond.overrides.max_iter=10"
  "generation.prefix.overrides.max_iter=10"
)

echo "[INFO] Running DPLM smoke test for ${#SUBSET_INDICES[@]} configs"

for idx in "${SUBSET_INDICES[@]}"; do
    SPEC=$(python scripts/py/run/sweeps/sweep_dplm.py --index "${idx}")
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
    append_plus_if_needed() {
        local ov="$1"
        if [[ "${ov}" == *".overrides."* && "${ov}" != +* ]]; then
            echo "+${ov}"
        else
            echo "${ov}"
        fi
    }
    for ov in "${CLEAN_OVERRIDES[@]}"; do
        CMD+=("$(append_plus_if_needed "${ov}")")
    done
    for extra in "${EXTRA_OVERRIDES[@]}"; do
        CMD+=("$(append_plus_if_needed "${extra}")")
    done

    echo "=== [RUN] ${RUN_ID} ==="
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"

    RUN_DIR=$(ls -td "outputs/${RUN_ID}"/run_* 2>/dev/null | head -n 1 || true)
    if [[ -z "${RUN_DIR}" ]]; then
        RUN_DIR=$(ls -td "outputs/${RUN_ID}"/z* 2>/dev/null | head -n 1 || true)
    fi
    if [[ -z "${RUN_DIR}" ]]; then
        echo "[WARN] No run dir for ${RUN_ID}, skipping evaluation"
        continue
    fi

    for fasta in "${RUN_DIR}/uncond.steer.fasta" "${RUN_DIR}/prefix.steer.fasta"; do
        [[ -f "${fasta}" ]] && python scripts/py/run/evaluate_sequences.py "${fasta}"
    done
done

echo "[INFO] DPLM smoke test finished."
