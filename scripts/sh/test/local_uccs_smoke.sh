#!/usr/bin/env bash

# Minimal UCCS (Utility-Controlled Contrastive Steering) smoke tests across all shipped models.
# Each model runs a single, tiny main_experiment invocation with the contrastive_layer method.
#
# Usage:
#   DEVICE=cuda SKIP_DPLM=1 bash scripts/sh/test/local_uccs_smoke.sh
# Environment variables:
#   DEVICE     Override runtime.device (default: cpu)
#   SKIP_DPLM  If set to 1, skip the DPLM run (useful when weights are unavailable).

set -euo pipefail

DEVICE="${DEVICE:-cpu}"
SKIP_DPLM="${SKIP_DPLM:-0}"

MODELS=(
  progen2_base
  progen2_small
  protgpt2
  esm2
  esm3
  dplm
)

EXTRA_OVERRIDES=(
  "runtime.device=${DEVICE}"
  "methods=contrastive_layer"
  "steering.layer=0"
  "generation.uncond.n=1"
  "generation.prefix.n=1"
  "generation.uncond.length_min=8"
  "generation.uncond.length_max=12"
  "generation.prefix.prefix_frac=0.2"
  "split.train=2"
  "split.test=2"
  "dataset.opt.target_per_side=20"
)

run_one() {
  local model="$1"
  if [[ "${model}" == "dplm" && "${SKIP_DPLM}" == "1" ]]; then
    echo "[SKIP] DPLM skipped by SKIP_DPLM=1"
    return
  fi

  local exp_id="smoke_uccs_${model}"
  local cmd=(python scripts/py/run/main_experiment.py "exp.id=${exp_id}" "models=${model}")
  for ov in "${EXTRA_OVERRIDES[@]}"; do
    cmd+=("${ov}")
  done

  echo "=== [RUN] ${exp_id} on ${DEVICE} ==="
  echo "Command: ${cmd[*]}"
  "${cmd[@]}"

  local run_dir
  run_dir=$(ls -td "outputs/${exp_id}"/run_* 2>/dev/null | head -n 1 || true)
  if [[ -z "${run_dir}" ]]; then
    echo "[WARN] No run dir for ${exp_id}, skipping evaluation"
    return
  fi

  for fasta in "${run_dir}/uncond.steer.fasta" "${run_dir}/prefix.steer.fasta"; do
    [[ -f "${fasta}" ]] && python scripts/py/run/evaluate_sequences.py "${fasta}" --skip-structure
  done
}

for model in "${MODELS[@]}"; do
  run_one "${model}"
done

echo "[INFO] UCCS smoke tests completed."
