#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-0}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1000000_finish}"
TRAIN_ROOT="${TRAIN_ROOT:-results/official_atena}"
OUTDIR="${OUTDIR:-results/official_atena_eval}"
RUN_DIR="${RUN_DIR:-latest}"

for schema in flights cyber; do
  official_schema="flights"
  if [[ "${schema}" == "cyber" ]]; then
    official_schema="networking"
  fi
  for dataset in 1 2 3 4; do
    label="${official_schema}${dataset}_seed${SEED}"
    run_root="${TRAIN_ROOT}/${label}"
    if [[ "${RUN_DIR}" == "latest" ]]; then
      latest_run="$(find "${run_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
      if [[ -z "${latest_run}" ]]; then
        echo "No official ATENA run directories found under ${run_root}" >&2
        exit 1
      fi
      load_path="${latest_run}/${CHECKPOINT_STEP}"
    else
      load_path="${run_root}/${RUN_DIR}/${CHECKPOINT_STEP}"
    fi
    echo "=== evaluate official_atena schema=${schema} dataset=${dataset} seed=${SEED} load=${load_path} ==="
    python scripts/evaluate_official_atena.py \
      --schema "${schema}" \
      --dataset_number "${dataset}" \
      --seed "${SEED}" \
      --load "${load_path}" \
      --episode_length 12 \
      --output_dir "${OUTDIR}"
  done
done

python scripts/summarize_results.py \
  --root "${OUTDIR}" \
  --output results/official_atena_summary.csv
