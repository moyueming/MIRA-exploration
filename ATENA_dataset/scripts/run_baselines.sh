#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-16}"
STEPS="${STEPS:-1000000}"
SEED="${SEED:-0}"
METHODS="${METHODS:-random greedy dora pure_a3c}"
SCHEMAS="${SCHEMAS:-flights cyber}"
DATASETS="${DATASETS:-1 2 3 4}"

for method in ${METHODS}; do
  for schema in ${SCHEMAS}; do
    for dataset in ${DATASETS}; do
      echo "=== method=${method} schema=${schema} dataset=${dataset} seed=${SEED} workers=${WORKERS} steps=${STEPS} ==="
      python run_atena_baselines.py \
        --method "${method}" \
        --schema "${schema}" \
        --dataset_number "${dataset}" \
        --seed "${SEED}" \
        --workers "${WORKERS}" \
        --steps "${STEPS}" \
        --episode_length 12
    done
  done
done

python scripts/summarize_results.py \
  --root results \
  --output results/atena_baselines_summary.csv \
  --row-source last
