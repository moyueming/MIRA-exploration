#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

SCHEMAS="${SCHEMAS:-flights cyber}"
DATASETS="${DATASETS:-1 2 3 4}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1000000}"

for schema in ${SCHEMAS}; do
  for dataset in ${DATASETS}; do
    echo "=== MIRA schema=${schema} dataset=${dataset} seed=${SEED} workers=${WORKERS} steps=${STEPS} avp=0 ==="
    python MIRA/run.py \
      --schema "${schema}" \
      --dataset_number "${dataset}" \
      --workers "${WORKERS}" \
      --seed "${SEED}" \
      --steps "${STEPS}" \
      --avp 0
  done
done
