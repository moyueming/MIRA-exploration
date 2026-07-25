#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../ATENA-A-EDA/atena-basic"

STEPS="${STEPS:-1000000}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-16}"
OUTROOT="../../results/official_atena"

for schema in FLIGHTS NETWORKING; do
  for dataset in 1 2 3 4; do
    official_dataset_index=$((dataset - 1))
    label="$(echo "${schema}" | tr '[:upper:]' '[:lower:]')${dataset}_seed${SEED}"
    echo "=== official_atena schema=${schema} dataset=${dataset} seed=${SEED} steps=${STEPS} envs=${NUM_ENVS} ==="
    python train.py \
      --env ATENAcont-v0 \
      --schema "${schema}" \
      --dataset-number "${official_dataset_index}" \
      --seed "${SEED}" \
      --algo chainerrl_ppo \
      --arch FFParamSoftmax \
      --episode-length 12 \
      --steps "${STEPS}" \
      --eval-interval 10000 \
      --stack-obs-num 3 \
      --num-envs "${NUM_ENVS}" \
      --outdir "${OUTROOT}/${label}"
  done
done
