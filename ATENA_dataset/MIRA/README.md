# MIRA on the Official ATENA Benchmark

This directory contains the standalone MIRA implementation used for the
official ATENA A-EDA task. It depends on the sibling
`ATENA-A-EDA/benchmark` submodule and does not import the comparison methods in
`atena_baselines/`.

## Public Release Contract

- AVP is not included and is fixed to `0`.
- The action vocabulary is built only from bounded data-frequency terms.
- The online encoder, target encoder, dynamics model, directional latent
  reward, consolidation schedule, and SWA final policy remain enabled.
- Formal evaluation uses the final SWA policy and deterministic 12-step
  masked-argmax sessions.

Every run writes `avp_manifest.json` with `requested: "0"`, `available: false`,
and `active: false`. Passing any value other than `--avp 0` is rejected.

## Environment

Run from `ATENA_dataset` with Python 3.8-3.10:

```bash
python -m pip install -r MIRA/requirements.txt
```

## Smoke Run

```bash
python MIRA/run.py \
  --schema cyber \
  --dataset_number 1 \
  --workers 2 \
  --seed 0 \
  --steps 24 \
  --eval_interval 1 \
  --avp 0 \
  --output_dir results/_mira_smoke
```

## Formal Run

```bash
python MIRA/run.py \
  --schema cyber \
  --dataset_number 1 \
  --workers 28 \
  --seed 0 \
  --steps 1000000 \
  --avp 0
```

Run all eight official datasets:

```bash
SCHEMAS="cyber flights" DATASETS="1 2 3 4" \
WORKERS=28 SEED=0 STEPS=1000000 bash MIRA/scripts/run_all.sh
```

Generated files are written below `results/MIRA/` and are intentionally
ignored by Git.
