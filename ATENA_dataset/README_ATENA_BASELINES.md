# ATENA Baselines

This package runs four comparison methods on the official ATENA A-EDA
benchmark:

- `random`: uniformly samples from legal actions.
- `greedy`: selects the legal action with the highest immediate official
  compound reward and performs no learning.
- `dora`: A3C trained with the fixed DORA target/curiosity reward.
- `pure_a3c`: A3C trained with the standard task reward.

The baseline package does not import the formal `MIRA/` package and contains
no evaluator-reference vocabulary injection.

## Prerequisites

Run from `ATENA_dataset` with Python 3.8-3.10:

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

The official benchmark must exist at `ATENA-A-EDA/benchmark/`.

## Single Run

```bash
python run_atena_baselines.py \
  --method pure_a3c \
  --schema cyber \
  --dataset_number 1 \
  --workers 28 \
  --seed 0 \
  --steps 1000000
```

## Batch Run

```bash
METHODS="random greedy dora pure_a3c" \
SCHEMAS="cyber flights" DATASETS="1 2 3 4" \
WORKERS=28 SEED=0 STEPS=1000000 bash scripts/run_baselines.sh
```

Results are written to:

```text
results/{method}/{schema}{dataset}/seed{seed}/
```

Learning methods write `train_log.csv`, `eval_metrics.csv`,
`final_metrics.json`, `policy.weights.h5`, and final action JSON files.
`random` and `greedy` do not write policy checkpoints.

## Official Metric Recalculation

After all eight datasets are available for the official method and four
baselines, run:

```bash
python scripts/recalculate_baselines.py --results_dir results --seed 0
```

This validates every reconstructed session against its saved scalar metrics,
then computes T-BLEU as an official corpus metric over all eight datasets.
Precision and EDA-Sim are arithmetic means over the eight datasets.

## Verification

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
