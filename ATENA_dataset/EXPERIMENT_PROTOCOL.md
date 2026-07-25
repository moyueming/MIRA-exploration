# ATENA Official Benchmark Protocol

## Compared Methods

- Official ATENA, evaluated from the released repository checkpoint.
- `random`, `greedy`, `dora`, and `pure_a3c` from `atena_baselines/`.
- Formal `MIRA` from the standalone `MIRA/` package.

All methods use the same official ATENA schemas, eight datasets, 12-action
session length, seed convention, 1,000,000 training-step budget for learning
methods, and final checkpoint policy. Intermediate evaluator scores are not
used for checkpoint selection.

## Formal MIRA Contract

MIRA keeps its online encoder, target encoder, dynamics model, directional
latent reward, fixed task reward, progress-only consolidation schedule, and
SWA final policy. AVP is not part of this release and is fixed to `0`. Every
formal run records the disabled state in `avp_manifest.json`.

## Evaluation

Each deterministic method contributes one final 12-action session for each of
the eight datasets. Random contributes 16 independently generated sessions per
dataset. Metrics are computed by the released evaluator:

- Precision: arithmetic mean of the eight per-dataset values.
- T-BLEU-1/2/3: official corpus calculation over the eight sessions.
- EDA-Sim: arithmetic mean of the eight per-dataset values.

The canonical recalculation command is:

```bash
python scripts/recalculate_baselines.py --results_dir results --seed 0
```

## Result Layout

```text
results/MIRA/{schema}{dataset}/seed{seed}/
results/{baseline}/{schema}{dataset}/seed{seed}/
results/official_atena_eval/{schema}/dataset{dataset}/seed{seed}/
```

Generated artifacts under `results/` are intentionally excluded from the
repository.
