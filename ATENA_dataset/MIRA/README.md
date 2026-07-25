# MIRA Standalone

This directory contains the standalone `MIRA` implementation. It
depends on the sibling `ATENA-A-EDA/benchmark` checkout and datasets, but does
not import or modify the current `atena_baselines` package.

## AVP

AVP is an optional action-vocabulary module and is disabled by default. Only
the exact value `--avp 1` enables it. `--avp 0` and every other value disable
it. Removing `mira/avp.py` also disables AVP without preventing MIRA from
running.

When enabled, AVP extracts filter terms from the target dataset's official
evaluator session and appends them after each column's ten data-frequency
terms. This reproduces the action vocabulary used by the earlier enabled MIRA.
Report AVP-enabled results as an AVP variant or upper-bound ablation.

Every run writes `avp_manifest.json` with the requested value, module
availability, effective state, source path, extracted terms, and digest. With
AVP disabled, no AVP term enters the action vocabulary.

## Frozen Method

- Bounded data-frequency action vocabulary with optional AVP augmentation.
- Fixed task reward combining official KL, compaction, display diversity,
  column coverage, group coverage, and action-structure rewards.
- Full MIRA online encoder, target encoder, and dynamics model.
- Synchronous ordered rollout batches with the original worker seed formula.
- Consolidation schedule for learning rates, MIRA alpha, entropy, and auxiliary
  reward weights, with SWA beginning at 40% progress.
- Last 1M-step SWA policy as the formal result.
- Deterministic 12-step masked-argmax formal evaluation.

The persistent spawn pool changes process lifetime only. Each rollout still
creates a fresh environment, policy, and full MIRA inference module from the
same per-update weight snapshot.

## Prerequisites

Start in `ATENA_dataset`. These paths must exist:

```text
ATENA-A-EDA/benchmark/
MIRA/
```

Use Python 3.8-3.10 and TensorFlow 2.9:

```bash
python -m pip install -r MIRA/requirements.txt
```

## Smoke Test

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

## Single Run

Default AVP-off run:

```bash
python MIRA/run.py \
  --schema cyber \
  --dataset_number 1 \
  --workers 28 \
  --seed 0 \
  --steps 1000000 \
  --avp 0
```

AVP-enabled run matching the earlier enabled MIRA vocabulary:

```bash
python MIRA/run.py \
  --schema cyber \
  --dataset_number 1 \
  --workers 28 \
  --seed 0 \
  --steps 1000000 \
  --avp 1
```

## Batch Run

```bash
SCHEMAS="cyber flights" DATASETS="1 2 3 4" \
WORKERS=28 SEED=0 STEPS=1000000 AVP=1 bash MIRA/scripts/run_all.sh
```

Set `AVP=0` or omit `AVP` for the default disabled mode.

## Outputs

The default result directory is:

```text
results/MIRA/{schema}{dataset}/seed{seed}/
```

Formal files:

```text
config.json
avp_manifest.json
train_log.csv
eval_metrics.csv
final_metrics.json
policy.weights.h5
actions_steps{steps}.json
```

Online diagnostics use `eval_metrics_online.csv`,
`final_metrics_online.json`, `policy.online.weights.h5`, and
`actions_online_steps{steps}.json`. Formal reporting must use only the final
SWA files, never an online or intermediate row selected by evaluator score.

## Server Synchronization

Copy the complete directory:

```text
MIRA/
```

To deploy a permanently disabled build, remove `mira/avp.py`. Keeping the
file and using `AVP=0` has the same action-vocabulary effect.

Do not modify files while a MIRA process is running. Restart from a clean output
directory after synchronization.

## Verification

```bash
cd MIRA
python -m unittest discover -s tests -p "test_*.py" -v
```

The TensorFlow model and 24-step smoke tests must run without skips on the
server.
