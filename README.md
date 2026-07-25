# MIRA Exploration

This repository contains reinforcement-learning code for target-set exploration
on SDSS galaxy data and the Covertype tabular benchmark. The current method is
a BILE-guided dual-actor A3C explorer: the policy is conditioned on a latent
direction `z`, and an online BILE encoder `phi` is trained from transition
pairs and dynamics prediction error.

## Repository Layout

```text
app/                         Galaxy data pipeline and set operators
rl/A3C_2_actors/             Main Galaxy RL implementation
rl/targets/                  Fixed Galaxy target sets
baselines/                   Galaxy baselines
covertype-exploration/       Covertype benchmark and baselines
```

Large generated artifacts are intentionally not tracked:

```text
app/data/sdss/galaxies.csv
app/data/sdss/galaxies.tar.gz
covertype-exploration/covertype.csv
covertype-exploration/preprocessed/
outputs/
saved_models/
wandb/
```

## Environment

The code was developed with Python 3.8/3.9 style dependencies and TensorFlow
2.9.

```bash
python -m pip install -r requirements.txt
```

For Covertype-only experiments:

```bash
cd covertype-exploration
python -m pip install -r requirements.txt
```

## Data

### Galaxy

The SDSS galaxy CSV is not tracked directly because it is too large for normal
GitHub usage. Instead, the repository tracks the split archive parts:

```text
app/data/sdss/galaxies.tar.gz.aa
app/data/sdss/galaxies.tar.gz.ab
app/data/sdss/galaxies.tar.gz.ac
app/data/sdss/galaxies.tar.gz.ad
app/data/sdss/galaxies.tar.gz.ae
app/data/sdss/galaxies.tar.gz.af
```

Verify the tracked parts before reconstruction. The expected SHA-256 values are:

```text
galaxies.tar.gz.aa  b8733cb51e54108687888be55eb700eb1402790958492cf351d5a2747813a689
galaxies.tar.gz.ab  b55600e3aa09d6ef1154a096d97111a96010663063cd349da4d29606ad78a9f1
galaxies.tar.gz.ac  fa381d4240a99b6f92c6aec7da5b93129365f6af9e29e59b9fc90ea8fb4c6a8b
galaxies.tar.gz.ad  fb7baa46fe8943685cb3c859ed78c4aa580eac7e25941a53067ab90e58a3df41
galaxies.tar.gz.ae  c92bad1646e3ed8610af781cee83271b2282551615c4a2cc89623fa8d857083e
galaxies.tar.gz.af  28b409606489aa1c85f4fc6250701323b6697ad91930be16c7b98c9017d8f012
```

On Linux/macOS, run `sha256sum galaxies.tar.gz.*`. On Windows PowerShell, run
`Get-FileHash galaxies.tar.gz.* -Algorithm SHA256`. Reconstruct and extract the
dataset only after all six values match.

Reconstruct `app/data/sdss/galaxies.csv` before running Galaxy experiments.

On Linux/macOS:

```bash
cd app/data/sdss
cat galaxies.tar.gz.aa galaxies.tar.gz.ab galaxies.tar.gz.ac galaxies.tar.gz.ad galaxies.tar.gz.ae galaxies.tar.gz.af > galaxies.tar.gz
tar -xzf galaxies.tar.gz
```

On Windows PowerShell:

```powershell
cd app\data\sdss
cmd /c copy /b galaxies.tar.gz.aa+galaxies.tar.gz.ab+galaxies.tar.gz.ac+galaxies.tar.gz.ad+galaxies.tar.gz.ae+galaxies.tar.gz.af galaxies.tar.gz
tar -xzf galaxies.tar.gz
```

The repository includes the fixed target sets used by the experiments:

```text
The reconstructed files must match:

```text
galaxies.tar.gz                         ee8fdbf5b5a96a8536889b61d4b6d8df9ad85619e8141e1529c3c3fa4a7613e8
galaxies.csv                            4a8af7c032745039e43fd7dd3abf2720da84a021e3f81c6e67f2e0d970f2e072
galaxies_index/groups.csv               19e1248d467009b59903578a4bd5e947b36b84cc037033365b907b4601e87fe7
galaxies_index/correspondences.csv       e2187604101749becc8a74c4a1078ea0170f52acff3f8adb7b483b561f7d1454
```

The archive extracts `galaxies.csv`; the two tracked index CSVs remain under
`app/data/sdss/galaxies_index/`.

rl/targets/fixed_seed_1.json
rl/targets/fixed_seed_2.json
rl/targets/fixed_seed_3.json
```

### Covertype

From `covertype-exploration/`, download and prepare the official UCI dataset:

```bash
python scripts/download_covertype.py
```

The script fetches `https://archive.ics.uci.edu/static/public/31/covertype.zip`,
adds the canonical header to `covtype.data`, and verifies the result before
writing `covertype.csv`.

```text
covertype-exploration/covertype.csv
SHA-256: a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4
```

Fixed Covertype targets are stored under:

```text
covertype-exploration/targets/
```

The fixed-set universe is generated locally under
`covertype-exploration/preprocessed/` and is not tracked.

## Galaxy Commands

Main BILE method with extrinsic reward:

```bash
python RL-launcher.py \
  --mode scattered \
  --target_set fixed_seed_1 \
  --workers 12 \
  --name galaxy_bile_fixed_seed1_final
```

Run seed 2 and seed 3:

```bash
python RL-launcher.py --mode scattered --target_set fixed_seed_2 --workers 12 --name galaxy_bile_fixed_seed2_final
python RL-launcher.py --mode scattered --target_set fixed_seed_3 --workers 12 --name galaxy_bile_fixed_seed3_final
```

Pure A3C backbone baseline:

```bash
python RL-launcher-pure-a3c.py --mode scattered --target_set fixed_seed_1 --workers 12 --name galaxy_pure_a3c_fixed_seed1_final
python RL-launcher-pure-a3c.py --mode scattered --target_set fixed_seed_2 --workers 12 --name galaxy_pure_a3c_fixed_seed2_final
python RL-launcher-pure-a3c.py --mode scattered --target_set fixed_seed_3 --workers 12 --name galaxy_pure_a3c_fixed_seed3_final
```

Policy-reward no-extrinsic ablation:

```bash
python RL-launcher.py --mode scattered --target_set fixed_seed_1 --workers 12 --w_ext 0 --name galaxy_bile_no_ext_fixed_seed1_final
python RL-launcher.py --mode scattered --target_set fixed_seed_2 --workers 12 --w_ext 0 --name galaxy_bile_no_ext_fixed_seed2_final
python RL-launcher.py --mode scattered --target_set fixed_seed_3 --workers 12 --w_ext 0 --name galaxy_bile_no_ext_fixed_seed3_final
```

Target-blind, non-learning Greedy EDA baseline:

```bash
python RL-launcher-greedy-eda.py --mode scattered --target_set fixed_seed_1 --seed 1 --workers 12 --episodes 1000 --steps 250 --output_prefix outputs/GreedyEDA/greedy_eda_seed1
python RL-launcher-greedy-eda.py --mode scattered --target_set fixed_seed_2 --seed 2 --workers 12 --episodes 1000 --steps 250 --output_prefix outputs/GreedyEDA/greedy_eda_seed2
python RL-launcher-greedy-eda.py --mode scattered --target_set fixed_seed_3 --seed 3 --workers 12 --episodes 1000 --steps 250 --output_prefix outputs/GreedyEDA/greedy_eda_seed3
```

Greedy EDA directly ranks the candidate sets produced by the selected legal
operator using normalized interestingness, coherency, and diversity. Operator
families are balanced by current-episode usage. The policy is reset every
episode, never reads target membership, and never uses current or historical
extrinsic reward for selection. Extrinsic reward is recorded for evaluation only.

Outputs are written to:

```text
outputs/<run_name>/
saved_models/<run_name>/
```

The main reward curve file is:

```text
outputs/<run_name>/<run_name>_fusion_rewards.csv
```

## Covertype Commands

From `covertype-exploration/`, the full Ray/TensorFlow runner is:

```bash
python run_full_a3c.py --baseline pure_a3c --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix pure_a3c_seed1_full
python run_full_a3c.py --baseline paper_a3c --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix paper_a3c_seed1_full
python run_full_a3c.py --baseline atena --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix atena_seed1_full
python run_full_a3c.py --baseline atena_extrinsic --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix atena_ext_seed1_full
python run_full_a3c.py --baseline mira --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix mira_seed1_full
python run_full_a3c.py --baseline mira_no_ext --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix mira_no_ext_seed1_full
```

If preprocessing output is missing, the runner creates it automatically. To
preprocess explicitly:

```bash
python preprocess_fixed_sets.py --target_set fixed_seed_1 --seed 1 --n_sets 50000
```

Covertype outputs are written to:

```text
covertype-exploration/outputs/<output_prefix>/
```

## BILE Summary

For one transition `(s, a, s')`, the BILE directional bonus is:

```text
r_bile = cosine(phi(s') - phi(s), z)
```

The online metric-learning target combines reward difference, dynamics
prediction error, and a target-encoder next-state distance:

```text
target_metric =
    |r_i - r_j|
  + beta_pe * 0.5 * (PE_i + PE_j)
  + gamma * ||phi_target(s'_i) - phi_target(s'_j)||
```

The Galaxy method keeps the same BILE core and adds task-level phase control
for bootstrap/escape exploration and successful-trajectory reuse.

## Notes for Reproduction

- Use fixed target sets (`fixed_seed_1`, `fixed_seed_2`, `fixed_seed_3`) for
  paper-style comparisons.
- Do not commit generated `outputs/`, `saved_models/`, `preprocessed/`, or raw
  dataset CSV files.
- Reward CSVs are sufficient for reward-curve plots; exploration trace/state
  CSVs are used for PCA and set-coverage visualizations.
