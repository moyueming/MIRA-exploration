# Covertype Exploration RL

This folder contains a Covertype generalization benchmark for the galaxy
exploration project. The task now follows the same fixed-set-universe principle
as the galaxy dataset:

1. Offline preprocessing builds a fixed universe of Covertype sets.
2. Each set has a stable `set_id`, state vector, size, target hits, and graph
   transitions.
3. During training, methods can only move on this fixed set graph.
4. The environment never creates new sets at runtime.
5. Each run writes reward curves and exploration traces.

## Fixed Set Universe

The default universe size is:

```text
50,000 fixed sets
```

The preprocessing output is stored under:

```text
preprocessed/fixed_sets_seed<seed>_n<n_sets>_min<min_set_size>/
```

The generated files are:

- `metadata.json`
- `constraints.npy`
- `set_states.npy`
- `set_sizes.npy`
- `set_graph.npy`
- `target_offsets.npy`
- `target_items.npy`

The fixed graph uses the same operator style as the galaxy task:

- facet / drill-down
- superset / roll-up
- neighbor movement
- distribution movement

Invalid graph actions are masked during action selection.

## Baselines

- `paper_a3c`: actor-critic policy trained with the same reward structure used
  by the galaxy paper baseline:
  `(1 - counter_curiosity_ratio) * extrinsic_reward + counter_curiosity_ratio * counter_curiosity`.
  The reward CSV keeps a `familiarity` column for compatibility; in this
  strict Galaxy-aligned baseline it is the same value as `extrinsic_reward`.
- `atena`: actor-critic policy trained with the same ATENA-style compound
  reward used by the galaxy baseline:
  `w_int * interestingness + w_coh * coherency + w_div * diversity`.
- `atena_extrinsic`: actor-critic policy trained with the same target-aware
  ATENA reward used by the galaxy baseline:
  `w_ext * extrinsic + w_int * interestingness + w_coh * coherency + w_div * diversity`.
- `mira`: dual-actor A3C with online BILE metric learning. The policy is
  conditioned on an episode-level latent direction `z`, the BILE encoder `phi`
  and dynamics model are updated online from a central BILE transition buffer,
  and the policy reward is `w_ext * extrinsic + alpha * bile_bonus`. The first
  `100` episodes use local rollout pairs as a fixed warm-up; after that, each
  `phi` update samples a replay mini-batch and randomly permutes it to form
  `(s_i, s_j)` state pairs. By default, `z`
  is sampled from a mixture of random directions and directions derived from
  prior successful transitions. The pool stores both local positive
  `(s, s_next)` steps and trajectory directions from the segment
  anchor to later positive hits, then recomputes `normalize(phi(s_next)-phi(s))`
  with the current global encoder when sampling. The pool keeps a mix of elite
  and ordinary/recent successful steps, decays the success-sampling probability
  toward a random-exploration floor, and can blend a pool direction with a fresh
  random direction. This prevents one early successful pocket from locking the
  policy into a single path.
- `mira_no_ext`: exploration-only MIRA ablation. It uses the same
  z-conditioned policy, encoder, and dynamics model, but excludes extrinsic
  reward from both the policy reward and the BILE metric reward-difference
  term. It uses the same BILE replay-buffer pair sampling and fixed warm-up as
  the main method. Extrinsic reward is still logged as the evaluation metric.
  This ablation keeps `z` random and does not use the successful-transition
  pool.
- `greedy_eda`: target-blind, non-learning traditional EDA baseline. It ranks
  the same target-independent candidate slots exposed to learned methods using
  normalized interestingness, coherency, and within-episode diversity. Each
  worker keeps target-independent set visit counts across its episodes and
  samples from the three highest-scoring least-visited candidates with fixed
  linear rank weights. Target membership and extrinsic reward are never used
  for action selection; extrinsic reward is recorded for evaluation only.
  Official runs read the existing 100,000-set preprocessing artifacts and
  never create or modify preprocessing data.

Run the three fixed-target seeds from `covertype-exploration/`:

```bash
python baselines/greedy_eda/run.py --target_set fixed_seed_1 --seed 1 --preprocess_name by_distribution_path100k_seed1 --workers 12 --episodes 1000 --steps 250 --candidate_slots 10 --selection_top_k 3 --output_dir outputs/GreedyEDA_count_balanced --output_prefix greedy_eda_seed1_full
python baselines/greedy_eda/run.py --target_set fixed_seed_2 --seed 2 --preprocess_name by_distribution_path100k_seed2 --workers 12 --episodes 1000 --steps 250 --candidate_slots 10 --selection_top_k 3 --output_dir outputs/GreedyEDA_count_balanced --output_prefix greedy_eda_seed2_full
python baselines/greedy_eda/run.py --target_set fixed_seed_3 --seed 3 --preprocess_name by_distribution_path100k_seed3 --workers 12 --episodes 1000 --steps 250 --candidate_slots 10 --selection_top_k 3 --output_dir outputs/GreedyEDA_count_balanced --output_prefix greedy_eda_seed3_full
```

`--workers` defaults to 12 and can be increased for a faster run. Keep the same
worker count for all three seeds because worker-local memory makes it part of
the reproducible experiment configuration.

The runner writes reward, exploration-trace, visited-state, and resolved-config
artifacts using the same columns and 1,000-by-250 interaction budget as the
other target-discovery baselines.

## Outputs

Each run writes files into a dedicated `outputs/<output_prefix>/` folder:

- `<prefix>_<baseline>_rewards.csv`
- `<prefix>_<baseline>_exploration_trace.csv`
- `<prefix>_<baseline>_visited_set_states.csv`
- `<prefix>_<baseline>_config.json`

The reward CSV is enough for reward curves, cumulative unique set curves, and
target-efficiency curves. The trace/state CSVs are used for PCA exploration
visualizations.

## Target Set Design

Covertype targets are generated as sparse clustered regions, not as independent
random rows. Each region is anchored by one row and converted into constraints
that the set-operator action space can express:

- `Cover_Type`
- `Wilderness_Area`
- several continuous feature bins
- optionally one `Soil_Type`

Rows nearest to the regional anchor inside those predicate regions become target
items. This mirrors the galaxy task more closely: rewards are sparse, but once a
policy reaches a useful region, local exploration can discover multiple related
target items. Existing target JSON files are loaded as fixed target sets; use a
new target name or replace the JSON file when changing the task definition.

The default target setup is:

```text
1,000 target rows
8 target regions
125 target rows per region
```

The extrinsic reward follows the galaxy-style target discovery logic: repeated
hits of the same target item do not keep producing reward unless the item is
found through a higher-quality set ratio.

## Preprocessing
## Dataset Download

Download and prepare the canonical UCI Covertype dataset from this directory:

```bash
python scripts/download_covertype.py
```

The script downloads the official UCI archive from
`https://archive.ics.uci.edu/static/public/31/covertype.zip`, extracts
`covtype.data`, adds the canonical 55-column header, and writes
`covertype.csv`. It verifies the prepared file before replacing any existing
dataset.

Expected SHA-256:

```text
covertype.csv  a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4
```

`covertype.csv` is intentionally not tracked by Git. Keep the filename and
checksum unchanged for reproducible runs.


Preprocess explicitly:

```bash
python preprocess_fixed_sets.py --target_set fixed_seed_1 --seed 1 --n_sets 50000
```

If preprocessing output is missing, `run_full_a3c.py` creates it
automatically before training. Use `--force_preprocess` to overwrite an existing
fixed universe.

## Example Commands

Default experiment settings are `1000` episodes and `250` steps per episode.
The default target set size is `1000`.

For the strict paper experiments, use the full Ray/TensorFlow dual-actor A3C
runner:

```bash
python run_full_a3c.py --baseline pure_a3c --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix pure_a3c_seed1_full
python run_full_a3c.py --baseline paper_a3c --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix paper_a3c_seed1_full
python run_full_a3c.py --baseline atena --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix atena_seed1_full
python run_full_a3c.py --baseline atena_extrinsic --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix atena_ext_seed1_full
python run_full_a3c.py --baseline mira --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix mira_seed1_full
python run_full_a3c.py --baseline mira_no_ext --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix mira_no_ext_seed1_full
```

Useful `mira` BILE training and direction-pool controls:

- `--bile_pair_warmup_episodes 100`: fixed number of initial episodes that
  train `phi` with local rollout pairs before switching to replay sampling.
- `--bile_replay_buffer_size 50000`: maximum number of BILE transitions kept
  in the central replay buffer.
- `--bile_phi_batch_size 128`: replay mini-batch size used for `phi` and
  dynamics updates after warm-up.
- `--bile_min_replay_size 256`: minimum replay-buffer size before replay
  random-pair training is used.
- `--bile_success_prob 0.6`: probability that an episode samples `z` from the
  successful-transition pool instead of a fresh random direction.
- `--bile_success_prob_min 0.35`: lower bound reached by the success-pool
  probability after the decay horizon, preserving random direction exploration.
- `--bile_success_prob_decay_episodes 600`: episodes over which
  `bile_success_prob` decays toward `bile_success_prob_min`. Set to `0` to keep
  a fixed success-pool probability.
- `--candidate_slots 10`: number of candidate next-set slots exposed to the set
  actor.
- `--bile_success_pool_size 128`: maximum retained successful directions. Set
  to `0` to disable the pool.
- `--bile_success_noise_scale 0.2`: Gaussian perturbation added before
  normalizing a pool-sampled direction.
- `--bile_success_mix_random_prob 0.25`: probability of blending a sampled
  success direction with a fresh random direction.
- `--bile_success_mix_random_weight 0.35`: random-direction weight used during
  that blend.
- `--bile_success_trajectory_score_scale 0.5`: score scale for segment-level
  trajectory directions; local positive transitions are still stored directly.
- `--bile_min_success_reward 0.0`: minimum immediate extrinsic reward needed
  before adding a transition to the pool.
- `--bile_success_score_clip 100.0`: cap stored success scores so rare very high
  reward transitions do not dominate the direction pool too abruptly.
- `--bile_success_elite_fraction 0.5`: fraction of the pool reserved for highest
  scoring transitions; the rest stays mixed across ordinary/recent successes.
- `--bile_success_weight_power 0.5`: sampling weight exponent over clipped
  success scores.
- `--enable_bile_stability_guard`: optional diagnostic guard that restores and
  locks best rolling weights after a sharp late collapse. It is disabled by
  default.

The full runner uses the same structural pattern as the galaxy implementation:

- `SetActor`: LSTM policy over candidate next-set slots.
- `OperationActor`: LSTM policy over concrete set-operator actions.
- `Critic`: LSTM value network over the set-state sequence.
- `BILEModule` for `mira*`: online `phi(obs)` encoder, target encoder,
  and dynamics model `P(obs, set_action, operation_action) -> obs_next`. The
  metric loss uses local rollout pairs only during the fixed warm-up, then uses
  replay-buffer random permutation pairs for the main training stage.
- Ray `ParameterServer`: owns global weights, applies worker gradients, writes
  reward and exploration CSV files.
- Ray workers: run episodes in parallel and push n-step actor-critic gradients.

In Covertype, the fixed graph action is decomposed into two decisions:

1. the set actor chooses one of up to 10 candidate next-set slots;
2. the operation actor chooses a valid graph operator that reaches that selected
   candidate set.

This preserves the fixed-set Covertype task while making the training framework
match the galaxy dual-actor A3C design.
