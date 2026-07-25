import csv
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .avp_loader import avp_manifest
from .env import make_env
from .evaluate import (
    evaluate_policy,
    masked_probs,
    write_json,
    write_metrics_csv,
)
from .rollout_pool import PersistentRolloutExecutor
from .schedule import runtime_args, training_schedule
from .swa import RunningWeightAverage


METHOD = "MIRA"
FORMAL_METRICS_FILE = "eval_metrics.csv"
ONLINE_METRICS_FILE = "eval_metrics_online.csv"
FORMAL_FINAL_FILE = "final_metrics.json"
ONLINE_FINAL_FILE = "final_metrics_online.json"
FORMAL_POLICY_FILE = "policy.weights.h5"
ONLINE_POLICY_FILE = "policy.online.weights.h5"
FORMAL_ACTION_PREFIX = "actions_steps"
ONLINE_ACTION_PREFIX = "actions_online_steps"
TRAIN_LOG_FIELDS = (
    "update",
    "steps",
    "episode_reward",
    "episode_task_reward",
    "policy_lr",
    "mira_lr",
    "alpha",
    "entropy_coef",
    "auxiliary_reward_scale",
    "swa_count",
    "swa_active",
    "policy_loss",
    "value_loss",
    "entropy",
    "phi_loss",
    "dyn_loss",
    "metric_target",
)


def formal_final_row(metrics_rows):
    return dict(metrics_rows[-1]) if metrics_rows else None


def build_worker_args(
    args,
    runtime_args,
    policy_weights,
    mira_weights,
    update_idx,
    rollout_count,
    rng,
    direction_sampler,
):
    payloads = []
    for worker_id in range(int(rollout_count)):
        worker_runtime = SimpleNamespace(**vars(runtime_args))
        worker_seed = int(
            args.seed * 100000 + update_idx * args.workers + worker_id
        )
        direction = direction_sampler(rng, args.mira_latent_dim)
        payloads.append((
            worker_runtime,
            policy_weights,
            mira_weights,
            worker_seed,
            direction,
        ))
    return payloads


def run_experiment(args):
    _validate_runtime(args)
    with PersistentRolloutExecutor(
        workers=args.workers,
        worker_function=rollout_worker,
    ) as executor:
        return _run_experiment(args, executor)


def _validate_runtime(args):
    if getattr(args, "method", None) != METHOD:
        raise ValueError("Standalone package only supports {}".format(METHOD))
    if int(args.episode_length) != 12:
        raise ValueError("Standalone MIRA requires 12-step episodes")


def _run_experiment(args, rollout_executor):
    random.seed(args.seed)
    np.random.seed(args.seed)

    result_dir = (
        Path(args.output_dir)
        / METHOD
        / "{}{}".format(args.schema, int(args.dataset_number))
        / "seed{}".format(int(args.seed))
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "config.json", vars(args))
    write_json(
        result_dir / "avp_manifest.json",
        avp_manifest(args.schema, args.dataset_number, args.avp),
    )

    probe_env = make_env(args.schema, args.dataset_number, args.seed, args)
    state_dim = probe_env.state_dim
    action_dim = probe_env.action_dim

    import tensorflow as tf
    from .models import MiraMetricModule, PolicyValueNet, sample_direction

    tf.random.set_seed(args.seed)
    model = PolicyValueNet(state_dim, action_dim, hidden=args.hidden)
    optimizer = tf.keras.optimizers.Adam(float(args.lr))
    mira = MiraMetricModule(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.mira_latent_dim,
        hidden=args.mira_hidden,
        lr=args.mira_lr,
        tau=args.mira_tau,
    )
    swa = RunningWeightAverage(start_fraction=args.swa_start)
    formal_model = PolicyValueNet(state_dim, action_dim, hidden=args.hidden)
    formal_model.set_weights(model.get_weights())

    metrics_rows = []
    online_metrics_rows = []
    train_log_path = result_dir / "train_log.csv"
    _init_train_log(train_log_path)
    steps_done = 0
    update_idx = 0
    rng = np.random.default_rng(args.seed + 20260627)

    while steps_done < args.steps:
        schedule = training_schedule(args, steps_done)
        current_args = runtime_args(args, schedule)
        _set_optimizer_lr(optimizer, schedule["policy_lr"])
        _set_optimizer_lr(mira.encoder_opt, schedule["mira_lr"])
        _set_optimizer_lr(mira.dynamics_opt, schedule["mira_lr"])

        remaining = int(args.steps) - steps_done
        rollout_count = min(
            int(args.workers),
            max(1, (remaining + args.episode_length - 1) // args.episode_length),
        )
        worker_args = build_worker_args(
            args=args,
            runtime_args=current_args,
            policy_weights=model.get_weights(),
            mira_weights=mira.get_weights(),
            update_idx=update_idx,
            rollout_count=rollout_count,
            rng=rng,
            direction_sampler=sample_direction,
        )
        rollouts = rollout_executor.map(worker_args)
        batch = _merge_rollouts(rollouts)
        steps_done += int(len(batch["rewards"]))
        update_idx += 1

        update_stats = _a3c_update(model, optimizer, batch, current_args)
        progress = min(
            max(float(steps_done) / max(float(args.steps), 1.0), 0.0),
            1.0,
        )
        swa.update(model.get_weights(), progress)
        mira_stats = mira.update(
            batch["states"],
            batch["actions"],
            batch["task_rewards"],
            batch["next_states"],
            gamma=args.gamma,
            beta_pe=args.mira_beta_pe,
            metric_clip=args.mira_metric_clip,
        )

        _append_train_log(train_log_path, {
            "update": update_idx,
            "steps": steps_done,
            "episode_reward": float(np.mean([
                rollout["episode_reward"] for rollout in rollouts
            ])),
            "episode_task_reward": float(np.mean([
                rollout["episode_task_reward"] for rollout in rollouts
            ])),
            "policy_lr": schedule["policy_lr"],
            "mira_lr": schedule["mira_lr"],
            "alpha": schedule["alpha"],
            "entropy_coef": schedule["entropy_coef"],
            "auxiliary_reward_scale": schedule["auxiliary_reward_scale"],
            "swa_count": swa.count,
            "swa_active": swa.active,
            **update_stats,
            **mira_stats,
        })

        if update_idx % max(1, int(args.eval_interval)) == 0 or steps_done >= args.steps:
            formal_model.set_weights(swa.formal_weights(model.get_weights()))
            formal_row = evaluate_policy(
                current_args,
                formal_model,
                result_dir,
                steps_done,
                action_filename_prefix=FORMAL_ACTION_PREFIX,
            )
            online_row = evaluate_policy(
                current_args,
                model,
                result_dir,
                steps_done,
                action_filename_prefix=ONLINE_ACTION_PREFIX,
            )
            metrics_rows.append(formal_row)
            online_metrics_rows.append(online_row)
            write_metrics_csv(result_dir / FORMAL_METRICS_FILE, metrics_rows)
            write_metrics_csv(result_dir / ONLINE_METRICS_FILE, online_metrics_rows)
            formal_model.save_weights(str(result_dir / FORMAL_POLICY_FILE))
            model.save_weights(str(result_dir / ONLINE_POLICY_FILE))

    write_json(result_dir / FORMAL_FINAL_FILE, formal_final_row(metrics_rows))
    write_json(
        result_dir / ONLINE_FINAL_FILE,
        formal_final_row(online_metrics_rows),
    )
    return metrics_rows


def _seed_rollout(worker_seed):
    import tensorflow as tf

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    tf.random.set_seed(worker_seed)


def rollout_worker(payload):
    args, policy_weights, mira_weights, worker_seed, direction = payload
    from .models import MiraMetricModule, PolicyValueNet

    _seed_rollout(worker_seed)
    env = make_env(args.schema, args.dataset_number, worker_seed, args)
    model = PolicyValueNet(env.state_dim, env.action_dim, hidden=args.hidden)
    model.set_weights(policy_weights)
    mira = MiraMetricModule(
        env.state_dim,
        env.action_dim,
        latent_dim=args.mira_latent_dim,
        hidden=args.mira_hidden,
        lr=args.mira_lr,
        tau=args.mira_tau,
    )
    mira.set_weights(mira_weights)
    return _run_rollout(args, direction, env, model, mira)


def _run_rollout(args, direction, env, model, mira):
    state = env.reset()
    done = False
    states = []
    actions = []
    rewards = []
    task_rewards = []
    values = []
    next_states = []
    masks = []
    total_reward = 0.0
    total_task_reward = 0.0

    while not done:
        state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        probs, value = model(state_array, training=False)
        mask = env.legal_action_mask()
        policy = masked_probs(probs.numpy()[0], mask)
        action = int(np.random.choice(env.action_dim, p=policy))
        next_state, task_reward, done, _ = env.step(action)
        mira_bonus = mira.compute_bonus(
            state,
            next_state,
            direction,
            clip_value=args.mira_bonus_clip,
        )
        reward = float(task_reward) + float(args.alpha) * float(mira_bonus)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        task_rewards.append(task_reward)
        values.append(float(value.numpy()[0, 0]))
        next_states.append(next_state)
        masks.append(mask)
        total_reward += reward
        total_task_reward += float(task_reward)
        state = next_state

    values_array = np.asarray(values + [0.0], dtype=np.float32)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    advantages, returns = _gae(
        rewards_array,
        values_array,
        args.gamma,
        args.gae_lambda,
    )
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": rewards_array,
        "task_rewards": np.asarray(task_rewards, dtype=np.float32),
        "returns": returns.astype(np.float32),
        "advantages": advantages.astype(np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "masks": np.asarray(masks, dtype=np.float32),
        "episode_reward": float(total_reward),
        "episode_task_reward": float(total_task_reward),
    }


def _merge_rollouts(rollouts):
    keys = (
        "states",
        "actions",
        "rewards",
        "task_rewards",
        "returns",
        "advantages",
        "next_states",
        "masks",
    )
    return {
        key: np.concatenate([rollout[key] for rollout in rollouts], axis=0)
        for key in keys
    }


def _masked_probs_tf(probs, mask):
    import tensorflow as tf

    mask = tf.convert_to_tensor(mask, dtype=probs.dtype)
    binary_mask = tf.cast(mask > 0, probs.dtype)
    finite_probs = tf.where(
        tf.math.is_finite(probs),
        tf.maximum(probs, 0.0),
        tf.zeros_like(probs),
    )
    masked = finite_probs * binary_mask
    total = tf.reduce_sum(masked, axis=-1, keepdims=True)
    valid_count = tf.reduce_sum(binary_mask, axis=-1, keepdims=True)
    fallback = tf.math.divide_no_nan(binary_mask, valid_count)
    return tf.where(
        tf.logical_and(tf.math.is_finite(total), total > 0.0),
        tf.math.divide_no_nan(masked, total),
        fallback,
    )


def _a3c_update(model, optimizer, batch, args):
    import tensorflow as tf

    states = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
    actions = tf.convert_to_tensor(batch["actions"], dtype=tf.int32)
    returns = tf.convert_to_tensor(
        batch["returns"].reshape(-1, 1),
        dtype=tf.float32,
    )
    advantages = batch["advantages"].astype(np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages_tensor = tf.convert_to_tensor(
        advantages.reshape(-1, 1),
        dtype=tf.float32,
    )
    masks = tf.convert_to_tensor(batch["masks"], dtype=tf.float32)

    with tf.GradientTape() as tape:
        probs, values = model(states, training=True)
        probs = _masked_probs_tf(probs, masks)
        action_one_hot = tf.one_hot(actions, probs.shape[-1])
        action_probs = tf.reduce_sum(
            probs * action_one_hot,
            axis=1,
            keepdims=True,
        )
        log_probabilities = tf.math.log(tf.maximum(action_probs, 1e-12))
        policy_loss = -tf.reduce_mean(log_probabilities * advantages_tensor)
        value_loss = tf.reduce_mean(tf.square(returns - values))
        entropy = -tf.reduce_mean(tf.reduce_sum(
            probs * tf.math.log(tf.maximum(probs, 1e-12)),
            axis=1,
        ))
        loss = (
            policy_loss
            + float(args.value_coef) * value_loss
            - float(args.entropy_coef) * entropy
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    if float(args.max_grad_norm) > 0 and pairs:
        clipped, _ = tf.clip_by_global_norm(
            [gradient for gradient, _ in pairs],
            float(args.max_grad_norm),
        )
        pairs = [
            (gradient, variable)
            for gradient, (_, variable) in zip(clipped, pairs)
        ]
    optimizer.apply_gradients(pairs)
    return {
        "policy_loss": float(policy_loss.numpy()),
        "value_loss": float(value_loss.numpy()),
        "entropy": float(entropy.numpy()),
    }


def _gae(rewards, values, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + float(gamma) * values[index + 1] - values[index]
        running = delta + float(gamma) * float(gae_lambda) * running
        advantages[index] = running
    return advantages, advantages + values[:-1]


def _set_optimizer_lr(optimizer, value):
    learning_rate = optimizer.learning_rate
    if hasattr(learning_rate, "assign"):
        learning_rate.assign(float(value))
    else:
        optimizer.learning_rate = float(value)


def _init_train_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDS).writeheader()


def _append_train_log(path, row):
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDS)
        writer.writerow({field: row[field] for field in TRAIN_LOG_FIELDS})
