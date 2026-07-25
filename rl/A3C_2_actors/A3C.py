import argparse
import csv
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import tensorflow as tf
os.environ.setdefault("WANDB_MODE", "offline")
import wandb

from .bile import BILEModule, normalize_direction, sample_direction
from .critic import Critic
from .operation_actor import OperationActor
from .pipeline_environment import PipelineEnvironment
from .set_actor import SetActor

tf.keras.backend.set_floatx('float64')

now = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=20)
parser.add_argument('--actor_lr', type=float, default=0.00003)
parser.add_argument('--critic_lr', type=float, default=0.00003)
parser.add_argument('--workers', type=int, default=12)
parser.add_argument('--lstm_steps', type=int, default=5)
parser.add_argument('--target_set', type=str, default=None)
parser.add_argument('--target_seed', type=int, default=None)
parser.add_argument('--target_samples_per_file', type=int, default=100)

parser.add_argument('--alpha', type=float, default=0.5, help="Scaling factor for Exploration Reward")
parser.add_argument('--w_ext', type=float, default=8.0, help="Weight for Extrinsic Reward")
parser.add_argument('--w_int', type=float, default=0.2, help="Weight for Galaxy interestingness")
parser.add_argument('--w_bile', type=float, default=1.0, help="Weight for BILE directional reward")
parser.add_argument('--bile_bonus_clip', type=float, default=1.0, help="Clip absolute BILE directional reward")
parser.add_argument('--bile_latent_dim', type=int, default=16)
parser.add_argument('--bile_lr', type=float, default=0.0003)
parser.add_argument('--bile_dense_units', type=int, default=256)
parser.add_argument('--bile_beta_pe', type=float, default=1.0)
parser.add_argument('--bile_metric_clip', type=float, default=10.0)
parser.add_argument('--bile_target_tau', type=float, default=0.01)
parser.add_argument('--bile_phi_weight', type=float, default=1.0)
parser.add_argument('--bile_dyn_weight', type=float, default=1.0)
parser.add_argument('--bile_pair_warmup_episodes', type=int, default=100)
parser.add_argument('--bile_replay_buffer_size', type=int, default=50000)
parser.add_argument('--bile_phi_batch_size', type=int, default=128)
parser.add_argument('--bile_min_replay_size', type=int, default=256)
parser.add_argument('--bile_success_prob', type=float, default=0.60, help="Probability of sampling z from successful BILE directions")
parser.add_argument('--bile_orthogonal_prob', type=float, default=0.30, help="Probability of sampling z from orthogonal latent directions")
parser.add_argument('--bile_success_noise_scale', type=float, default=0.20, help="Noise added when reusing successful BILE directions")
parser.add_argument('--bile_success_pool_size', type=int, default=256, help="Maximum stored successful BILE state-pair directions")
parser.add_argument('--bile_min_success_reward', type=float, default=0.0, help="Minimum positive extrinsic reward for storing a success pair")
parser.add_argument('--bile_success_score_clip', type=float, default=150.0)
parser.add_argument('--bile_success_weight_power', type=float, default=0.5)
parser.add_argument('--bile_success_trajectory_score_scale', type=float, default=1.0)
parser.add_argument('--bootstrap_window', type=int, default=15, help="Recent episode window for aggressive bootstrap")
parser.add_argument('--bootstrap_ext_threshold', type=float, default=10.0, help="Bootstrap if recent average extrinsic is below this")
parser.add_argument('--bootstrap_success_threshold', type=float, default=100.0, help="Episode is counted as successful if extrinsic reaches this value")
parser.add_argument('--bootstrap_success_ratio_threshold', type=float, default=0.30, help="Optional success-ratio bootstrap threshold")
parser.add_argument('--bootstrap_use_success_ratio', action='store_true', help="Use success-ratio bootstrap trigger in addition to average extrinsic trigger")
parser.add_argument('--bootstrap_zpool_threshold', type=int, default=20, help="Bootstrap if successful z pool is smaller than this")
parser.add_argument('--w_bootstrap', type=float, default=1.0, help="Weight for aggressive bootstrap diversity")
parser.add_argument('--bootstrap_distance_eta', type=float, default=0.1, help="Distance scale for bootstrap diversity")
parser.add_argument('--escape_window', type=int, default=20, help="Recent episode window for low-reward escape mode")
parser.add_argument('--escape_ext_threshold', type=float, default=10.0, help="Enable escape mode if recent average extrinsic is below this")
parser.add_argument('--escape_low_reward_threshold', type=float, default=10.0, help="Episode below this extrinsic is counted as low reward")
parser.add_argument('--escape_low_ratio_threshold', type=float, default=0.70, help="Enable escape/bootstrap if low-reward ratio in the recent window reaches this")
parser.add_argument('--escape_success_prob', type=float, default=0.80, help="Probability of using stored successful z while escape mode is active")
parser.add_argument('--escape_orthogonal_prob', type=float, default=0.80, help="Probability of orthogonal z while escape mode is active")
parser.add_argument('--escape_random_action_prob', type=float, default=0.20, help="Forced random valid-action probability while escape mode is active")
parser.add_argument('--bootstrap_random_action_prob', type=float, default=0.05, help="Forced random valid-action probability while bootstrap mode is active")
parser.add_argument('--escape_alpha', type=float, default=0.10, help="Exploration reward scale while escape mode is active")
parser.add_argument('--escape_bootstrap_scale', type=float, default=0.20, help="Scale bootstrap diversity reward while escape mode is active")

parser.add_argument('--notes', type=str, default="")
parser.add_argument('--mode', type=str, default="scattered")
parser.add_argument('--name', type=str, default="")
parser.add_argument('--output_dir', type=str, default="outputs")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_step', type=int, default=None)
parser.add_argument(
    '--operators',
    nargs='+',
    type=str,
    default=["by_facet", "by_superset", "by_neighbors", "by_distribution"],
)
args = parser.parse_args()

if args.resume_step is not None:
    args.resume = True

if args.name == "":
    args.name = (
        f"{args.mode}-bile-exploration-lstm-{args.lstm_steps}-"
        f"alr-{args.actor_lr}-clr-{args.critic_lr}-{now.strftime('%m%d%Y_%H%M%S')}"
    )

args.result_dir = str(Path(args.output_dir) / args.name)
Path(args.result_dir).mkdir(parents=True, exist_ok=True)

if not args.resume:
    args.id = wandb.util.generate_id()
    if not os.path.exists("saved_models/" + args.name):
        os.makedirs("saved_models/" + args.name)
    with open("saved_models/" + args.name + "/info.json", 'w') as f:
        json.dump(vars(args), f, indent=1)
    with open(Path(args.result_dir) / "info.json", 'w') as f:
        json.dump(vars(args), f, indent=1)
else:
    with open("./saved_models/" + args.name + "/info.json") as f:
        items = json.load(f)
        for key in items.keys():
            if key != "resume" and key != "resume_step":
                setattr(args, key, items[key])
    args.result_dir = str(Path(args.output_dir) / args.name)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

wandb.init(name=args.name, project="deep-rl-tf2", id=args.id, resume=args.resume, config=vars(args))


@ray.remote
class ParameterServer:
    def __init__(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim, args):
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.args = args
        self.episodes_done = args.resume_step if args.resume_step is not None else 0

        model_path = None
        if args.resume:
            if args.resume_step:
                model_path = f"./saved_models/{args.name}/{args.resume_step}/"
            else:
                model_path = f"./saved_models/{args.name}/current/"

        self.global_set_actor = SetActor(
            set_state_dim,
            set_action_dim,
            args.lstm_steps,
            args.actor_lr,
            args.name,
            model_path=model_path + "set_actor" if model_path else None,
        )
        self.global_operation_actor = OperationActor(
            operation_state_dim,
            operation_action_dim,
            args.lstm_steps,
            args.actor_lr,
            args.name,
            model_path=model_path + "operation_actor" if model_path else None,
        )
        self.global_critic = Critic(
            set_state_dim,
            args.lstm_steps,
            args.critic_lr,
            args.name,
            model_path=model_path + "critic" if model_path else None,
        )
        self.base_set_state_dim = int(getattr(args, "base_set_state_dim", max(1, set_state_dim - int(args.bile_latent_dim))))
        self.bile_latent_dim = int(args.bile_latent_dim)
        self.z_sample_count = 0
        self.success_states = []
        self.success_next_states = []
        self.success_scores = []
        self.success_kinds = []
        self.bile_buffer_states = None
        self.bile_buffer_next_states = None
        self.bile_buffer_set_actions = None
        self.bile_buffer_operation_actions = None
        self.bile_buffer_rewards_ext = None
        self.bile_buffer_size = 0
        self.bile_buffer_pos = 0
        self.bile_rng = np.random.default_rng(int(getattr(args, "target_seed", 0) or 0) + 20260621)
        self.recent_extrinsic_rewards = []
        self.global_sets_viewed = set()
        self.cumulative_extrinsic_reward = 0.0
        self.logged_exploration_state_ids = set()
        self.global_bile = BILEModule(
            state_dim=self.base_set_state_dim,
            set_action_dim=set_action_dim,
            operation_action_dim=operation_action_dim,
            latent_dim=self.bile_latent_dim,
            lr=args.bile_lr,
            dense_units=args.bile_dense_units,
            target_tau=args.bile_target_tau,
        )
        self.orthogonal_directions = self._build_orthogonal_directions(
            max(args.workers * 2, 16),
            self.bile_latent_dim,
        )

    def _normalize_direction(self, direction):
        return normalize_direction(direction, self.bile_latent_dim)

    def _build_orthogonal_directions(self, count, dim):
        random_matrix = np.random.normal(size=(dim, min(count, dim))).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)
        directions = [q[:, i].astype(np.float32) for i in range(q.shape[1])]
        while len(directions) < count:
            directions.append(self._normalize_direction(np.random.normal(size=dim)))
        return directions

    def _sample_random_direction(self):
        return sample_direction(self.bile_rng, self.bile_latent_dim)

    def _recent_extrinsic_average(self):
        window = max(1, int(self.args.bootstrap_window))
        if not self.recent_extrinsic_rewards:
            return 0.0
        recent = self.recent_extrinsic_rewards[-window:]
        return float(np.mean(recent))

    def _recent_success_ratio(self):
        window = max(1, int(self.args.bootstrap_window))
        if not self.recent_extrinsic_rewards:
            return 0.0
        recent = self.recent_extrinsic_rewards[-window:]
        success_threshold = float(self.args.bootstrap_success_threshold)
        return float(np.mean([1.0 if value >= success_threshold else 0.0 for value in recent]))

    def _recent_low_reward_ratio(self, window=None):
        window = max(1, int(window if window is not None else self.args.bootstrap_window))
        if not self.recent_extrinsic_rewards:
            return 0.0
        recent = self.recent_extrinsic_rewards[-window:]
        low_threshold = float(self.args.escape_low_reward_threshold)
        return float(np.mean([1.0 if value < low_threshold else 0.0 for value in recent]))

    def _bootstrap_active(self):
        zpool_small = len(self.success_states) < int(self.args.bootstrap_zpool_threshold)
        recent_low = (
            self._recent_extrinsic_average() < float(self.args.bootstrap_ext_threshold)
            or self._recent_low_reward_ratio(self.args.bootstrap_window) >= float(self.args.escape_low_ratio_threshold)
        )
        recent_weak = (
            self._recent_success_ratio() < float(self.args.bootstrap_success_ratio_threshold)
            if self.args.bootstrap_use_success_ratio
            else False
        )
        return bool(zpool_small or recent_low or recent_weak)

    def _escape_active(self):
        window = max(1, int(self.args.escape_window))
        if len(self.recent_extrinsic_rewards) < window:
            return False
        recent = self.recent_extrinsic_rewards[-window:]
        recent_avg_low = float(np.mean(recent)) < float(self.args.escape_ext_threshold)
        recent_low_ratio = self._recent_low_reward_ratio(window) >= float(self.args.escape_low_ratio_threshold)
        return bool(recent_avg_low or recent_low_ratio)

    def _sample_success_direction(self):
        if not self.success_states:
            return None
        scores = np.asarray(self.success_scores, dtype=np.float64)
        score_clip = float(self.args.bile_success_score_clip)
        if score_clip > 0.0:
            scores = np.minimum(scores, score_clip)
        weights = np.power(np.maximum(scores, 0.0), float(self.args.bile_success_weight_power)) + 1e-6
        probs = weights / max(float(weights.sum()), 1e-12)
        index = int(self.bile_rng.choice(len(self.success_states), p=probs))
        pair = np.asarray(
            [self.success_states[index], self.success_next_states[index]],
            dtype=np.float32,
        ).reshape((2, self.base_set_state_dim))
        phi_pair = self.global_bile.embed(pair)
        base_direction = phi_pair[1] - phi_pair[0]
        noise = self._sample_random_direction()
        return self._normalize_direction(base_direction + (self.args.bile_success_noise_scale * noise))

    def sample_bile_direction(self):
        self.z_sample_count += 1
        escape_active = self._escape_active()
        bootstrap_active = self._bootstrap_active() or escape_active
        if escape_active:
            success_prob = max(0.0, min(1.0, float(self.args.escape_success_prob)))
            orthogonal_prob = max(0.0, min(1.0, float(self.args.escape_orthogonal_prob)))
            draw = float(self.bile_rng.random())
            success_z = self._sample_success_direction() if self.success_states else None
            if success_z is not None and draw < success_prob:
                z = success_z
                z_source = "escape_success"
            else:
                if float(self.bile_rng.random()) < orthogonal_prob:
                    index = self.z_sample_count % len(self.orthogonal_directions)
                    z = self.orthogonal_directions[index]
                    z_source = "escape_orthogonal"
                else:
                    z = self._sample_random_direction()
                    z_source = "escape_random"
            return {
                "z": z,
                "bootstrap_active": True,
                "escape_active": True,
                "zpool_size": len(self.success_states),
                "recent_ext_avg": self._recent_extrinsic_average(),
                "recent_success_ratio": self._recent_success_ratio(),
                "recent_low_reward_ratio": self._recent_low_reward_ratio(self.args.escape_window),
                "z_source": z_source,
            }

        if bootstrap_active:
            if float(self.bile_rng.random()) < 0.80:
                index = self.z_sample_count % len(self.orthogonal_directions)
                z = self.orthogonal_directions[index]
                z_source = "bootstrap_orthogonal"
            else:
                z = self._sample_random_direction()
                z_source = "bootstrap_random"
            return {
                "z": z,
                "bootstrap_active": True,
                "escape_active": False,
                "zpool_size": len(self.success_states),
                "recent_ext_avg": self._recent_extrinsic_average(),
                "recent_success_ratio": self._recent_success_ratio(),
                "recent_low_reward_ratio": self._recent_low_reward_ratio(self.args.bootstrap_window),
                "z_source": z_source,
            }

        success_prob = max(0.0, min(1.0, self.args.bile_success_prob))
        orthogonal_prob = max(0.0, min(1.0, self.args.bile_orthogonal_prob))
        draw = float(self.bile_rng.random())

        if self.success_states and draw < success_prob:
            return {
                "z": self._sample_success_direction(),
                "bootstrap_active": False,
                "escape_active": False,
                "zpool_size": len(self.success_states),
                "recent_ext_avg": self._recent_extrinsic_average(),
                "recent_success_ratio": self._recent_success_ratio(),
                "recent_low_reward_ratio": self._recent_low_reward_ratio(self.args.bootstrap_window),
                "z_source": "success",
            }

        if draw < success_prob + orthogonal_prob:
            index = self.z_sample_count % len(self.orthogonal_directions)
            z = self.orthogonal_directions[index]
            z_source = "orthogonal"
        else:
            z = self._sample_random_direction()
            z_source = "random"

        return {
            "z": z,
            "bootstrap_active": False,
            "escape_active": False,
            "zpool_size": len(self.success_states),
            "recent_ext_avg": self._recent_extrinsic_average(),
            "recent_success_ratio": self._recent_success_ratio(),
            "recent_low_reward_ratio": self._recent_low_reward_ratio(self.args.bootstrap_window),
            "z_source": z_source,
        }

    def update_bile_success_transitions(self, states, next_states, scores, kinds=None):
        states = list(states or [])
        next_states = list(next_states or [])
        scores = list(scores or [])
        kinds = list(kinds or ["local"] * len(states))
        if len(kinds) < len(states):
            kinds.extend(["local"] * (len(states) - len(kinds)))
        for state, next_state, score, kind in zip(states, next_states, scores, kinds):
            score = float(score)
            if score <= self.args.bile_min_success_reward:
                continue
            score_clip = float(self.args.bile_success_score_clip)
            if score_clip > 0.0:
                score = min(score, score_clip)
            self.success_states.append(np.asarray(state, dtype=np.float32).reshape(self.base_set_state_dim))
            self.success_next_states.append(np.asarray(next_state, dtype=np.float32).reshape(self.base_set_state_dim))
            self.success_scores.append(score)
            self.success_kinds.append(str(kind or "local"))

        pool_size = max(1, int(self.args.bile_success_pool_size))
        if len(self.success_states) > pool_size:
            order = np.argsort(np.asarray(self.success_scores, dtype=np.float64))[::-1][:pool_size]
            self.success_states = [self.success_states[int(i)] for i in order]
            self.success_next_states = [self.success_next_states[int(i)] for i in order]
            self.success_scores = [self.success_scores[int(i)] for i in order]
            self.success_kinds = [self.success_kinds[int(i)] for i in order]

        return len(self.success_states)

    def update_and_sample_bile_batch(
        self,
        states,
        set_actions,
        operation_actions,
        rewards_ext,
        next_states,
        episode,
    ):
        stored = self._store_bile_transitions(
            states,
            set_actions,
            operation_actions,
            rewards_ext,
            next_states,
        )
        if stored <= 0:
            return {"source": "local_empty", "buffer_size": int(self.bile_buffer_size)}

        warmup_episodes = int(self.args.bile_pair_warmup_episodes)
        if int(episode) <= warmup_episodes:
            return {"source": "local_warmup", "buffer_size": int(self.bile_buffer_size)}

        min_replay_size = int(self.args.bile_min_replay_size)
        if self.bile_buffer_size < max(2, min_replay_size):
            return {"source": "local_small_buffer", "buffer_size": int(self.bile_buffer_size)}

        batch_size = int(self.args.bile_phi_batch_size)
        sample_size = min(max(2, batch_size), int(self.bile_buffer_size))
        idx = self.bile_rng.choice(int(self.bile_buffer_size), size=sample_size, replace=False)
        pair_indices = self.bile_rng.permutation(sample_size).astype(np.int64)
        if sample_size > 1 and np.any(pair_indices == np.arange(sample_size, dtype=np.int64)):
            pair_indices = np.roll(pair_indices, 1)
        return {
            "source": "replay_random_perm",
            "buffer_size": int(self.bile_buffer_size),
            "states": self.bile_buffer_states[idx].astype(np.float32).tolist(),
            "next_states": self.bile_buffer_next_states[idx].astype(np.float32).tolist(),
            "set_actions": self.bile_buffer_set_actions[idx].astype(np.int64).reshape(-1, 1).tolist(),
            "operation_actions": self.bile_buffer_operation_actions[idx].astype(np.int64).reshape(-1, 1).tolist(),
            "rewards_ext": self.bile_buffer_rewards_ext[idx].astype(np.float32).reshape(-1, 1).tolist(),
            "pair_indices": pair_indices.astype(np.int64).tolist(),
        }

    def _store_bile_transitions(self, states, set_actions, operation_actions, rewards_ext, next_states):
        max_size = int(self.args.bile_replay_buffer_size)
        if max_size <= 0:
            return 0

        state_dim = int(self.base_set_state_dim)
        states_np = np.asarray(states, dtype=np.float32).reshape((-1, state_dim))
        next_states_np = np.asarray(next_states, dtype=np.float32).reshape((-1, state_dim))
        set_actions_np = np.asarray(set_actions, dtype=np.int64).reshape(-1)
        operation_actions_np = np.asarray(operation_actions, dtype=np.int64).reshape(-1)
        rewards_np = np.asarray(rewards_ext, dtype=np.float32).reshape((-1, 1))
        count = min(
            states_np.shape[0],
            next_states_np.shape[0],
            set_actions_np.shape[0],
            operation_actions_np.shape[0],
            rewards_np.shape[0],
        )
        if count <= 0:
            return 0

        states_np = states_np[-count:]
        next_states_np = next_states_np[-count:]
        set_actions_np = set_actions_np[-count:]
        operation_actions_np = operation_actions_np[-count:]
        rewards_np = rewards_np[-count:]
        if count > max_size:
            states_np = states_np[-max_size:]
            next_states_np = next_states_np[-max_size:]
            set_actions_np = set_actions_np[-max_size:]
            operation_actions_np = operation_actions_np[-max_size:]
            rewards_np = rewards_np[-max_size:]
            count = max_size

        if self.bile_buffer_states is None or self.bile_buffer_states.shape[0] != max_size:
            self.bile_buffer_states = np.zeros((max_size, state_dim), dtype=np.float32)
            self.bile_buffer_next_states = np.zeros((max_size, state_dim), dtype=np.float32)
            self.bile_buffer_set_actions = np.zeros((max_size,), dtype=np.int64)
            self.bile_buffer_operation_actions = np.zeros((max_size,), dtype=np.int64)
            self.bile_buffer_rewards_ext = np.zeros((max_size, 1), dtype=np.float32)
            self.bile_buffer_size = 0
            self.bile_buffer_pos = 0

        written = 0
        while written < count:
            take = min(count - written, max_size - int(self.bile_buffer_pos))
            start = int(self.bile_buffer_pos)
            end = start + take
            src_end = written + take
            self.bile_buffer_states[start:end] = states_np[written:src_end]
            self.bile_buffer_next_states[start:end] = next_states_np[written:src_end]
            self.bile_buffer_set_actions[start:end] = set_actions_np[written:src_end]
            self.bile_buffer_operation_actions[start:end] = operation_actions_np[written:src_end]
            self.bile_buffer_rewards_ext[start:end] = rewards_np[written:src_end]
            self.bile_buffer_pos = (self.bile_buffer_pos + take) % max_size
            self.bile_buffer_size = min(max_size, self.bile_buffer_size + take)
            written = src_end
        return int(count)

    def record_episode_metrics(self, ep_ext_score, episode_set_ids):
        self.recent_extrinsic_rewards.append(float(ep_ext_score))
        max_len = max(1, max(int(self.args.bootstrap_window), int(self.args.escape_window)) * 4)
        if len(self.recent_extrinsic_rewards) > max_len:
            self.recent_extrinsic_rewards = self.recent_extrinsic_rewards[-max_len:]

        episode_sets = set()
        for set_id in episode_set_ids:
            try:
                parsed_set_id = int(set_id)
            except (TypeError, ValueError):
                continue
            if parsed_set_id >= 0:
                episode_sets.add(parsed_set_id)

        self.global_sets_viewed.update(episode_sets)
        self.cumulative_extrinsic_reward += float(ep_ext_score)

        episode_sets_viewed = len(episode_sets)
        cumulative_unique_sets_viewed = len(self.global_sets_viewed)
        stats = {
            "zpool_size": len(self.success_states),
            "recent_ext_avg": self._recent_extrinsic_average(),
            "recent_success_ratio": self._recent_success_ratio(),
            "recent_low_reward_ratio": self._recent_low_reward_ratio(self.args.escape_window),
            "bootstrap_active": self._bootstrap_active(),
            "escape_active": self._escape_active(),
            "sets_viewed": episode_sets_viewed,
            "cumulative_unique_sets_viewed": cumulative_unique_sets_viewed,
            "target_efficiency": float(ep_ext_score) / max(episode_sets_viewed, 1),
            "cumulative_extrinsic_reward": self.cumulative_extrinsic_reward,
            "cumulative_target_efficiency": self.cumulative_extrinsic_reward / max(cumulative_unique_sets_viewed, 1),
        }
        stats["episode"] = self.increment_and_check_save()
        return stats

    def record_exploration_logs(self, episode, trace_rows, state_rows, context=None):
        context = context or {}
        result_dir = Path(getattr(self.args, "result_dir", "."))
        result_dir.mkdir(parents=True, exist_ok=True)

        if trace_rows:
            trace_file = result_dir / f"{self.args.name}_exploration_trace.csv"
            file_exists = os.path.isfile(trace_file)
            with open(trace_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'episode',
                        'agent_id',
                        'step',
                        'set_id',
                        'step_extrinsic_reward',
                        'step_interestingness',
                        'operator',
                        'parameter',
                        'input_set_id',
                        'operation_action',
                        'bootstrap_active',
                        'escape_active',
                        'z_source',
                    ])
                for row in trace_rows:
                    writer.writerow([
                        episode,
                        row.get("agent_id", -1),
                        row.get("step", -1),
                        row.get("set_id", -1),
                        row.get("step_extrinsic_reward", 0.0),
                        row.get("step_interestingness", 0.0),
                        row.get("operator", ""),
                        row.get("parameter", ""),
                        row.get("input_set_id", -1),
                        row.get("operation_action", -1),
                        int(context.get("bootstrap_active", 0)),
                        int(context.get("escape_active", 0)),
                        context.get("z_source", ""),
                    ])

        new_state_rows = []
        for row in state_rows:
            try:
                set_id = int(row.get("set_id", -1))
            except (TypeError, ValueError):
                continue
            if set_id < 0 or set_id in self.logged_exploration_state_ids:
                continue
            state = row.get("state", [])
            if not state:
                continue
            self.logged_exploration_state_ids.add(set_id)
            new_state_rows.append((set_id, state))

        if new_state_rows:
            state_file = result_dir / f"{self.args.name}_visited_set_states.csv"
            file_exists = os.path.isfile(state_file)
            with open(state_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    state_dim = len(new_state_rows[0][1])
                    writer.writerow(['set_id'] + [f'state_{i}' for i in range(state_dim)])
                for set_id, state in new_state_rows:
                    writer.writerow([set_id] + state)

        return {
            "trace_rows": len(trace_rows),
            "new_state_rows": len(new_state_rows),
        }

    def apply_gradients_and_get_weights(
        self,
        set_grads,
        op_grads,
        critic_grads,
        bile_encoder_grads=None,
        bile_dynamics_grads=None,
    ):
        if set_grads:
            self.global_set_actor.apply_gradients(set_grads)
        if op_grads:
            self.global_operation_actor.apply_gradients(op_grads)
        if critic_grads:
            self.global_critic.apply_gradients(critic_grads)
        self.global_bile.apply_gradients(bile_encoder_grads, bile_dynamics_grads)
        return self.get_weights()

    def get_weights(self):
        return {
            'set': self.global_set_actor.model.get_weights(),
            'op': self.global_operation_actor.model.get_weights(),
            'critic': self.global_critic.model.get_weights(),
            'bile': self.global_bile.get_weights(),
        }

    def increment_and_check_save(self):
        self.episodes_done += 1
        ep = self.episodes_done
        if ep != 0 and ep % 250 == 0:
            self.global_operation_actor.save_model(step=ep)
            self.global_set_actor.save_model(step=ep)
            self.global_critic.save_model(step=ep)
            self.global_bile.save_models(Path("saved_models") / self.args.name / str(ep) / "bile")
        return ep

    def save_final_models(self):
        self.global_operation_actor.save_model()
        self.global_set_actor.save_model()
        self.global_critic.save_model()
        self.global_bile.save_models(Path("saved_models") / self.args.name / "final" / "bile")


@ray.remote
class WorkerAgent:
    def __init__(self, pipeline, ps_handle, agentId, args, episode_steps):
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.agentId = agentId
        self.ps = ps_handle
        self.args = args
        self.episode_steps = episode_steps

        self.env = PipelineEnvironment(
            pipeline,
            mode=args.mode,
            agentId=agentId,
            episode_steps=episode_steps,
            operators=args.operators,
            target_set_name=args.target_set,
            target_seed=args.target_seed,
            target_samples_per_file=args.target_samples_per_file,
        )
        self.base_set_state_dim = self.env.set_state_dim
        self.base_operation_state_dim = self.env.operation_state_dim
        self.bile_latent_dim = int(args.bile_latent_dim)
        self.set_state_dim = self.base_set_state_dim + self.bile_latent_dim
        self.operation_state_dim = self.base_operation_state_dim + self.bile_latent_dim
        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        self.steps = args.lstm_steps

        self.set_actor = SetActor(self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, args.name)
        self.operation_actor = OperationActor(
            self.operation_state_dim,
            self.operation_action_dim,
            self.steps,
            args.actor_lr,
            args.name,
        )
        self.critic = Critic(self.set_state_dim, self.steps, args.critic_lr, args.name)
        self.bile = BILEModule(
            state_dim=self.base_set_state_dim,
            set_action_dim=self.set_action_dim,
            operation_action_dim=self.operation_action_dim,
            latent_dim=self.bile_latent_dim,
            lr=args.bile_lr,
            dense_units=args.bile_dense_units,
            target_tau=args.bile_target_tau,
        )
        self.bootstrap_visit_counts = {}

        self.sync_with_ps()

    def sync_with_ps(self):
        weights = ray.get(self.ps.get_weights.remote())
        self.set_actor.model.set_weights(weights['set'])
        self.operation_actor.model.set_weights(weights['op'])
        self.critic.model.set_weights(weights['critic'])
        self.bile.set_weights(weights['bile'])

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value
        for k in reversed(range(0, len(rewards))):
            cumulative = self.args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def list_to_batch(self, list_data):
        batch = list_data[0]
        for elem in list_data[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def sample_bile_direction(self):
        sample = ray.get(self.ps.sample_bile_direction.remote())
        return (
            np.array(sample["z"], dtype=np.float32),
            bool(sample["bootstrap_active"]),
            bool(sample["escape_active"]),
            int(sample["zpool_size"]),
            float(sample["recent_ext_avg"]),
            float(sample["recent_success_ratio"]),
            float(sample["recent_low_reward_ratio"]),
            str(sample["z_source"]),
        )

    def sample_action(self, probs, action_dim, random_prob=0.0, fallback_action=None):
        probs = np.array(probs, dtype=np.float64).flatten()
        valid_actions = np.where(np.isfinite(probs) & (probs > 0.0))[0]

        if valid_actions.size == 0:
            if fallback_action is not None:
                return int(fallback_action)
            return int(np.random.choice(action_dim))

        if random_prob > 0.0 and np.random.random() < random_prob:
            return int(np.random.choice(valid_actions))

        safe_probs = np.zeros(action_dim, dtype=np.float64)
        copy_len = min(action_dim, probs.size)
        safe_probs[:copy_len] = np.nan_to_num(probs[:copy_len], nan=0.0, posinf=0.0, neginf=0.0)
        safe_probs[safe_probs < 0.0] = 0.0
        total = float(safe_probs.sum())
        if total <= 0.0:
            if fallback_action is not None:
                return int(fallback_action)
            return int(np.random.choice(valid_actions))

        safe_probs /= total
        return int(np.random.choice(action_dim, p=safe_probs))

    def augment_set_state(self, set_state, z):
        return np.concatenate([
            np.array(set_state, dtype=np.float32).flatten(),
            np.array(z, dtype=np.float32).flatten(),
        ])

    def augment_operation_state(self, operation_state, z):
        return np.concatenate([
            np.array(operation_state, dtype=np.float32).flatten(),
            np.array(z, dtype=np.float32).flatten(),
        ])

    def embed_bile_state(self, state):
        phi = self.bile.embed(np.asarray(state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
        return np.asarray(phi[0], dtype=np.float32).reshape(-1)

    def compute_bootstrap_diversity_reward(self, phi_s_next, episode_history_phi):
        state_hash = hash(np.array(phi_s_next, dtype=np.float32).tobytes())
        self.bootstrap_visit_counts[state_hash] = self.bootstrap_visit_counts.get(state_hash, 0) + 1
        count_reward = 1.0 / np.sqrt(float(self.bootstrap_visit_counts[state_hash]))

        if not episode_history_phi:
            distance_reward = 1.0
        else:
            distances = [np.linalg.norm(phi_s_next - history_phi) for history_phi in episode_history_phi]
            min_distance = float(np.min(distances))
            distance_reward = float(1.0 - np.exp(-self.args.bootstrap_distance_eta * min_distance))

        return float(0.5 * count_reward + 0.5 * distance_reward)

    def train_loop(self, max_episodes):
        curr_episode = 0

        while max_episodes >= curr_episode:
            set_state_batch = []
            operation_state_batch = []
            set_action_batch = []
            operation_action_batch = []
            reward_batch = []
            bile_state_batch = []
            bile_next_state_batch = []
            bile_set_action_batch = []
            bile_operation_action_batch = []
            bile_reward_ext_batch = []

            ep_ext_score = 0
            ep_int_score = 0
            ep_bile_score = 0
            ep_bootstrap_score = 0
            ep_exploration_score = 0
            ep_total_reward = 0
            episode_success_states = []
            episode_success_next_states = []
            episode_success_scores = []
            episode_success_kinds = []
            episode_history_phi = []
            bile_phi_loss = 0.0
            bile_dyn_loss = 0.0
            bile_prediction_error = 0.0
            bile_mean_d_phi = 0.0
            bile_mean_metric_target = 0.0
            bile_update_count = 0
            bile_pair_source = ""
            bile_replay_size = 0
            done = False

            set_action_steps = [[0.0] * self.set_state_dim] * self.steps
            operation_action_steps = [[0.0] * self.operation_state_dim] * self.steps

            (
                bile_z,
                bootstrap_active,
                escape_active,
                zpool_size,
                recent_ext_avg,
                recent_success_ratio,
                recent_low_reward_ratio,
                z_source,
            ) = self.sample_bile_direction()
            forced_random_action_prob = (
                float(self.args.escape_random_action_prob)
                if escape_active
                else (float(self.args.bootstrap_random_action_prob) if bootstrap_active else 0.0)
            )
            set_state = self.env.reset()
            segment_anchor_state = np.asarray(set_state, dtype=np.float32).reshape(-1).tolist()
            segment_positive_reward = 0.0
            set_action_steps.pop(0)
            set_action_steps.append(self.augment_set_state(set_state, bile_z))
            episode_history_phi.append(self.embed_bile_state(set_state))
            failed = False

            try:
                while not done:
                    probs = self.set_actor.model.predict(
                        np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)),
                        verbose=0,
                    )
                    probs = self.env.fix_possible_set_action_probs(probs[0])
                    set_action = self.sample_action(
                        probs,
                        self.set_action_dim,
                        random_prob=forced_random_action_prob,
                        fallback_action=0,
                    )

                    operation_state = self.augment_operation_state(
                        self.env.get_operation_state(set_action),
                        bile_z,
                    )
                    operation_action_steps.pop(0)
                    operation_action_steps.append(operation_state)

                    probs = self.operation_actor.model.predict(
                        np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)),
                        verbose=0,
                    )
                    probs = self.env.fix_possible_operation_action_probs(set_action, probs[0])
                    operation_action = self.sample_action(
                        probs,
                        self.operation_action_dim,
                        random_prob=forced_random_action_prob,
                    )

                    current_set_state = np.asarray(set_state, dtype=np.float32).reshape(-1)
                    next_set_state, env_r_ext, env_r_int_js, done, set_op_pair = self.env.step(
                        set_action,
                        operation_action,
                    )

                    r_ext = float(np.squeeze(env_r_ext))
                    r_int = float(np.squeeze(env_r_int_js))
                    phi_s_next = self.embed_bile_state(next_set_state)
                    r_bile = self.bile.compute_bonus(
                        current_set_state,
                        next_set_state,
                        bile_z,
                        clip_value=self.args.bile_bonus_clip,
                    )
                    r_bootstrap = (
                        self.compute_bootstrap_diversity_reward(phi_s_next, episode_history_phi)
                        if bootstrap_active
                        else 0.0
                    )
                    if escape_active:
                        r_bootstrap *= float(self.args.escape_bootstrap_scale)
                    if r_ext > float(self.args.bile_min_success_reward):
                        episode_success_states.append(current_set_state.tolist())
                        episode_success_next_states.append(np.asarray(next_set_state, dtype=np.float32).reshape(-1).tolist())
                        episode_success_scores.append(r_ext)
                        episode_success_kinds.append("local")
                        segment_positive_reward += max(float(r_ext), 0.0)
                        trajectory_score = max(
                            float(r_ext),
                            segment_positive_reward * float(self.args.bile_success_trajectory_score_scale),
                        )
                        episode_success_states.append(list(segment_anchor_state))
                        episode_success_next_states.append(np.asarray(next_set_state, dtype=np.float32).reshape(-1).tolist())
                        episode_success_scores.append(float(trajectory_score))
                        episode_success_kinds.append("trajectory")

                    episode_history_phi.append(phi_s_next)

                    exploration_reward = (
                        (self.args.w_int * r_int)
                        + (self.args.w_bile * r_bile)
                        + (self.args.w_bootstrap * r_bootstrap)
                    )
                    effective_alpha = float(self.args.escape_alpha) if escape_active else float(self.args.alpha)
                    total_step_reward = (self.args.w_ext * r_ext) + (effective_alpha * exploration_reward)

                    ep_ext_score += r_ext
                    ep_int_score += r_int
                    ep_bile_score += r_bile
                    ep_bootstrap_score += r_bootstrap
                    ep_exploration_score += exploration_reward
                    ep_total_reward += total_step_reward

                    next_set_action_steps = set_action_steps.copy()
                    next_set_action_steps.pop(0)
                    next_set_action_steps.append(self.augment_set_state(next_set_state, bile_z))

                    reward_batch.append(np.reshape(total_step_reward, [1, 1]))
                    set_state_batch.append(np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                    set_action_batch.append(np.reshape(set_action, [1, 1]))
                    operation_state_batch.append(
                        np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim))
                    )
                    operation_action_batch.append(np.reshape(operation_action, [1, 1]))
                    bile_state_batch.append(current_set_state.reshape((1, self.base_set_state_dim)))
                    bile_next_state_batch.append(np.asarray(next_set_state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
                    bile_set_action_batch.append(np.reshape(set_action, [1, 1]))
                    bile_operation_action_batch.append(np.reshape(operation_action, [1, 1]))
                    bile_reward_ext_batch.append(np.reshape(r_ext, [1, 1]))

                    if len(set_state_batch) >= self.args.update_interval or done:
                        set_states = self.list_to_batch(set_state_batch)
                        set_actions = self.list_to_batch(set_action_batch)
                        operation_states = self.list_to_batch(operation_state_batch)
                        operation_actions = self.list_to_batch(operation_action_batch)
                        rewards = self.list_to_batch(reward_batch)

                        next_v_value = self.critic.model.predict(
                            np.array(next_set_action_steps).reshape((1, self.steps, self.set_state_dim)),
                            verbose=0,
                        )
                        td_targets = self.n_step_td_target(rewards, next_v_value, done)
                        advantages = td_targets - self.critic.model.predict(set_states, verbose=0)

                        try:
                            set_grads, _ = self.set_actor.get_gradients(set_states, set_actions, advantages)
                            op_grads, _ = self.operation_actor.get_gradients(
                                operation_states,
                                operation_actions,
                                advantages,
                            )
                            critic_grads, _ = self.critic.get_gradients(set_states, td_targets)

                            set_grads_np = [g.numpy() for g in set_grads]
                            op_grads_np = [g.numpy() for g in op_grads]
                            critic_grads_np = [g.numpy() for g in critic_grads]

                            bile_encoder_grads_np = None
                            bile_dynamics_grads_np = None
                            if bile_state_batch:
                                local_bile_states = self.list_to_batch(bile_state_batch)
                                local_bile_next_states = self.list_to_batch(bile_next_state_batch)
                                local_bile_set_actions = self.list_to_batch(bile_set_action_batch)
                                local_bile_operation_actions = self.list_to_batch(bile_operation_action_batch)
                                local_bile_rewards_ext = self.list_to_batch(bile_reward_ext_batch)
                                bile_pair_sample = ray.get(
                                    self.ps.update_and_sample_bile_batch.remote(
                                        local_bile_states,
                                        local_bile_set_actions,
                                        local_bile_operation_actions,
                                        local_bile_rewards_ext,
                                        local_bile_next_states,
                                        curr_episode + 1,
                                    )
                                )
                                bile_pair_source = str(bile_pair_sample.get("source", "local"))
                                bile_replay_size = int(bile_pair_sample.get("buffer_size", 0))
                                if bile_pair_sample.get("source") == "replay_random_perm":
                                    train_bile_states = np.asarray(bile_pair_sample["states"], dtype=np.float32)
                                    train_bile_next_states = np.asarray(bile_pair_sample["next_states"], dtype=np.float32)
                                    train_bile_set_actions = np.asarray(bile_pair_sample["set_actions"], dtype=np.int64)
                                    train_bile_operation_actions = np.asarray(
                                        bile_pair_sample["operation_actions"],
                                        dtype=np.int64,
                                    )
                                    train_bile_rewards_ext = np.asarray(
                                        bile_pair_sample["rewards_ext"],
                                        dtype=np.float32,
                                    )
                                    pair_indices = np.asarray(bile_pair_sample["pair_indices"], dtype=np.int64)
                                else:
                                    train_bile_states = local_bile_states
                                    train_bile_next_states = local_bile_next_states
                                    train_bile_set_actions = local_bile_set_actions
                                    train_bile_operation_actions = local_bile_operation_actions
                                    train_bile_rewards_ext = local_bile_rewards_ext
                                    pair_indices = None

                                bile_stats = self.bile.get_gradients(
                                    train_bile_states,
                                    train_bile_set_actions,
                                    train_bile_operation_actions,
                                    train_bile_rewards_ext,
                                    train_bile_next_states,
                                    pair_indices=pair_indices,
                                    gamma=self.args.gamma,
                                    beta_pe=self.args.bile_beta_pe,
                                    metric_clip=self.args.bile_metric_clip,
                                    phi_weight=self.args.bile_phi_weight,
                                    dyn_weight=self.args.bile_dyn_weight,
                                    use_reward_diff=True,
                                )
                                if bile_stats is not None:
                                    bile_encoder_grads_np = [
                                        grad.numpy() if grad is not None else None
                                        for grad in bile_stats["encoder_grads"]
                                    ]
                                    bile_dynamics_grads_np = [
                                        grad.numpy() if grad is not None else None
                                        for grad in bile_stats["dynamics_grads"]
                                    ]
                                    bile_phi_loss += float(bile_stats["phi_loss"])
                                    bile_dyn_loss += float(bile_stats["dyn_loss"])
                                    bile_prediction_error += float(bile_stats["prediction_error"])
                                    bile_mean_d_phi += float(bile_stats["mean_d_phi"])
                                    bile_mean_metric_target += float(bile_stats["mean_metric_target"])
                                    bile_update_count += 1

                            new_weights = ray.get(
                                self.ps.apply_gradients_and_get_weights.remote(
                                    set_grads_np,
                                    op_grads_np,
                                    critic_grads_np,
                                    bile_encoder_grads_np,
                                    bile_dynamics_grads_np,
                                )
                            )

                            self.set_actor.model.set_weights(new_weights['set'])
                            self.operation_actor.model.set_weights(new_weights['op'])
                            self.critic.model.set_weights(new_weights['critic'])
                            self.bile.set_weights(new_weights['bile'])

                        except Exception as error:
                            print(error)
                            traceback.print_tb(error.__traceback__)
                            print('Episode gradient push failed, retrying')
                            failed = True
                            done = True

                        set_state_batch = []
                        operation_state_batch = []
                        set_action_batch = []
                        operation_action_batch = []
                        reward_batch = []
                        bile_state_batch = []
                        bile_next_state_batch = []
                        bile_set_action_batch = []
                        bile_operation_action_batch = []
                        bile_reward_ext_batch = []

                    set_state = next_set_state
                    set_action_steps = next_set_action_steps

                if not failed:
                    success_pool_size = None
                    if bile_update_count > 0:
                        update_count = float(bile_update_count)
                        bile_phi_loss /= update_count
                        bile_dyn_loss /= update_count
                        bile_prediction_error /= update_count
                        bile_mean_d_phi /= update_count
                        bile_mean_metric_target /= update_count
                    if episode_success_states:
                        success_pool_size = ray.get(
                            self.ps.update_bile_success_transitions.remote(
                                episode_success_states,
                                episode_success_next_states,
                                episode_success_scores,
                                episode_success_kinds,
                            )
                        )
                    episode_set_ids = list(getattr(self.env, "sets_viewed", []))
                    bootstrap_stats = ray.get(self.ps.record_episode_metrics.remote(ep_ext_score, episode_set_ids))
                    curr_episode = bootstrap_stats["episode"]
                    trace_rows, state_rows = self.env.consume_exploration_logs()
                    ray.get(self.ps.record_exploration_logs.remote(
                        curr_episode,
                        trace_rows,
                        state_rows,
                        {
                            "bootstrap_active": int(bootstrap_active),
                            "escape_active": int(escape_active),
                            "z_source": z_source,
                            "bile_pair_source": bile_pair_source,
                        },
                    ))

                    print(
                        f'EP{curr_episode} Agent{self.agentId} | '
                        f'Ext_R: {ep_ext_score:.1f} | '
                        f'Int: {ep_int_score:.1f} | '
                        f'BILE: {ep_bile_score:.1f} | '
                        f'BootDiv: {ep_bootstrap_score:.1f} | '
                        f'Explore: {ep_exploration_score:.1f} | '
                        f'ZPool: {success_pool_size if success_pool_size is not None else bootstrap_stats["zpool_size"]} | '
                        f'RecentHit15: {bootstrap_stats["recent_success_ratio"]:.2f} | '
                        f'LowRatio: {bootstrap_stats["recent_low_reward_ratio"]:.2f} | '
                        f'Bootstrap: {bootstrap_active} | '
                        f'Escape: {escape_active} | '
                        f'Z: {z_source}'
                    )

                    result_dir = Path(getattr(self.args, "result_dir", "."))
                    result_dir.mkdir(parents=True, exist_ok=True)
                    csv_file = result_dir / f"{self.args.name}_fusion_rewards.csv"
                    file_exists = os.path.isfile(csv_file)

                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow([
                                'episode',
                                'extrinsic_reward',
                                'interestingness',
                                'bile_bonus',
                                'bootstrap_diversity',
                                'exploration_reward',
                                'total_reward',
                                'sets_viewed',
                                'cumulative_unique_sets_viewed',
                                'target_efficiency',
                                'cumulative_extrinsic_reward',
                                'cumulative_target_efficiency',
                                'zpool_size',
                                'recent_success_ratio',
                                'recent_low_reward_ratio',
                                'bootstrap_active',
                                'escape_active',
                                'z_source',
                                'bile_phi_loss',
                                'bile_dyn_loss',
                                'bile_prediction_error',
                                'bile_mean_d_phi',
                                'bile_mean_metric_target',
                                'bile_update_count',
                                'bile_pair_source',
                                'bile_replay_size',
                            ])
                        writer.writerow([
                            curr_episode,
                            ep_ext_score,
                            ep_int_score,
                            ep_bile_score,
                            ep_bootstrap_score,
                            ep_exploration_score,
                            ep_total_reward,
                            bootstrap_stats["sets_viewed"],
                            bootstrap_stats["cumulative_unique_sets_viewed"],
                            bootstrap_stats["target_efficiency"],
                            bootstrap_stats["cumulative_extrinsic_reward"],
                            bootstrap_stats["cumulative_target_efficiency"],
                            success_pool_size if success_pool_size is not None else bootstrap_stats["zpool_size"],
                            bootstrap_stats["recent_success_ratio"],
                            bootstrap_stats["recent_low_reward_ratio"],
                            int(bootstrap_active),
                            int(escape_active),
                            z_source,
                            bile_phi_loss,
                            bile_dyn_loss,
                            bile_prediction_error,
                            bile_mean_d_phi,
                            bile_mean_metric_target,
                            bile_update_count,
                            bile_pair_source,
                            bile_replay_size,
                        ])

            except Exception as error:
                print(error)
                traceback.print_tb(error.__traceback__)
                print('Episode failed, retrying')


class Agent:
    def __init__(self, env_name, pipeline=None):
        self.pipeline = pipeline
        self.env_name = env_name
        self.episode_steps = 250 if args.mode == "scattered" else 25
        self.num_workers = args.workers

        dummy_env = PipelineEnvironment(
            self.pipeline,
            mode=args.mode,
            episode_steps=self.episode_steps,
            operators=args.operators,
            target_set_name=args.target_set,
            target_seed=args.target_seed,
            target_samples_per_file=args.target_samples_per_file,
        )
        self.base_set_state_dim = dummy_env.set_state_dim
        self.base_operation_state_dim = dummy_env.operation_state_dim
        args.base_set_state_dim = self.base_set_state_dim
        args.base_operation_state_dim = self.base_operation_state_dim
        self.bile_latent_dim = int(args.bile_latent_dim)
        self.set_state_dim = self.base_set_state_dim + self.bile_latent_dim
        self.operation_state_dim = self.base_operation_state_dim + self.bile_latent_dim
        self.set_action_dim = dummy_env.set_action_space.n
        self.operation_action_dim = dummy_env.operation_action_space.n
        self.target_items = sorted(map(int, dummy_env.target_items)) if hasattr(dummy_env, "target_items") else []

        if not args.resume and len(self.target_items) > 0:
            target_snapshot_path = Path(args.result_dir) / "target_items.json"
            target_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_snapshot_path, "w") as f:
                json.dump(self.target_items, f, indent=1)
            saved_model_target_snapshot_path = Path("saved_models") / args.name / "target_items.json"
            saved_model_target_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(saved_model_target_snapshot_path, "w") as f:
                json.dump(self.target_items, f, indent=1)

    def train(self, max_episodes=1000):
        ps = ParameterServer.remote(
            self.set_state_dim,
            self.operation_state_dim,
            self.set_action_dim,
            self.operation_action_dim,
            args,
        )
        workers = [WorkerAgent.remote(self.pipeline, ps, i, args, self.episode_steps) for i in range(self.num_workers)]
        ray.get([worker.train_loop.remote(max_episodes) for worker in workers])
        ray.get(ps.save_final_models.remote())
