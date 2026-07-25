import argparse
import csv
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM

from .actions import build_action_space
from .bile import BILEModule, normalize_direction, sample_direction
from .data import load_covertype
from .environment import FixedSetEnvironment
from .fixed_sets import ensure_fixed_universe, load_fixed_universe
from .targets import resolve_target_set


tf.keras.backend.set_floatx("float32")


MIRA_BASELINES = {"mira", "mira_no_ext"}
BASELINES = {"paper_a3c", "atena", "atena_extrinsic", "pure_a3c", *MIRA_BASELINES}


class SetActor:
    def __init__(self, state_dim, action_dim, steps, lr, name, model_path=None, dense_units=512, lstm_units=512):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.steps = int(steps)
        self.name = name
        self.dense_units = int(dense_units)
        self.lstm_units = int(lstm_units)
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.entropy_beta = 0.05
        self.model = tf.keras.models.load_model(model_path) if model_path else self.create_model()

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input((self.steps, self.state_dim)),
                Dense(self.dense_units, activation="relu"),
                Dense(self.dense_units, activation="relu"),
                LSTM(self.lstm_units, return_sequences=False),
                Dense(self.action_dim, activation="softmax"),
            ]
        )

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        policy_loss = ce_loss(tf.cast(actions, tf.int32), logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def get_gradients(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(actions, logits, advantages)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return grads, loss

    def apply_gradients(self, grads):
        clean = []
        for grad, variable in zip(grads, self.model.trainable_variables):
            if grad is not None:
                clean.append((grad, variable))
        if clean:
            self.opt.apply_gradients(clean)

    def save_model(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save(directory)


class OperationActor:
    def __init__(self, state_dim, action_dim, steps, lr, name, model_path=None, dense_units=512, lstm_units=256):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.steps = int(steps)
        self.name = name
        self.dense_units = int(dense_units)
        self.lstm_units = int(lstm_units)
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.entropy_beta = 0.05
        self.model = tf.keras.models.load_model(model_path) if model_path else self.create_model()

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input((self.steps, self.state_dim)),
                Dense(self.dense_units, activation="relu"),
                Dense(self.dense_units, activation="relu"),
                LSTM(self.lstm_units, return_sequences=False),
                Dense(self.action_dim, activation="softmax"),
            ]
        )

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        policy_loss = ce_loss(tf.cast(actions, tf.int32), logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def get_gradients(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(actions, logits, advantages)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return grads, loss

    def apply_gradients(self, grads):
        clean = []
        for grad, variable in zip(grads, self.model.trainable_variables):
            if grad is not None:
                clean.append((grad, variable))
        if clean:
            self.opt.apply_gradients(clean)

    def save_model(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save(directory)


class Critic:
    def __init__(self, state_dim, steps, lr, name, model_path=None, dense_units=(512, 256, 128), lstm_units=128):
        self.state_dim = int(state_dim)
        self.steps = int(steps)
        self.name = name
        self.dense_units = tuple(int(unit) for unit in dense_units)
        self.lstm_units = int(lstm_units)
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.model = tf.keras.models.load_model(model_path) if model_path else self.create_model()

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input((self.steps, self.state_dim)),
                Dense(self.dense_units[0], activation="relu"),
                Dense(self.dense_units[1], activation="relu"),
                Dense(self.dense_units[2], activation="relu"),
                LSTM(self.lstm_units, return_sequences=False),
                Dense(1, activation="linear"),
            ]
        )

    def get_gradients(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            loss = tf.keras.losses.MeanSquaredError()(tf.stop_gradient(td_targets), v_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return grads, loss

    def apply_gradients(self, grads):
        clean = []
        for grad, variable in zip(grads, self.model.trainable_variables):
            if grad is not None:
                clean.append((grad, variable))
        if clean:
            self.opt.apply_gradients(clean)

    def save_model(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save(directory)


class CovertypeDualActorEnvironment:
    """Fixed-set Covertype environment exposed as a Galaxy-style dual actor task.

    Set actor: chooses one candidate next set slot.
    Operation actor: chooses the concrete graph action that reaches that slot.
    """

    def __init__(self, universe, actions, episode_steps=250, seed=0, candidate_slots=10):
        self.universe = universe
        self.actions = actions
        self.episode_steps = int(episode_steps)
        self.candidate_slots = int(candidate_slots)
        self.base_env = FixedSetEnvironment(universe, actions, episode_steps=episode_steps, seed=seed)
        self.set_state_dim = self.candidate_slots * self.universe.state_dim
        self.operation_state_dim = self.set_state_dim + self.universe.state_dim
        self.set_action_dim = self.candidate_slots
        self.operation_action_dim = self.universe.action_dim
        self.step_count = 0
        self.set_state = None
        self.candidate_set_ids = []
        self.candidate_action_ids = []
        self.candidate_bile_scores = []
        self.sets_viewed = set()
        self.exploration_trace_rows = []
        self.visited_set_state_rows = {}

    def reset(self):
        self.step_count = 0
        current = self.base_env.reset()
        self.sets_viewed = set()
        self.exploration_trace_rows = []
        self.visited_set_state_rows = {}
        self.visited_set_state_rows[int(self.base_env.current_set_id)] = current.astype(float).tolist()
        self._refresh_candidates()
        return self.set_state

    def _candidate_groups_for_set(self, set_id):
        set_id = int(set_id)
        row = np.asarray(self.universe.graph[set_id], dtype=np.int64)
        grouped = {}
        family_by_set = {}
        for action_id, next_set_id in enumerate(row.tolist()):
            next_set_id = int(next_set_id)
            if next_set_id == set_id:
                continue
            grouped.setdefault(next_set_id, []).append(int(action_id))
            family_by_set.setdefault(next_set_id, set()).add(str(self.actions[action_id].family))
        return list(grouped.keys()), grouped, family_by_set

    def _build_set_state(self, candidate_set_ids):
        pieces = []
        for set_id in candidate_set_ids:
            if int(set_id) >= 0:
                pieces.append(self.universe.state_for_set(int(set_id)))
            else:
                pieces.append(np.zeros(self.universe.state_dim, dtype=np.float32))
        return np.concatenate(pieces).astype(np.float32)

    def _apply_candidates(self, candidate_set_ids, grouped, score_by_set=None):
        candidate_set_ids = list(candidate_set_ids)
        candidate_action_ids = [list(grouped.get(int(set_id), [])) if int(set_id) >= 0 else [] for set_id in candidate_set_ids]
        while len(candidate_set_ids) < self.candidate_slots:
            candidate_set_ids.append(-1)
            candidate_action_ids.append([])
        candidate_set_ids = candidate_set_ids[: self.candidate_slots]
        candidate_action_ids = candidate_action_ids[: self.candidate_slots]

        self.candidate_set_ids = candidate_set_ids
        self.candidate_action_ids = candidate_action_ids
        if score_by_set is None:
            self.candidate_bile_scores = [float("nan") if int(set_id) >= 0 else -np.inf for set_id in candidate_set_ids]
        else:
            self.candidate_bile_scores = [float(score_by_set.get(int(set_id), -np.inf)) if int(set_id) >= 0 else -np.inf for set_id in candidate_set_ids]
        self.set_state = self._build_set_state(self.candidate_set_ids)

    def _refresh_candidates(self):
        current_set_id = int(self.base_env.current_set_id)
        all_candidate_set_ids, grouped, family_by_set = self._candidate_groups_for_set(current_set_id)
        candidate_set_ids = self._select_candidate_set_ids(
            all_candidate_set_ids,
            family_by_set,
            reference_set_size=self.universe.size_for_set(current_set_id),
        )
        self._apply_candidates(candidate_set_ids, grouped)

    def _select_candidate_set_ids(self, all_candidate_set_ids, family_by_set, reference_set_size=None):
        """Select visible next sets without using target or reward information.

        Galaxy exposes a compact list of operator result sets at each step. The
        fixed Covertype graph can expose many more next sets, so we keep the same
        compact interface while preserving operator-family and set-size diversity.
        """
        if len(all_candidate_set_ids) <= self.candidate_slots:
            return list(all_candidate_set_ids)

        selected = []
        selected_set = set()

        def add(set_id):
            set_id = int(set_id)
            if set_id in selected_set:
                return False
            if len(selected) >= self.candidate_slots:
                return False
            selected.append(set_id)
            selected_set.add(set_id)
            return True

        reference_set_size = (
            self.base_env.previous_set_size
            if reference_set_size is None
            else int(reference_set_size)
        )
        for family in ("by_facet", "by_superset", "by_neighbors", "by_distribution"):
            family_candidates = [
                int(set_id)
                for set_id in all_candidate_set_ids
                if family in family_by_set.get(int(set_id), set())
            ]
            if not family_candidates:
                continue
            family_candidates.sort(
                key=lambda set_id: (
                    abs(np.log1p(float(self.universe.size_for_set(set_id))) - np.log1p(float(reference_set_size))),
                    int(set_id),
                )
            )
            add(family_candidates[0])

        remaining = [int(set_id) for set_id in all_candidate_set_ids if int(set_id) not in selected_set]
        if not remaining:
            return selected

        remaining.sort(key=lambda set_id: (float(self.universe.size_for_set(set_id)), int(set_id)))
        quantile_indices = np.linspace(0, len(remaining) - 1, max(1, self.candidate_slots - len(selected)))
        for index in quantile_indices.round().astype(int).tolist():
            add(remaining[int(index)])

        if len(selected) < self.candidate_slots:
            selected_states = [self.universe.state_for_set(set_id) for set_id in selected]
            leftovers = [set_id for set_id in remaining if set_id not in selected_set]
            while leftovers and len(selected) < self.candidate_slots:
                if not selected_states:
                    add(leftovers.pop(0))
                    selected_states.append(self.universe.state_for_set(selected[-1]))
                    continue
                best_pos = 0
                best_distance = -1.0
                for pos, set_id in enumerate(leftovers):
                    state = self.universe.state_for_set(set_id)
                    distance = min(float(np.linalg.norm(state - existing)) for existing in selected_states)
                    if distance > best_distance:
                        best_distance = distance
                        best_pos = pos
                chosen = leftovers.pop(best_pos)
                if add(chosen):
                    selected_states.append(self.universe.state_for_set(chosen))

        return selected

    def _default_context_state_for_set(self, set_id):
        set_id = int(set_id)
        all_candidate_set_ids, _grouped, family_by_set = self._candidate_groups_for_set(set_id)
        candidate_set_ids = self._select_candidate_set_ids(
            all_candidate_set_ids,
            family_by_set,
            reference_set_size=self.universe.size_for_set(set_id),
        )
        while len(candidate_set_ids) < self.candidate_slots:
            candidate_set_ids.append(-1)
        return self._build_set_state(candidate_set_ids[: self.candidate_slots])

    def rerank_candidates_by_bile(self, bile, z, family_keep=4):
        """Rerank visible next sets by alignment with the current BILE direction.

        This does not use target membership or reward. It only asks whether the
        latent transition from the current candidate context to each possible
        next candidate context follows the episode-level BILE direction z.
        """
        if bile is None or z is None:
            return self.set_state

        current_set_id = int(self.base_env.current_set_id)
        all_candidate_set_ids, grouped, family_by_set = self._candidate_groups_for_set(current_set_id)
        if not all_candidate_set_ids:
            return self.set_state

        z = np.asarray(z, dtype=np.float32).reshape(-1)
        z_norm = float(np.linalg.norm(z))
        if z_norm <= 1e-8:
            return self.set_state
        z = (z / z_norm).astype(np.float32)

        try:
            current_context = np.asarray(self.set_state, dtype=np.float32).reshape(1, -1)
            next_contexts = np.asarray(
                [self._default_context_state_for_set(set_id) for set_id in all_candidate_set_ids],
                dtype=np.float32,
            )
            phi = bile.embed(np.concatenate([current_context, next_contexts], axis=0))
        except Exception:
            return self.set_state

        phi_current = np.asarray(phi[0], dtype=np.float32)
        phi_next = np.asarray(phi[1:], dtype=np.float32)
        deltas = phi_next - phi_current.reshape(1, -1)
        norms = np.linalg.norm(deltas, axis=1)
        valid = norms > 1e-8
        scores = np.full(len(all_candidate_set_ids), -np.inf, dtype=np.float64)
        if np.any(valid):
            scores[valid] = np.dot(deltas[valid] / norms[valid, None], z)

        score_by_set = {
            int(set_id): float(score)
            for set_id, score in zip(all_candidate_set_ids, scores)
        }
        selected = []
        selected_set = set()

        def add(set_id):
            set_id = int(set_id)
            if set_id in selected_set:
                return False
            if len(selected) >= self.candidate_slots:
                return False
            selected.append(set_id)
            selected_set.add(set_id)
            return True

        family_keep = max(0, int(family_keep))
        if family_keep > 0:
            for family in ("by_facet", "by_superset", "by_neighbors", "by_distribution")[:family_keep]:
                family_candidates = [
                    int(set_id)
                    for set_id in all_candidate_set_ids
                    if family in family_by_set.get(int(set_id), set())
                ]
                if family_candidates:
                    family_candidates.sort(
                        key=lambda set_id: (score_by_set.get(int(set_id), -np.inf), -int(set_id)),
                        reverse=True,
                    )
                    add(family_candidates[0])

        ranked = sorted(
            [int(set_id) for set_id in all_candidate_set_ids if int(set_id) not in selected_set],
            key=lambda set_id: (score_by_set.get(int(set_id), -np.inf), -int(set_id)),
            reverse=True,
        )
        for set_id in ranked:
            if not add(set_id):
                break

        selected.sort(key=lambda set_id: (score_by_set.get(int(set_id), -np.inf), -int(set_id)), reverse=True)
        self._apply_candidates(selected, grouped, score_by_set=score_by_set)
        return self.set_state

    def bile_candidate_prior_probs(self, temperature=0.25):
        scores = np.asarray(self.candidate_bile_scores, dtype=np.float64)
        mask = np.asarray([1.0 if int(set_id) >= 0 else 0.0 for set_id in self.candidate_set_ids], dtype=np.float64)
        if scores.size != mask.size or mask.sum() <= 0.0:
            return _masked_probs(np.ones_like(mask, dtype=np.float64), mask)
        finite = np.isfinite(scores) & (mask > 0.0)
        if not np.any(finite):
            return _masked_probs(np.ones_like(mask, dtype=np.float64), mask)
        shifted = np.full_like(scores, -np.inf, dtype=np.float64)
        valid_scores = scores[finite]
        temperature = max(float(temperature), 1e-6)
        shifted[finite] = (valid_scores - float(np.max(valid_scores))) / temperature
        exp_scores = np.zeros_like(scores, dtype=np.float64)
        exp_scores[finite] = np.exp(np.clip(shifted[finite], -60.0, 60.0))
        return _masked_probs(exp_scores, mask)

    def fix_possible_set_action_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        mask = np.asarray([1.0 if int(set_id) >= 0 else 0.0 for set_id in self.candidate_set_ids], dtype=np.float64)
        return _masked_probs(probs, mask)

    def get_operation_state(self, set_action):
        set_action = int(set_action)
        if 0 <= set_action < len(self.candidate_set_ids) and int(self.candidate_set_ids[set_action]) >= 0:
            candidate_state = self.universe.state_for_set(int(self.candidate_set_ids[set_action]))
        else:
            candidate_state = np.zeros(self.universe.state_dim, dtype=np.float32)
        return np.concatenate([self.set_state, candidate_state]).astype(np.float32)

    def fix_possible_operation_action_probs(self, set_action, probs):
        probs = np.asarray(probs, dtype=np.float64)
        mask = np.zeros(self.operation_action_dim, dtype=np.float64)
        if 0 <= int(set_action) < len(self.candidate_action_ids):
            actions = self.candidate_action_ids[int(set_action)]
            if actions:
                mask[np.asarray(actions, dtype=np.int64)] = 1.0
        if mask.sum() <= 0.0:
            valid = self.base_env.valid_actions()
            mask[np.asarray(valid, dtype=np.int64)] = 1.0
        return _masked_probs(probs, mask)

    def step(self, set_action, operation_action):
        self.step_count += 1
        set_action = int(set_action)
        operation_action = int(operation_action)
        allowed = self.candidate_action_ids[set_action] if 0 <= set_action < len(self.candidate_action_ids) else []
        if operation_action not in allowed:
            operation_action = int(allowed[0]) if allowed else int(self.base_env.valid_actions()[0])

        previous_set_id = int(self.base_env.current_set_id)
        next_state, metrics, next_set_id, valid = self.base_env.step(operation_action)
        self.sets_viewed.add(int(next_set_id))
        self.visited_set_state_rows[int(next_set_id)] = np.asarray(next_state, dtype=np.float32).astype(float).tolist()

        action = self.actions[operation_action]
        self.exploration_trace_rows.append(
            {
                "step": int(self.step_count),
                "set_id": int(next_set_id),
                "step_extrinsic_reward": float(metrics["extrinsic_reward"]),
                "step_interestingness": float(metrics["interestingness"]),
                "operator": str(action.family),
                "parameter": str(action.label),
                "input_set_id": int(previous_set_id),
                "operation_action": int(operation_action),
                "set_action": int(set_action),
                "set_size": int(metrics["set_size"]),
                "valid": int(valid),
            }
        )

        done = self.step_count >= self.episode_steps
        self._refresh_candidates()
        return self.set_state, float(metrics["extrinsic_reward"]), float(metrics["interestingness"]), done, f"{previous_set_id}-{operation_action}", metrics

    def consume_exploration_logs(self):
        trace_rows = list(self.exploration_trace_rows)
        state_rows = [
            {"set_id": int(set_id), "state": state}
            for set_id, state in self.visited_set_state_rows.items()
        ]
        return trace_rows, state_rows


@ray.remote
class ParameterServer:
    def __init__(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim, args_dict):
        _configure_tf()
        self.args = SimpleNamespace(**args_dict)
        self.episodes_reserved = int(getattr(self.args, "resume_start_episode", 0) or 0)
        self.episodes_done = int(getattr(self.args, "resume_start_episode", 0) or 0)
        self.global_sets_viewed = set()
        self.cumulative_extrinsic_reward = 0.0
        self.logged_state_ids = set()
        self.is_bile = self.args.baseline in MIRA_BASELINES
        self.bile_success_states = []
        self.bile_success_next_states = []
        self.bile_success_scores = []
        self.bile_success_kinds = []
        self.bile_success_episodes = []
        self.bile_buffer_states = None
        self.bile_buffer_next_states = None
        self.bile_buffer_set_actions = None
        self.bile_buffer_operation_actions = None
        self.bile_buffer_rewards_ext = None
        self.bile_buffer_size = 0
        self.bile_buffer_pos = 0
        self.bile_rng = np.random.default_rng(int(self.args.seed) + 20260619)
        self.recent_extrinsic_rewards = []
        self.z_sample_count = 0

        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = self.args.output_prefix or f"{self.args.baseline}_seed{self.args.seed}_full_a3c"
        self.run_dir = self.output_dir / self.prefix
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rewards_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_rewards.csv"
        self.trace_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_exploration_trace.csv"
        self.states_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_visited_set_states.csv"
        self.config_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_config.json"
        self.model_dir = self.run_dir / f"{self.prefix}_{self.args.baseline}_models"

        critic_units = _parse_units(self.args.critic_dense_units)
        resume_model_dir = Path(self.args.resume_model_dir) if getattr(self.args, "resume_model_dir", None) else None
        set_model_path = str(resume_model_dir / "set_actor") if resume_model_dir else None
        op_model_path = str(resume_model_dir / "operation_actor") if resume_model_dir else None
        critic_model_path = str(resume_model_dir / "critic") if resume_model_dir else None

        self.global_set_actor = SetActor(
            set_state_dim,
            set_action_dim,
            self.args.lstm_steps,
            self.args.actor_lr,
            self.args.name,
            model_path=set_model_path,
            dense_units=self.args.set_dense_units,
            lstm_units=self.args.set_lstm_units,
        )
        self.global_operation_actor = OperationActor(
            operation_state_dim,
            operation_action_dim,
            self.args.lstm_steps,
            self.args.actor_lr,
            self.args.name,
            model_path=op_model_path,
            dense_units=self.args.op_dense_units,
            lstm_units=self.args.op_lstm_units,
        )
        self.global_critic = Critic(
            set_state_dim,
            self.args.lstm_steps,
            self.args.critic_lr,
            self.args.name,
            model_path=critic_model_path,
            dense_units=critic_units,
            lstm_units=self.args.critic_lstm_units,
        )
        self.global_bile = None
        if self.is_bile:
            self.global_bile = BILEModule(
                state_dim=self.args.base_set_state_dim,
                set_action_dim=set_action_dim,
                operation_action_dim=operation_action_dim,
                latent_dim=self.args.bile_latent_dim,
                lr=self.args.bile_lr,
                dense_units=self.args.bile_dense_units,
                target_tau=self.args.bile_target_tau,
            )
        self.orthogonal_directions = self._build_orthogonal_directions(
            max(int(self.args.workers) * 2, 16),
            int(self.args.bile_latent_dim),
        )

        self._write_config(set_state_dim, operation_state_dim, set_action_dim, operation_action_dim)
        self._init_csv_files()

    def _write_config(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim):
        config = {
            **vars(self.args),
            "runner": "full_a3c_ray_dual_actor",
            "set_state_dim": int(set_state_dim),
            "operation_state_dim": int(operation_state_dim),
            "set_action_dim": int(set_action_dim),
            "operation_action_dim": int(operation_action_dim),
            "base_set_state_dim": int(getattr(self.args, "base_set_state_dim", set_state_dim)),
            "base_operation_state_dim": int(getattr(self.args, "base_operation_state_dim", operation_state_dim)),
            "note": "Covertype fixed-set universe with Galaxy-style set actor, operation actor, critic, optional BILE encoder/dynamics, and Ray parameter server.",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _init_csv_files(self):
        with open(self.rewards_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "extrinsic_reward",
                    "mean_step_extrinsic",
                    "target_hits",
                    "interestingness",
                    "familiarity",
                    "counter_curiosity",
                    "coherency",
                    "diversity",
                    "total_reward",
                    "sets_viewed",
                    "cumulative_unique_sets_viewed",
                    "target_efficiency",
                    "cumulative_extrinsic_reward",
                    "cumulative_target_efficiency",
                    "valid_steps",
                    "bile_bonus",
                    "bootstrap_diversity",
                    "exploration_reward",
                    "bile_phi_loss",
                    "bile_dyn_loss",
                    "bile_prediction_error",
                    "bile_mean_d_phi",
                    "bile_mean_metric_target",
                    "bile_z_source",
                    "bile_zpool_size",
                    "recent_success_ratio",
                    "recent_low_reward_ratio",
                    "bootstrap_active",
                    "escape_active",
                    "bile_pair_source",
                    "bile_replay_size",
                ]
            )
        with open(self.trace_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "agent_id",
                    "step",
                    "set_id",
                    "step_extrinsic_reward",
                    "step_interestingness",
                    "operator",
                    "parameter",
                    "input_set_id",
                    "set_action",
                    "operation_action",
                    "step_coherency",
                    "step_diversity",
                    "step_total_reward",
                    "set_size",
                    "valid",
                    "step_bile_bonus",
                    "step_bootstrap_diversity",
                    "bootstrap_active",
                    "escape_active",
                    "bile_z_source",
                ]
            )

    def next_episode(self):
        if self.episodes_reserved >= int(self.args.episodes):
            return None
        self.episodes_reserved += 1
        return self.episodes_reserved

    def get_weights(self):
        return {
            "set": self.global_set_actor.model.get_weights(),
            "op": self.global_operation_actor.model.get_weights(),
            "critic": self.global_critic.model.get_weights(),
            "bile": self.global_bile.get_weights() if self.global_bile is not None else None,
        }

    def apply_gradients_and_get_weights(self, set_grads, op_grads, critic_grads, bile_encoder_grads=None, bile_dynamics_grads=None):
        if set_grads:
            self.global_set_actor.apply_gradients(set_grads)
        if op_grads:
            self.global_operation_actor.apply_gradients(op_grads)
        if critic_grads:
            self.global_critic.apply_gradients(critic_grads)
        if self.global_bile is not None:
            self.global_bile.apply_gradients(bile_encoder_grads, bile_dynamics_grads)
        return self.get_weights()

    def _normalize_direction(self, direction):
        return normalize_direction(direction, int(self.args.bile_latent_dim))

    def _build_orthogonal_directions(self, count, dim):
        dim = int(dim)
        count = int(count)
        random_matrix = self.bile_rng.normal(size=(dim, min(count, dim))).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)
        directions = [q[:, idx].astype(np.float32) for idx in range(q.shape[1])]
        while len(directions) < count:
            directions.append(self._normalize_direction(self.bile_rng.normal(size=dim)))
        return directions

    def _sample_random_direction(self):
        return sample_direction(self.bile_rng, int(self.args.bile_latent_dim))

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
        threshold = float(self.args.bootstrap_success_threshold)
        return float(np.mean([1.0 if value >= threshold else 0.0 for value in recent]))

    def _recent_low_reward_ratio(self, window=None):
        window = max(1, int(window if window is not None else self.args.bootstrap_window))
        if not self.recent_extrinsic_rewards:
            return 0.0
        recent = self.recent_extrinsic_rewards[-window:]
        threshold = float(self.args.escape_low_reward_threshold)
        return float(np.mean([1.0 if value < threshold else 0.0 for value in recent]))

    def _bootstrap_active(self):
        if self.args.baseline != "mira":
            return False
        zpool_small = len(self.bile_success_states) < int(self.args.bootstrap_zpool_threshold)
        recent_low = (
            self._recent_extrinsic_average() < float(self.args.bootstrap_ext_threshold)
            or self._recent_low_reward_ratio(self.args.bootstrap_window) >= float(self.args.escape_low_ratio_threshold)
        )
        recent_weak = (
            self._recent_success_ratio() < float(self.args.bootstrap_success_ratio_threshold)
            if bool(self.args.bootstrap_use_success_ratio)
            else False
        )
        return bool(zpool_small or recent_low or recent_weak)

    def _escape_active(self):
        if self.args.baseline != "mira":
            return False
        window = max(1, int(self.args.escape_window))
        if len(self.recent_extrinsic_rewards) < window:
            return False
        recent = self.recent_extrinsic_rewards[-window:]
        recent_avg_low = float(np.mean(recent)) < float(self.args.escape_ext_threshold)
        recent_low_ratio = self._recent_low_reward_ratio(window) >= float(self.args.escape_low_ratio_threshold)
        return bool(recent_avg_low or recent_low_ratio)

    def _sample_success_direction(self):
        if not self.bile_success_states:
            return None
        scores = np.asarray(self.bile_success_scores, dtype=np.float64)
        score_clip = float(self.args.bile_success_score_clip)
        if score_clip > 0.0:
            scores = np.minimum(scores, score_clip)
        weights = np.power(np.maximum(scores, 0.0), float(self.args.bile_success_weight_power)) + 1e-6
        weights = weights / max(float(weights.sum()), 1e-12)
        idx = int(self.bile_rng.choice(len(self.bile_success_states), p=weights))
        states = np.asarray(
            [
                self.bile_success_states[idx],
                self.bile_success_next_states[idx],
            ],
            dtype=np.float32,
        ).reshape((2, int(self.args.base_set_state_dim)))
        phi_pair = self.global_bile.embed(states)
        base_direction = phi_pair[1] - phi_pair[0]
        noise = self._sample_random_direction()
        return self._normalize_direction(
            base_direction + (float(self.args.bile_success_noise_scale) * noise)
        )

    def sample_bile_direction(self):
        if not self.is_bile:
            return {
                "z": [],
                "z_source": "none",
                "zpool_size": 0,
                "bootstrap_active": False,
                "escape_active": False,
                "recent_ext_avg": 0.0,
                "recent_success_ratio": 0.0,
                "recent_low_reward_ratio": 0.0,
            }

        if self.args.baseline != "mira":
            z = self._sample_random_direction()
            return {
                "z": z.astype(np.float32).tolist(),
                "z_source": "random",
                "zpool_size": 0,
                "bootstrap_active": False,
                "escape_active": False,
                "recent_ext_avg": 0.0,
                "recent_success_ratio": 0.0,
                "recent_low_reward_ratio": 0.0,
            }

        self.z_sample_count += 1
        escape_active = self._escape_active()
        bootstrap_active = self._bootstrap_active() or escape_active

        if escape_active:
            success_prob = max(0.0, min(1.0, float(self.args.escape_success_prob)))
            orthogonal_prob = max(0.0, min(1.0, float(self.args.escape_orthogonal_prob)))
            draw = float(self.bile_rng.random())
            success_z = self._sample_success_direction() if self.bile_success_states else None
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
        elif bootstrap_active:
            if float(self.bile_rng.random()) < 0.80:
                index = self.z_sample_count % len(self.orthogonal_directions)
                z = self.orthogonal_directions[index]
                z_source = "bootstrap_orthogonal"
            else:
                z = self._sample_random_direction()
                z_source = "bootstrap_random"
        else:
            success_prob = max(0.0, min(1.0, float(self.args.bile_success_prob)))
            orthogonal_prob = max(0.0, min(1.0, float(self.args.bile_orthogonal_prob)))
            draw = float(self.bile_rng.random())
            if self.bile_success_states and draw < success_prob:
                z = self._sample_success_direction()
                z_source = "success"
            elif draw < success_prob + orthogonal_prob:
                index = self.z_sample_count % len(self.orthogonal_directions)
                z = self.orthogonal_directions[index]
                z_source = "orthogonal"
            else:
                z = self._sample_random_direction()
                z_source = "random"

        return {
            "z": z.astype(np.float32).tolist(),
            "z_source": z_source,
            "zpool_size": len(self.bile_success_states),
            "bootstrap_active": bool(bootstrap_active),
            "escape_active": bool(escape_active),
            "recent_ext_avg": self._recent_extrinsic_average(),
            "recent_success_ratio": self._recent_success_ratio(),
            "recent_low_reward_ratio": self._recent_low_reward_ratio(
                self.args.escape_window if escape_active else self.args.bootstrap_window
            ),
        }

    def _success_sampling_scores(self):
        scores = np.asarray(self.bile_success_scores, dtype=np.float64)
        score_clip = float(self.args.bile_success_score_clip)
        if score_clip > 0.0:
            scores = np.minimum(scores, score_clip)
        return scores

    def update_and_sample_bile_batch(
        self,
        states,
        set_actions,
        operation_actions,
        rewards_ext,
        next_states,
        episode,
    ):
        if not self.is_bile:
            return {"source": "none", "buffer_size": 0}

        stored = self._store_bile_transitions(
            states,
            set_actions,
            operation_actions,
            rewards_ext,
            next_states,
        )
        if stored <= 0:
            return {"source": "local_empty", "buffer_size": int(self.bile_buffer_size)}

        warmup_episodes = int(getattr(self.args, "bile_pair_warmup_episodes", 100))
        if int(episode) <= warmup_episodes:
            return {"source": "local_warmup", "buffer_size": int(self.bile_buffer_size)}

        min_replay_size = int(getattr(self.args, "bile_min_replay_size", 256))
        if self.bile_buffer_size < max(2, min_replay_size):
            return {"source": "local_small_buffer", "buffer_size": int(self.bile_buffer_size)}

        batch_size = int(getattr(self.args, "bile_phi_batch_size", 128))
        batch_size = min(max(2, batch_size), int(self.bile_buffer_size))
        indices = self.bile_rng.choice(int(self.bile_buffer_size), size=batch_size, replace=False)
        pair_indices = self.bile_rng.permutation(batch_size).astype(np.int64)
        if batch_size > 1 and np.any(pair_indices == np.arange(batch_size, dtype=np.int64)):
            pair_indices = np.roll(pair_indices, 1)

        return {
            "source": "replay_random_perm",
            "buffer_size": int(self.bile_buffer_size),
            "states": self.bile_buffer_states[indices].astype(np.float32),
            "set_actions": self.bile_buffer_set_actions[indices].astype(np.int64),
            "operation_actions": self.bile_buffer_operation_actions[indices].astype(np.int64),
            "rewards_ext": self.bile_buffer_rewards_ext[indices].astype(np.float32),
            "next_states": self.bile_buffer_next_states[indices].astype(np.float32),
            "pair_indices": pair_indices,
        }

    def _store_bile_transitions(self, states, set_actions, operation_actions, rewards_ext, next_states):
        max_size = int(getattr(self.args, "bile_replay_buffer_size", 50000))
        if max_size <= 0:
            return 0

        state_dim = int(self.args.base_set_state_dim)
        states_np = np.asarray(states, dtype=np.float32).reshape((-1, state_dim))
        next_states_np = np.asarray(next_states, dtype=np.float32).reshape((-1, state_dim))
        set_actions_np = np.asarray(set_actions, dtype=np.int64).reshape(-1)
        operation_actions_np = np.asarray(operation_actions, dtype=np.int64).reshape(-1)
        rewards_np = np.asarray(rewards_ext, dtype=np.float32).reshape(-1, 1)
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

    def update_bile_success_transitions(self, states, next_states, scores, kinds=None, episode=None):
        if self.args.baseline != "mira":
            return len(self.bile_success_states)
        max_size = int(self.args.bile_success_pool_size)
        if max_size <= 0:
            self.bile_success_states = []
            self.bile_success_next_states = []
            self.bile_success_scores = []
            self.bile_success_kinds = []
            self.bile_success_episodes = []
            return 0
        min_score = float(getattr(self.args, "bile_success_quality_min_score", self.args.bile_min_success_reward))
        score_clip = float(self.args.bile_success_score_clip)
        state_dim = int(self.args.base_set_state_dim)
        episode_id = int(episode if episode is not None else self.episodes_done)
        states = list(states or [])
        next_states = list(next_states or [])
        scores = list(scores or [])
        kinds = list(kinds or ["local"] * len(states))
        if len(kinds) < len(states):
            kinds.extend(["local"] * (len(states) - len(kinds)))
        for state, next_state, score, kind in zip(states, next_states, scores, kinds):
            score = float(score)
            if score <= min_score:
                continue
            if score_clip > 0.0:
                score = min(score, score_clip)
            self.bile_success_states.append(np.asarray(state, dtype=np.float32).reshape(state_dim))
            self.bile_success_next_states.append(np.asarray(next_state, dtype=np.float32).reshape(state_dim))
            self.bile_success_scores.append(score)
            self.bile_success_kinds.append(str(kind or "local"))
            self.bile_success_episodes.append(episode_id)
        if len(self.bile_success_states) > max_size:
            keep = np.argsort(np.asarray(self.bile_success_scores, dtype=np.float64))[::-1][:max_size]
            self.bile_success_states = [self.bile_success_states[int(idx)] for idx in keep]
            self.bile_success_next_states = [self.bile_success_next_states[int(idx)] for idx in keep]
            self.bile_success_scores = [self.bile_success_scores[int(idx)] for idx in keep]
            self.bile_success_kinds = [self.bile_success_kinds[int(idx)] for idx in keep]
            self.bile_success_episodes = [self.bile_success_episodes[int(idx)] for idx in keep]
        return len(self.bile_success_states)

    def record_episode(self, episode, totals, episode_set_ids, trace_rows, state_rows):
        self.episodes_done += 1
        completed_episode = int(self.episodes_done)
        episode_sets = {int(set_id) for set_id in episode_set_ids if int(set_id) >= 0}
        self.global_sets_viewed.update(episode_sets)
        self.cumulative_extrinsic_reward += float(totals["extrinsic_reward"])
        self.recent_extrinsic_rewards.append(float(totals["extrinsic_reward"]))
        max_recent = max(1, max(int(self.args.bootstrap_window), int(self.args.escape_window)) * 4)
        if len(self.recent_extrinsic_rewards) > max_recent:
            self.recent_extrinsic_rewards = self.recent_extrinsic_rewards[-max_recent:]
        cumulative_unique = len(self.global_sets_viewed)
        sets_viewed = len(episode_sets)
        cumulative_efficiency = self.cumulative_extrinsic_reward / max(cumulative_unique, 1)

        with open(self.rewards_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    completed_episode,
                    float(totals["extrinsic_reward"]),
                    float(totals["extrinsic_reward"]) / max(int(self.args.steps), 1),
                    int(totals["target_hits"]),
                    float(totals["interestingness"]),
                    float(totals["familiarity"]),
                    float(totals["counter_curiosity"]),
                    float(totals["coherency"]),
                    float(totals["diversity"]),
                    float(totals["total_reward"]),
                    int(sets_viewed),
                    int(cumulative_unique),
                    float(totals["extrinsic_reward"]) / max(sets_viewed, 1),
                    float(self.cumulative_extrinsic_reward),
                    float(cumulative_efficiency),
                    int(totals["valid_steps"]),
                    float(totals.get("bile_bonus", 0.0)),
                    float(totals.get("bootstrap_diversity", 0.0)),
                    float(totals.get("exploration_reward", 0.0)),
                    float(totals.get("bile_phi_loss", 0.0)),
                    float(totals.get("bile_dyn_loss", 0.0)),
                    float(totals.get("bile_prediction_error", 0.0)),
                    float(totals.get("bile_mean_d_phi", 0.0)),
                    float(totals.get("bile_mean_metric_target", 0.0)),
                    totals.get("bile_z_source", ""),
                    int(totals.get("bile_zpool_size", 0)),
                    float(self._recent_success_ratio()),
                    float(self._recent_low_reward_ratio(self.args.escape_window)),
                    int(totals.get("bootstrap_active", 0)),
                    int(totals.get("escape_active", 0)),
                    totals.get("bile_pair_source", ""),
                    int(totals.get("bile_replay_size", 0)),
                ]
            )

        with open(self.trace_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in trace_rows:
                writer.writerow(
                    [
                        completed_episode,
                        int(row.get("agent_id", -1)),
                        int(row.get("step", -1)),
                        int(row.get("set_id", -1)),
                        float(row.get("step_extrinsic_reward", 0.0)),
                        float(row.get("step_interestingness", 0.0)),
                        row.get("operator", ""),
                        row.get("parameter", ""),
                        int(row.get("input_set_id", -1)),
                        int(row.get("set_action", -1)),
                        int(row.get("operation_action", -1)),
                        float(row.get("step_coherency", 0.0)),
                        float(row.get("step_diversity", 0.0)),
                        float(row.get("step_total_reward", 0.0)),
                        int(row.get("set_size", 0)),
                        int(row.get("valid", 0)),
                        float(row.get("step_bile_bonus", 0.0)),
                        float(row.get("step_bootstrap_diversity", 0.0)),
                        int(row.get("bootstrap_active", 0)),
                        int(row.get("escape_active", 0)),
                        row.get("bile_z_source", ""),
                    ]
                )

        new_state_rows = []
        for row in state_rows:
            set_id = int(row.get("set_id", -1))
            if set_id < 0 or set_id in self.logged_state_ids:
                continue
            state = row.get("state", [])
            if not state:
                continue
            self.logged_state_ids.add(set_id)
            new_state_rows.append((set_id, state))
        if new_state_rows:
            file_exists = self.states_path.exists()
            with open(self.states_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    state_dim = len(new_state_rows[0][1])
                    writer.writerow(["set_id"] + [f"state_{idx}" for idx in range(state_dim)])
                for set_id, state in new_state_rows:
                    writer.writerow([set_id] + list(state))

        if completed_episode % int(self.args.save_interval) == 0:
            self.save_models(f"episode_{completed_episode}")
        return {
            "episode": completed_episode,
            "reserved_episode": int(episode),
            "episodes_done": completed_episode,
            "sets_viewed": int(sets_viewed),
            "cumulative_unique_sets_viewed": int(cumulative_unique),
            "cumulative_target_efficiency": float(cumulative_efficiency),
        }

    def save_models(self, label="final"):
        target = self.model_dir / str(label)
        self.global_set_actor.save_model(target / "set_actor")
        self.global_operation_actor.save_model(target / "operation_actor")
        self.global_critic.save_model(target / "critic")
        if self.global_bile is not None:
            self.global_bile.save_models(target / "bile")
        return str(target)

    def output_paths(self):
        return {
            "rewards": str(self.rewards_path),
            "trace": str(self.trace_path),
            "states": str(self.states_path),
            "config": str(self.config_path),
            "models": str(self.model_dir),
        }


@ray.remote
class WorkerAgent:
    def __init__(self, universe_dir, action_labels, ps_handle, agent_id, args_dict):
        _configure_tf()
        self.agent_id = int(agent_id)
        self.ps = ps_handle
        self.args = SimpleNamespace(**args_dict)
        self.universe = load_fixed_universe(universe_dir)
        self.actions = _actions_from_labels(action_labels)
        self.env = CovertypeDualActorEnvironment(
            self.universe,
            self.actions,
            episode_steps=self.args.steps,
            seed=int(self.args.seed) * 1000 + self.agent_id,
            candidate_slots=self.args.candidate_slots,
        )
        self.is_bile = self.args.baseline in MIRA_BASELINES
        self.base_set_state_dim = self.env.set_state_dim
        self.base_operation_state_dim = self.env.operation_state_dim
        self.set_state_dim = self.base_set_state_dim + (int(self.args.bile_latent_dim) if self.is_bile else 0)
        self.operation_state_dim = self.base_operation_state_dim + (int(self.args.bile_latent_dim) if self.is_bile else 0)
        self.set_action_dim = self.env.set_action_dim
        self.operation_action_dim = self.env.operation_action_dim
        self.steps = int(self.args.lstm_steps)
        critic_units = _parse_units(self.args.critic_dense_units)
        self.bile = None
        if self.is_bile:
            self.bile = BILEModule(
                state_dim=self.base_set_state_dim,
                set_action_dim=self.set_action_dim,
                operation_action_dim=self.operation_action_dim,
                latent_dim=self.args.bile_latent_dim,
                lr=self.args.bile_lr,
                dense_units=self.args.bile_dense_units,
                target_tau=self.args.bile_target_tau,
            )
        self.set_actor = SetActor(
            self.set_state_dim,
            self.set_action_dim,
            self.steps,
            self.args.actor_lr,
            self.args.name,
            dense_units=self.args.set_dense_units,
            lstm_units=self.args.set_lstm_units,
        )
        self.operation_actor = OperationActor(
            self.operation_state_dim,
            self.operation_action_dim,
            self.steps,
            self.args.actor_lr,
            self.args.name,
            dense_units=self.args.op_dense_units,
            lstm_units=self.args.op_lstm_units,
        )
        self.critic = Critic(
            self.set_state_dim,
            self.steps,
            self.args.critic_lr,
            self.args.name,
            dense_units=critic_units,
            lstm_units=self.args.critic_lstm_units,
        )
        self.set_op_counters = {}
        self.bootstrap_visit_counts = {}
        self.sync_with_ps()

    def sync_with_ps(self):
        weights = ray.get(self.ps.get_weights.remote())
        self.set_actor.model.set_weights(weights["set"])
        self.operation_actor.model.set_weights(weights["op"])
        self.critic.model.set_weights(weights["critic"])
        if self.bile is not None and weights.get("bile") is not None:
            self.bile.set_weights(weights["bile"])

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards, dtype=np.float64)
        cumulative = 0.0 if done else float(np.squeeze(next_v_value))
        for idx in reversed(range(len(rewards))):
            cumulative = float(self.args.gamma) * cumulative + float(rewards[idx])
            td_targets[idx] = cumulative
        return td_targets

    def list_to_batch(self, items):
        return np.concatenate(items, axis=0)

    def augment_set_state(self, set_state, z):
        if not self.is_bile:
            return np.asarray(set_state, dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(set_state, dtype=np.float32).reshape(-1),
                np.asarray(z, dtype=np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    def augment_operation_state(self, operation_state, z):
        if not self.is_bile:
            return np.asarray(operation_state, dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(operation_state, dtype=np.float32).reshape(-1),
                np.asarray(z, dtype=np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    def sample_bile_direction(self):
        sample = ray.get(self.ps.sample_bile_direction.remote())
        return (
            np.asarray(sample["z"], dtype=np.float32),
            bool(sample.get("bootstrap_active", False)),
            bool(sample.get("escape_active", False)),
            int(sample.get("zpool_size", 0)),
            float(sample.get("recent_ext_avg", 0.0)),
            float(sample.get("recent_success_ratio", 0.0)),
            float(sample.get("recent_low_reward_ratio", 0.0)),
            str(sample.get("z_source", "random")),
        )

    def refresh_bile_candidates(self, bile_z):
        if self.args.baseline != "mira" or self.bile is None:
            return
        if bool(getattr(self.args, "disable_bile_candidate_rerank", False)):
            return
        self.env.rerank_candidates_by_bile(
            self.bile,
            bile_z,
            family_keep=getattr(self.args, "bile_candidate_family_keep", 4),
        )

    def apply_bile_candidate_prior(self, probs, zpool_size=0, z_source=""):
        if self.args.baseline != "mira":
            return probs
        if bool(getattr(self.args, "disable_bile_candidate_rerank", False)):
            return probs
        min_zpool = int(getattr(self.args, "bile_candidate_prior_min_zpool", 20))
        if int(zpool_size) < min_zpool:
            return probs
        weight = float(getattr(self.args, "bile_candidate_prior_weight", 0.25))
        if "success" in str(z_source):
            weight = float(getattr(self.args, "bile_candidate_prior_success_weight", 0.55))
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight <= 0.0:
            return probs
        prior = self.env.bile_candidate_prior_probs(
            temperature=getattr(self.args, "bile_candidate_prior_temperature", 0.18)
        )
        mixed = ((1.0 - weight) * np.asarray(probs, dtype=np.float64)) + (weight * prior)
        return _masked_probs(mixed, np.asarray([1.0 if int(set_id) >= 0 else 0.0 for set_id in self.env.candidate_set_ids], dtype=np.float64))

    def sample_action(self, probs, action_dim, fallback_action=0, random_prob=0.0):
        probs = np.asarray(probs, dtype=np.float64)
        valid_actions = np.flatnonzero(np.isfinite(probs) & (probs > 0.0))
        if valid_actions.size == 0:
            return int(fallback_action)
        if random_prob > 0.0 and np.random.random() < float(random_prob):
            return int(np.random.choice(valid_actions))
        safe_probs = np.zeros(int(action_dim), dtype=np.float64)
        copy_len = min(int(action_dim), probs.size)
        safe_probs[:copy_len] = np.nan_to_num(probs[:copy_len], nan=0.0, posinf=0.0, neginf=0.0)
        safe_probs[safe_probs < 0.0] = 0.0
        total = float(safe_probs.sum())
        if total <= 0.0:
            return int(fallback_action)
        safe_probs /= total
        return int(np.random.choice(int(action_dim), p=safe_probs))

    def embed_bile_state(self, state):
        if self.bile is None:
            return np.asarray(state, dtype=np.float32).reshape(-1)
        phi = self.bile.embed(np.asarray(state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
        return np.asarray(phi[0], dtype=np.float32).reshape(-1)

    def compute_bootstrap_diversity_reward(self, phi_s_next, episode_history_phi):
        state_hash = hash(np.asarray(phi_s_next, dtype=np.float32).tobytes())
        self.bootstrap_visit_counts[state_hash] = self.bootstrap_visit_counts.get(state_hash, 0) + 1
        count_reward = 1.0 / np.sqrt(float(self.bootstrap_visit_counts[state_hash]))
        if not episode_history_phi:
            distance_reward = 1.0
        else:
            distances = [np.linalg.norm(phi_s_next - history_phi) for history_phi in episode_history_phi]
            min_distance = float(np.min(distances)) if distances else 0.0
            distance_reward = float(1.0 - np.exp(-float(self.args.bootstrap_distance_eta) * min_distance))
        return float(0.5 * count_reward + 0.5 * distance_reward)

    def train_loop(self):
        while True:
            episode = ray.get(self.ps.next_episode.remote())
            if episode is None:
                break
            try:
                self._run_episode(int(episode))
            except Exception as error:
                print(error)
                traceback.print_tb(error.__traceback__)
                print(f"Worker {self.agent_id} episode {episode} failed; continuing.", flush=True)
        return True

    def _run_episode(self, episode):
        self.sync_with_ps()

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
        episode_transition_states = []
        episode_transition_next_states = []
        episode_transition_ext_rewards = []
        episode_trajectory_states = []
        episode_trajectory_next_states = []
        episode_trajectory_scores = []
        episode_set_op_counters = {}
        episode_history_phi = []
        if self.is_bile:
            (
                bile_z,
                bootstrap_active,
                escape_active,
                bile_zpool_size,
                _recent_ext_avg,
                recent_success_ratio,
                recent_low_reward_ratio,
                bile_z_source,
            ) = self.sample_bile_direction()
        else:
            bile_z = None
            bootstrap_active = False
            escape_active = False
            bile_z_source = "none"
            bile_zpool_size = 0
            recent_success_ratio = 0.0
            recent_low_reward_ratio = 0.0

        totals = _zero_totals()
        totals["bile_z_source"] = bile_z_source
        totals["bile_zpool_size"] = bile_zpool_size
        totals["bootstrap_active"] = int(bootstrap_active)
        totals["escape_active"] = int(escape_active)
        totals["recent_success_ratio"] = float(recent_success_ratio)
        totals["recent_low_reward_ratio"] = float(recent_low_reward_ratio)
        set_action_steps = [np.zeros(self.set_state_dim, dtype=np.float32) for _ in range(self.steps)]
        operation_action_steps = [np.zeros(self.operation_state_dim, dtype=np.float32) for _ in range(self.steps)]

        set_state = self.env.reset()
        self.refresh_bile_candidates(bile_z)
        set_state = self.env.set_state
        segment_anchor_state = np.asarray(set_state, dtype=np.float32).reshape(-1).tolist()
        segment_positive_reward = 0.0
        set_action_steps.pop(0)
        set_action_steps.append(self.augment_set_state(set_state, bile_z))
        if self.is_bile:
            episode_history_phi.append(self.embed_bile_state(set_state))
        done = False
        forced_random_action_prob = (
            float(self.args.escape_random_action_prob)
            if escape_active
            else (float(self.args.bootstrap_random_action_prob) if bootstrap_active else 0.0)
        )

        while not done:
            set_probs = self.set_actor.model.predict(
                np.asarray(set_action_steps, dtype=np.float32).reshape((1, self.steps, self.set_state_dim)),
                verbose=0,
            )[0]
            set_probs = self.env.fix_possible_set_action_probs(set_probs)
            set_probs = self.apply_bile_candidate_prior(
                set_probs,
                zpool_size=bile_zpool_size,
                z_source=bile_z_source,
            )
            set_action = self.sample_action(
                set_probs,
                self.set_action_dim,
                fallback_action=0,
                random_prob=forced_random_action_prob,
            )

            operation_state = self.augment_operation_state(self.env.get_operation_state(set_action), bile_z)
            operation_action_steps.pop(0)
            operation_action_steps.append(operation_state)
            op_probs = self.operation_actor.model.predict(
                np.asarray(operation_action_steps, dtype=np.float32).reshape((1, self.steps, self.operation_state_dim)),
                verbose=0,
            )[0]
            op_probs = self.env.fix_possible_operation_action_probs(set_action, op_probs)
            operation_action = self.sample_action(
                op_probs,
                self.operation_action_dim,
                fallback_action=0,
                random_prob=forced_random_action_prob,
            )

            next_set_state, r_ext, r_int, done, set_op_pair, metrics = self.env.step(set_action, operation_action)
            self.refresh_bile_candidates(bile_z)
            next_set_state = self.env.set_state
            r_ext = float(r_ext)
            r_int = float(r_int)
            phi_s_t = self.embed_bile_state(set_state) if self.is_bile else np.asarray(set_state, dtype=np.float32).flatten()
            phi_s_next = self.embed_bile_state(next_set_state) if self.is_bile else np.asarray(next_set_state, dtype=np.float32).flatten()
            r_coh = _coherency(phi_s_t, phi_s_next)
            r_div = _episode_diversity(phi_s_next, episode_history_phi, eta=self.args.diversity_eta)
            bile_bonus = 0.0
            bootstrap_bonus = 0.0
            if self.bile is not None:
                bile_bonus = self.bile.compute_bonus(
                    set_state,
                    next_set_state,
                    bile_z,
                    clip_value=self.args.bile_bonus_clip,
                )
                r_div = bile_bonus
                bootstrap_bonus = (
                    self.compute_bootstrap_diversity_reward(phi_s_next, episode_history_phi)
                    if bootstrap_active and self.args.baseline == "mira"
                    else 0.0
                )
                if escape_active:
                    bootstrap_bonus *= float(self.args.escape_bootstrap_scale)
                if self.args.baseline == "mira":
                    episode_transition_states.append(np.asarray(set_state, dtype=np.float32).reshape(-1).tolist())
                    episode_transition_next_states.append(np.asarray(next_set_state, dtype=np.float32).reshape(-1).tolist())
                    success_quality_score = 0.0
                    if r_ext > float(self.args.bile_min_success_reward):
                        success_quality_score = _success_quality_score(
                            r_ext,
                            metrics.get("set_size", 0),
                            self.args,
                        )
                    episode_transition_ext_rewards.append(float(success_quality_score))
                    if r_ext > float(self.args.bile_min_success_reward):
                        segment_positive_reward += max(float(r_ext), 0.0)
                        trajectory_score = max(
                            float(r_ext),
                            segment_positive_reward * float(self.args.bile_success_trajectory_score_scale),
                        )
                        trajectory_quality_score = _success_quality_score(
                            trajectory_score,
                            metrics.get("set_size", 0),
                            self.args,
                        )
                        episode_trajectory_states.append(list(segment_anchor_state))
                        episode_trajectory_next_states.append(np.asarray(next_set_state, dtype=np.float32).reshape(-1).tolist())
                        episode_trajectory_scores.append(float(trajectory_quality_score))
            episode_history_phi.append(phi_s_next)

            counter_curiosity = _counter_curiosity(
                set_op_pair,
                episode_set_op_counters,
                self.set_op_counters,
                episode_steps=self.args.steps,
            )
            exploration_reward = 0.0
            effective_alpha = float(self.args.alpha)
            if self.args.baseline == "mira":
                exploration_reward = (
                    float(self.args.w_int) * float(r_int)
                    + float(self.args.w_bile) * float(bile_bonus)
                    + float(self.args.w_bootstrap) * float(bootstrap_bonus)
                )
                effective_alpha = float(self.args.escape_alpha) if escape_active else float(self.args.alpha)
                step_reward = (float(self.args.w_ext) * float(r_ext)) + (effective_alpha * exploration_reward)
            else:
                step_reward = _training_reward(
                    self.args.baseline,
                    r_ext,
                    r_int,
                    counter_curiosity,
                    r_coh,
                    r_div,
                    bile_bonus,
                    self.args,
                )
            next_set_action_steps = list(set_action_steps)
            next_set_action_steps.pop(0)
            next_set_action_steps.append(self.augment_set_state(next_set_state, bile_z))

            reward_batch.append(np.reshape(step_reward, (1, 1)))
            set_state_batch.append(np.asarray(set_action_steps, dtype=np.float32).reshape((1, self.steps, self.set_state_dim)))
            set_action_batch.append(np.reshape(set_action, (1, 1)))
            operation_state_batch.append(np.asarray(operation_action_steps, dtype=np.float32).reshape((1, self.steps, self.operation_state_dim)))
            operation_action_batch.append(np.reshape(operation_action, (1, 1)))
            if self.is_bile:
                bile_state_batch.append(np.asarray(set_state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
                bile_next_state_batch.append(np.asarray(next_set_state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
                bile_set_action_batch.append(np.reshape(set_action, (1, 1)))
                bile_operation_action_batch.append(np.reshape(operation_action, (1, 1)))
                bile_reward_ext_batch.append(np.reshape(r_ext, (1, 1)))

            _add_totals(
                totals,
                metrics,
                r_ext,
                r_int,
                counter_curiosity,
                r_coh,
                r_div,
                step_reward,
                bile_bonus,
                bootstrap_bonus,
                exploration_reward,
            )
            if self.env.exploration_trace_rows:
                self.env.exploration_trace_rows[-1]["agent_id"] = int(self.agent_id)
                self.env.exploration_trace_rows[-1]["step_coherency"] = float(r_coh)
                self.env.exploration_trace_rows[-1]["step_diversity"] = float(r_div)
                self.env.exploration_trace_rows[-1]["step_total_reward"] = float(step_reward)
                self.env.exploration_trace_rows[-1]["step_bile_bonus"] = float(bile_bonus)
                self.env.exploration_trace_rows[-1]["step_bootstrap_diversity"] = float(bootstrap_bonus)
                self.env.exploration_trace_rows[-1]["bootstrap_active"] = int(bootstrap_active)
                self.env.exploration_trace_rows[-1]["escape_active"] = int(escape_active)
                self.env.exploration_trace_rows[-1]["bile_z_source"] = bile_z_source

            if len(set_state_batch) >= int(self.args.update_interval) or done:
                set_states = self.list_to_batch(set_state_batch)
                set_actions = self.list_to_batch(set_action_batch)
                operation_states = self.list_to_batch(operation_state_batch)
                operation_actions = self.list_to_batch(operation_action_batch)
                rewards = self.list_to_batch(reward_batch)
                next_v_value = self.critic.model.predict(
                    np.asarray(next_set_action_steps, dtype=np.float32).reshape((1, self.steps, self.set_state_dim)),
                    verbose=0,
                )
                td_targets = self.n_step_td_target(rewards, next_v_value, done)
                advantages = td_targets - self.critic.model.predict(set_states, verbose=0)

                set_grads, _ = self.set_actor.get_gradients(set_states, set_actions, advantages)
                op_grads, _ = self.operation_actor.get_gradients(operation_states, operation_actions, advantages)
                critic_grads, _ = self.critic.get_gradients(set_states, td_targets)
                bile_encoder_grads = None
                bile_dynamics_grads = None
                if self.bile is not None and bile_state_batch:
                    local_bile_states = self.list_to_batch(bile_state_batch)
                    local_bile_set_actions = self.list_to_batch(bile_set_action_batch)
                    local_bile_operation_actions = self.list_to_batch(bile_operation_action_batch)
                    local_bile_rewards_ext = self.list_to_batch(bile_reward_ext_batch)
                    local_bile_next_states = self.list_to_batch(bile_next_state_batch)
                    bile_pair_sample = ray.get(
                        self.ps.update_and_sample_bile_batch.remote(
                            local_bile_states,
                            local_bile_set_actions,
                            local_bile_operation_actions,
                            local_bile_rewards_ext,
                            local_bile_next_states,
                            int(episode),
                        )
                    )
                    totals["bile_pair_source"] = str(bile_pair_sample.get("source", "local"))
                    totals["bile_replay_size"] = int(bile_pair_sample.get("buffer_size", 0))
                    if bile_pair_sample.get("source") == "replay_random_perm":
                        train_bile_states = np.asarray(bile_pair_sample["states"], dtype=np.float32)
                        train_bile_set_actions = np.asarray(bile_pair_sample["set_actions"], dtype=np.int64)
                        train_bile_operation_actions = np.asarray(bile_pair_sample["operation_actions"], dtype=np.int64)
                        train_bile_rewards_ext = np.asarray(bile_pair_sample["rewards_ext"], dtype=np.float32)
                        train_bile_next_states = np.asarray(bile_pair_sample["next_states"], dtype=np.float32)
                        pair_indices = np.asarray(bile_pair_sample["pair_indices"], dtype=np.int64)
                    else:
                        train_bile_states = local_bile_states
                        train_bile_set_actions = local_bile_set_actions
                        train_bile_operation_actions = local_bile_operation_actions
                        train_bile_rewards_ext = local_bile_rewards_ext
                        train_bile_next_states = local_bile_next_states
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
                        use_reward_diff=self.args.baseline == "mira",
                    )
                    if bile_stats is not None:
                        bile_encoder_grads = [
                            grad.numpy() if grad is not None else None
                            for grad in bile_stats["encoder_grads"]
                        ]
                        bile_dynamics_grads = [
                            grad.numpy() if grad is not None else None
                            for grad in bile_stats["dynamics_grads"]
                        ]
                        for key in (
                            "phi_loss",
                            "dyn_loss",
                            "prediction_error",
                            "mean_d_phi",
                            "mean_metric_target",
                        ):
                            totals[f"bile_{key}"] += float(bile_stats[key])
                        totals["bile_update_count"] += 1
                new_weights = ray.get(
                    self.ps.apply_gradients_and_get_weights.remote(
                        [grad.numpy() if grad is not None else None for grad in set_grads],
                        [grad.numpy() if grad is not None else None for grad in op_grads],
                        [grad.numpy() if grad is not None else None for grad in critic_grads],
                        bile_encoder_grads,
                        bile_dynamics_grads,
                    )
                )
                self.set_actor.model.set_weights(new_weights["set"])
                self.operation_actor.model.set_weights(new_weights["op"])
                self.critic.model.set_weights(new_weights["critic"])
                if self.bile is not None and new_weights.get("bile") is not None:
                    self.bile.set_weights(new_weights["bile"])
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

        for pair, count in episode_set_op_counters.items():
            self.set_op_counters[pair] = self.set_op_counters.get(pair, 0) + count

        if totals.get("bile_update_count", 0) > 0:
            update_count = float(totals["bile_update_count"])
            for key in (
                "bile_phi_loss",
                "bile_dyn_loss",
                "bile_prediction_error",
                "bile_mean_d_phi",
                "bile_mean_metric_target",
            ):
                totals[key] = float(totals[key]) / update_count

        success_candidate_states = episode_transition_states + episode_trajectory_states
        success_candidate_next_states = episode_transition_next_states + episode_trajectory_next_states
        success_candidate_scores = episode_transition_ext_rewards + episode_trajectory_scores
        success_candidate_kinds = (
            ["local"] * len(episode_transition_states)
            + ["trajectory"] * len(episode_trajectory_states)
        )
        success_states, success_next_states, success_scores, success_kinds = _select_success_transitions(
            success_candidate_states,
            success_candidate_next_states,
            success_candidate_scores,
            gamma=self.args.gamma,
            min_score=self.args.bile_success_quality_min_score,
            kinds=success_candidate_kinds,
        )

        if success_states:
            totals["bile_zpool_size"] = int(
                ray.get(
                    self.ps.update_bile_success_transitions.remote(
                        success_states,
                        success_next_states,
                        success_scores,
                        success_kinds,
                        int(episode),
                    )
                )
            )

        trace_rows, state_rows = self.env.consume_exploration_logs()
        stats = ray.get(
            self.ps.record_episode.remote(
                episode,
                totals,
                list(getattr(self.env, "sets_viewed", [])),
                trace_rows,
                state_rows,
            )
        )
        print(
            f"EP{stats['episode']} {self.args.baseline} Agent{self.agent_id} | "
            f"Ext_R: {totals['extrinsic_reward']:.2f} | Total: {totals['total_reward']:.2f} | "
            f"Sets: {stats['sets_viewed']} | Done: {stats['episodes_done']}",
            flush=True,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Run Covertype full Ray/TensorFlow dual-actor A3C baselines.")
    parser.add_argument("--baseline", choices=sorted(BASELINES), default=None)
    parser.add_argument("--csv_path", default="covertype.csv")
    parser.add_argument("--target_set", default="fixed_seed_1")
    parser.add_argument("--target_seed", type=int, default=None)
    parser.add_argument("--target_size", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--n_sets", type=int, default=50000)
    parser.add_argument("--min_set_size", type=int, default=10)
    parser.add_argument("--max_set_size", type=int, default=200000)
    parser.add_argument("--preprocess_name", default=None)
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--actor_lr", type=float, default=0.00003)
    parser.add_argument("--critic_lr", type=float, default=0.00003)
    parser.add_argument("--lstm_steps", type=int, default=5)
    parser.add_argument("--set_dense_units", type=int, default=512)
    parser.add_argument("--set_lstm_units", type=int, default=512)
    parser.add_argument("--op_dense_units", type=int, default=512)
    parser.add_argument("--op_lstm_units", type=int, default=256)
    parser.add_argument("--critic_dense_units", default="512,256,128")
    parser.add_argument("--critic_lstm_units", type=int, default=128)
    parser.add_argument("--candidate_slots", type=int, default=10)
    parser.add_argument("--disable_bile_candidate_rerank", action="store_true")
    parser.add_argument("--bile_candidate_family_keep", type=int, default=4)
    parser.add_argument("--bile_candidate_prior_weight", type=float, default=0.25)
    parser.add_argument("--bile_candidate_prior_success_weight", type=float, default=0.55)
    parser.add_argument("--bile_candidate_prior_temperature", type=float, default=0.18)
    parser.add_argument("--bile_candidate_prior_min_zpool", type=int, default=20)
    parser.add_argument("--counter_curiosity_ratio", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--w_ext", type=float, default=8.0)
    parser.add_argument("--w_int", type=float, default=0.2)
    parser.add_argument("--w_bile", type=float, default=1.0)
    parser.add_argument("--w_bootstrap", type=float, default=1.0)
    parser.add_argument("--w_coh", type=float, default=1.0)
    parser.add_argument("--w_div", type=float, default=1.0)
    parser.add_argument("--diversity_eta", type=float, default=0.1)
    parser.add_argument("--bile_latent_dim", type=int, default=16)
    parser.add_argument("--bile_lr", type=float, default=0.0003)
    parser.add_argument("--bile_dense_units", type=int, default=256)
    parser.add_argument("--bile_beta_pe", type=float, default=1.0)
    parser.add_argument("--bile_metric_clip", type=float, default=10.0)
    parser.add_argument("--bile_bonus_clip", type=float, default=1.0)
    parser.add_argument("--bile_target_tau", type=float, default=0.01)
    parser.add_argument("--bile_phi_weight", type=float, default=1.0)
    parser.add_argument("--bile_dyn_weight", type=float, default=1.0)
    parser.add_argument("--bile_pair_warmup_episodes", type=int, default=100)
    parser.add_argument("--bile_replay_buffer_size", type=int, default=50000)
    parser.add_argument("--bile_phi_batch_size", type=int, default=128)
    parser.add_argument("--bile_min_replay_size", type=int, default=256)
    parser.add_argument("--bile_success_prob", type=float, default=0.6)
    parser.add_argument("--bile_orthogonal_prob", type=float, default=0.3)
    parser.add_argument("--bile_success_prob_min", type=float, default=0.35)
    parser.add_argument("--bile_success_prob_decay_episodes", type=int, default=600)
    parser.add_argument("--bile_success_pool_size", type=int, default=256)
    parser.add_argument("--bile_success_noise_scale", type=float, default=0.2)
    parser.add_argument("--bile_success_mix_random_prob", type=float, default=0.25)
    parser.add_argument("--bile_success_mix_random_weight", type=float, default=0.35)
    parser.add_argument("--bile_success_trajectory_score_scale", type=float, default=1.0)
    parser.add_argument("--bile_min_success_reward", type=float, default=1.0)
    parser.add_argument("--bile_success_quality_mode", choices=["none", "size_penalty"], default="size_penalty")
    parser.add_argument("--bile_success_size_penalty_power", type=float, default=1.0)
    parser.add_argument("--bile_success_quality_min_score", type=float, default=0.0)
    parser.add_argument("--bile_success_episode_top_k", type=int, default=24)
    parser.add_argument("--bile_success_episode_min_fraction", type=float, default=0.25)
    parser.add_argument("--bile_success_recent_window", type=int, default=150)
    parser.add_argument("--bile_success_recent_fraction", type=float, default=0.35)
    parser.add_argument("--bile_success_recency_half_life", type=float, default=300.0)
    parser.add_argument("--bile_success_score_clip", type=float, default=150.0)
    parser.add_argument("--bile_success_weight_power", type=float, default=0.5)
    parser.add_argument("--bootstrap_window", type=int, default=15)
    parser.add_argument("--bootstrap_ext_threshold", type=float, default=10.0)
    parser.add_argument("--bootstrap_success_threshold", type=float, default=100.0)
    parser.add_argument("--bootstrap_success_ratio_threshold", type=float, default=0.30)
    parser.add_argument("--bootstrap_use_success_ratio", action="store_true")
    parser.add_argument("--bootstrap_zpool_threshold", type=int, default=20)
    parser.add_argument("--bootstrap_distance_eta", type=float, default=0.1)
    parser.add_argument("--escape_window", type=int, default=20)
    parser.add_argument("--escape_ext_threshold", type=float, default=10.0)
    parser.add_argument("--escape_low_reward_threshold", type=float, default=10.0)
    parser.add_argument("--escape_low_ratio_threshold", type=float, default=0.70)
    parser.add_argument("--escape_success_prob", type=float, default=0.80)
    parser.add_argument("--escape_orthogonal_prob", type=float, default=0.80)
    parser.add_argument("--escape_random_z_prob", type=float, default=0.20)
    parser.add_argument("--escape_min_success_score", type=float, default=5.0)
    parser.add_argument("--escape_stale_success_ratio_threshold", type=float, default=0.05)
    parser.add_argument("--escape_stale_success_scale", type=float, default=0.35)
    parser.add_argument("--escape_stale_orthogonal_prob", type=float, default=0.85)
    parser.add_argument("--escape_stale_random_z_prob", type=float, default=0.35)
    parser.add_argument("--escape_low_success_scale", type=float, default=0.60)
    parser.add_argument("--escape_low_random_z_prob", type=float, default=0.30)
    parser.add_argument("--escape_random_action_prob", type=float, default=0.20)
    parser.add_argument("--bootstrap_random_action_prob", type=float, default=0.05)
    parser.add_argument("--escape_alpha", type=float, default=0.10)
    parser.add_argument("--escape_bootstrap_scale", type=float, default=0.20)
    parser.add_argument("--name", default="")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--resume_model_dir", default=None)
    parser.add_argument("--resume_start_episode", type=int, default=0)
    parser.add_argument("--ray_address", default=None)
    parser.add_argument("--ray_temp_dir", default=None)
    return parser


def run(args):
    if args.baseline not in BASELINES:
        raise ValueError(f"Missing or unsupported full A3C baseline: {args.baseline}. Use one of {sorted(BASELINES)}.")

    root_dir = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = root_dir / csv_path

    if not args.name:
        args.name = args.output_prefix or f"{args.baseline}_seed{args.seed}_full_a3c"

    data = load_covertype(csv_path, n_bins=args.n_bins)
    target_items, target_path = resolve_target_set(
        data,
        root_dir=root_dir,
        target_set=args.target_set,
        target_seed=args.target_seed,
        target_size=args.target_size,
    )
    actions = build_action_space(n_continuous=data.continuous.shape[1], n_bins=data.n_bins)
    universe, universe_dir = ensure_fixed_universe(
        data=data,
        actions=actions,
        target_items=target_items,
        target_path=target_path,
        root_dir=root_dir,
        n_sets=args.n_sets,
        seed=args.seed,
        min_set_size=args.min_set_size,
        max_set_size=args.max_set_size,
        preprocess_name=args.preprocess_name,
        force=args.force_preprocess,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    args.csv_path = str(csv_path)
    args.target_path = str(target_path)
    args.fixed_universe_dir = str(universe_dir)
    args.fixed_universe_sets = int(universe.n_sets)
    args.target_items = int(len(target_items))

    dummy_env = CovertypeDualActorEnvironment(
        universe,
        actions,
        episode_steps=args.steps,
        seed=args.seed,
        candidate_slots=args.candidate_slots,
    )
    dummy_env.reset()
    args.base_set_state_dim = int(dummy_env.set_state_dim)
    args.base_operation_state_dim = int(dummy_env.operation_state_dim)
    set_state_dim = int(dummy_env.set_state_dim)
    operation_state_dim = int(dummy_env.operation_state_dim)
    if args.baseline in MIRA_BASELINES:
        set_state_dim += int(args.bile_latent_dim)
        operation_state_dim += int(args.bile_latent_dim)

    if not ray.is_initialized():
        ray_kwargs = {
            "address": args.ray_address,
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "_system_config": {
                "health_check_failure_threshold": 20,
                "health_check_period_ms": 1000,
            },
        }
        if args.ray_temp_dir:
            ray_kwargs["_temp_dir"] = args.ray_temp_dir
        ray.init(**ray_kwargs)

    args_dict = vars(args).copy()
    ps = ParameterServer.remote(
        set_state_dim,
        operation_state_dim,
        dummy_env.set_action_dim,
        dummy_env.operation_action_dim,
        args_dict,
    )
    action_labels = [action.label for action in actions]
    workers = [
        WorkerAgent.remote(str(universe_dir), action_labels, ps, worker_id, args_dict)
        for worker_id in range(max(1, int(args.workers)))
    ]
    ray.get([worker.train_loop.remote() for worker in workers])
    ray.get(ps.save_models.remote("final"))
    return ray.get(ps.output_paths.remote())


def main():
    paths = run(build_parser().parse_args())
    print("Wrote:")
    for key, value in paths.items():
        print(f"  {key}: {value}")


def _configure_tf():
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass


def _masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    masked = np.where(mask > 0.0, probs, 0.0)
    total = float(masked.sum())
    if total <= 0.0 or not np.isfinite(total):
        valid = np.flatnonzero(mask > 0.0)
        fallback = np.zeros_like(masked)
        if valid.size == 0:
            fallback[:] = 1.0 / float(fallback.size)
        else:
            fallback[valid] = 1.0 / float(valid.size)
        return fallback.astype(np.float64)
    return (masked / total).astype(np.float64)


def _actions_from_labels(labels):
    from .actions import Action

    actions = []
    for label in labels:
        parts = str(label).split(":")
        op = parts[0]
        if op in {"by_facet_cont", "by_superset_cont", "by_neighbors_cont"}:
            feature = int(parts[1]) if len(parts) > 1 else -1
            value = int(parts[2]) if len(parts) > 2 else -1
            delta = int(parts[3]) if len(parts) > 3 else 0
            actions.append(Action(op, feature=feature, value=value, delta=delta))
        elif op in {"by_facet_cover", "by_facet_wilderness", "by_facet_soil"}:
            value = int(parts[1]) if len(parts) > 1 else -1
            actions.append(Action(op, value=value))
        elif op == "by_distribution":
            delta = int(parts[1]) if len(parts) > 1 else 0
            actions.append(Action(op, delta=delta))
        else:
            actions.append(Action(op))
    return actions


def _parse_units(value):
    if isinstance(value, (list, tuple)):
        units = [int(item) for item in value]
    else:
        units = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(units) != 3:
        raise ValueError("--critic_dense_units must contain exactly three comma-separated integers, e.g. 512,256,128")
    return tuple(units)


def _counter_curiosity(pair, episode_counts, global_counts, episode_steps):
    episode_counts[pair] = episode_counts.get(pair, 0) + 1
    count = episode_counts[pair] + global_counts.get(pair, 0)
    return (100.0 / float(max(episode_steps, 1))) / float(max(count, 1))


def _coherency(phi_current, phi_next):
    if np.allclose(phi_current, phi_next, atol=1e-5):
        return 0.0
    denom = float(np.linalg.norm(phi_current) * np.linalg.norm(phi_next))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(phi_current, phi_next) / denom)


def _episode_diversity(phi_next, history, eta=0.1):
    if not history:
        return 1.0
    distances = [np.linalg.norm(phi_next - item) for item in history]
    min_distance = float(np.min(distances)) if distances else 0.0
    return float(1.0 - np.exp(-float(eta) * min_distance))


def _training_reward(
    baseline,
    r_ext,
    r_int,
    counter_curiosity,
    r_coh,
    r_div,
    bile_bonus,
    args,
):
    if baseline == "pure_a3c":
        return float(args.w_ext) * float(r_ext)
    if baseline == "paper_a3c":
        ratio = float(args.counter_curiosity_ratio)
        return (1.0 - ratio) * float(r_ext) + ratio * float(counter_curiosity)
    if baseline == "atena":
        return float(args.w_int) * float(r_int) + float(args.w_coh) * float(r_coh) + float(args.w_div) * float(r_div)
    if baseline == "atena_extrinsic":
        return (
            float(args.w_ext) * float(r_ext)
            + float(args.w_int) * float(r_int)
            + float(args.w_coh) * float(r_coh)
            + float(args.w_div) * float(r_div)
        )
    if baseline == "mira":
        return float(args.w_ext) * float(r_ext) + float(args.alpha) * float(bile_bonus)
    if baseline == "mira_no_ext":
        return float(args.alpha) * float(bile_bonus)
    raise ValueError(f"Unsupported baseline: {baseline}")


def _success_quality_score(score, set_size, args):
    score = float(score)
    mode = getattr(args, "bile_success_quality_mode", "size_penalty")
    if mode == "none":
        return score
    if mode == "size_penalty":
        safe_size = max(float(set_size), 1.0)
        power = float(getattr(args, "bile_success_size_penalty_power", 1.0))
        denom = max(np.log1p(safe_size) ** power, 1e-6)
        return score / denom
    return score


def _select_success_transitions(
    states,
    next_states,
    rewards_ext,
    gamma,
    min_score,
    kinds=None,
    top_k=None,
    min_fraction=0.0,
):
    if not states:
        return [], [], [], []

    _ = gamma, top_k, min_fraction
    kinds = list(kinds or ["local"] * len(states))
    selected_states = []
    selected_next_states = []
    selected_scores = []
    selected_kinds = []
    for idx, reward_ext in enumerate(rewards_ext):
        score = float(reward_ext)
        if score <= float(min_score):
            continue
        selected_states.append(states[idx])
        selected_next_states.append(next_states[idx])
        selected_scores.append(score)
        selected_kinds.append(str(kinds[idx] if idx < len(kinds) else "local"))
    return selected_states, selected_next_states, selected_scores, selected_kinds


def _zero_totals():
    return {
        "extrinsic_reward": 0.0,
        "target_hits": 0,
        "interestingness": 0.0,
        "familiarity": 0.0,
        "counter_curiosity": 0.0,
        "coherency": 0.0,
        "diversity": 0.0,
        "total_reward": 0.0,
        "valid_steps": 0,
        "bile_bonus": 0.0,
        "bootstrap_diversity": 0.0,
        "exploration_reward": 0.0,
        "bile_phi_loss": 0.0,
        "bile_dyn_loss": 0.0,
        "bile_prediction_error": 0.0,
        "bile_mean_d_phi": 0.0,
        "bile_mean_metric_target": 0.0,
        "bile_update_count": 0,
        "bile_z_source": "",
        "bile_zpool_size": 0,
        "recent_success_ratio": 0.0,
        "recent_low_reward_ratio": 0.0,
        "bootstrap_active": 0,
        "escape_active": 0,
        "bile_pair_source": "",
        "bile_replay_size": 0,
    }


def _add_totals(
    totals,
    metrics,
    r_ext,
    r_int,
    counter_curiosity,
    r_coh,
    r_div,
    step_reward,
    bile_bonus=0.0,
    bootstrap_bonus=0.0,
    exploration_reward=0.0,
):
    totals["extrinsic_reward"] += float(r_ext)
    totals["target_hits"] += int(metrics.get("target_hits", 0))
    totals["interestingness"] += float(r_int)
    totals["familiarity"] += float(r_ext)
    totals["counter_curiosity"] += float(counter_curiosity)
    totals["coherency"] += float(r_coh)
    totals["diversity"] += float(r_div)
    totals["total_reward"] += float(step_reward)
    totals["valid_steps"] += int(metrics.get("valid", 0))
    totals["bile_bonus"] += float(bile_bonus)
    totals["bootstrap_diversity"] += float(bootstrap_bonus)
    totals["exploration_reward"] += float(exploration_reward)


if __name__ == "__main__":
    main()
