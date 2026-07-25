import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


class BILEModule:
    """Online BILE encoder and dynamics model for set exploration."""

    def __init__(
        self,
        state_dim,
        set_action_dim,
        operation_action_dim,
        latent_dim=16,
        lr=3e-4,
        dense_units=256,
        target_tau=0.01,
    ):
        self.state_dim = int(state_dim)
        self.set_action_dim = int(set_action_dim)
        self.operation_action_dim = int(operation_action_dim)
        self.latent_dim = int(latent_dim)
        self.dense_units = int(dense_units)
        self.target_tau = float(target_tau)
        self.encoder = self._create_encoder()
        self.target_encoder = self._create_encoder()
        self.target_encoder.set_weights(self.encoder.get_weights())
        self.dynamics = self._create_dynamics()
        self.encoder_opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.dynamics_opt = tf.keras.optimizers.Adam(learning_rate=float(lr))

    def _create_encoder(self):
        return tf.keras.Sequential(
            [
                Input((self.state_dim,), dtype=tf.float32),
                Dense(self.dense_units, activation="relu", dtype=tf.float32),
                Dense(self.dense_units, activation="relu", dtype=tf.float32),
                Dense(self.latent_dim, activation="linear", dtype=tf.float32),
            ]
        )

    def _create_dynamics(self):
        action_dim = self.set_action_dim + self.operation_action_dim
        return tf.keras.Sequential(
            [
                Input((self.state_dim + action_dim,), dtype=tf.float32),
                Dense(self.dense_units, activation="relu", dtype=tf.float32),
                Dense(self.dense_units, activation="relu", dtype=tf.float32),
                Dense(self.state_dim, activation="linear", dtype=tf.float32),
            ]
        )

    def get_weights(self):
        return {
            "encoder": self.encoder.get_weights(),
            "target_encoder": self.target_encoder.get_weights(),
            "dynamics": self.dynamics.get_weights(),
        }

    def set_weights(self, weights):
        if not weights:
            return
        self.encoder.set_weights(weights["encoder"])
        self.target_encoder.set_weights(weights["target_encoder"])
        self.dynamics.set_weights(weights["dynamics"])

    def save_models(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save(directory / "bile_encoder")
        self.target_encoder.save(directory / "bile_target_encoder")
        self.dynamics.save(directory / "bile_dynamics")

    def action_features(self, set_actions, operation_actions):
        set_actions = np.asarray(set_actions, dtype=np.int64).reshape(-1)
        operation_actions = np.asarray(operation_actions, dtype=np.int64).reshape(-1)
        features = np.zeros(
            (set_actions.shape[0], self.set_action_dim + self.operation_action_dim),
            dtype=np.float32,
        )
        valid_set = (set_actions >= 0) & (set_actions < self.set_action_dim)
        valid_op = (operation_actions >= 0) & (operation_actions < self.operation_action_dim)
        rows = np.arange(set_actions.shape[0])
        features[rows[valid_set], set_actions[valid_set]] = 1.0
        features[rows[valid_op], self.set_action_dim + operation_actions[valid_op]] = 1.0
        return features

    def embed(self, states):
        states = _sanitize_states(states, self.state_dim)
        return self.encoder.predict(states, verbose=0)

    def compute_bonus(self, state, next_state, direction, clip_value=1.0):
        states = _sanitize_states(np.asarray([state, next_state], dtype=np.float32), self.state_dim)
        phi = self.encoder.predict(states, verbose=0)
        delta = phi[1] - phi[0]
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            return 0.0
        direction = normalize_direction(direction, self.latent_dim)
        reward = float(np.dot(delta / norm, direction))
        clip_value = float(clip_value)
        if clip_value > 0.0:
            reward = float(np.clip(reward, -clip_value, clip_value))
        return reward

    def get_gradients(
        self,
        states,
        set_actions,
        operation_actions,
        rewards_ext,
        next_states,
        pair_indices=None,
        gamma=0.99,
        beta_pe=1.0,
        metric_clip=10.0,
        phi_weight=1.0,
        dyn_weight=1.0,
        use_reward_diff=True,
    ):
        states_np = _sanitize_states(states, self.state_dim)
        next_states_np = _sanitize_states(next_states, self.state_dim)
        if states_np.shape[0] == 0:
            return None

        action_np = self.action_features(set_actions, operation_actions)
        rewards_np = np.asarray(rewards_ext, dtype=np.float32).reshape(-1, 1)
        if not bool(use_reward_diff):
            rewards_np = np.zeros_like(rewards_np, dtype=np.float32)
        has_pairs = states_np.shape[0] >= 2

        states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states_np, dtype=tf.float32)
        action_tf = tf.convert_to_tensor(action_np, dtype=tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards_np, dtype=tf.float32)
        if pair_indices is None:
            pair_idx = tf.roll(tf.range(tf.shape(states_tf)[0]), shift=1, axis=0)
        else:
            pair_np = np.asarray(pair_indices, dtype=np.int64).reshape(-1)
            if pair_np.shape[0] != states_np.shape[0]:
                pair_np = np.roll(np.arange(states_np.shape[0], dtype=np.int64), shift=1)
            pair_np = np.clip(pair_np, 0, max(states_np.shape[0] - 1, 0))
            pair_idx = tf.convert_to_tensor(pair_np, dtype=tf.int32)

        with tf.GradientTape(persistent=True) as tape:
            dyn_input = tf.concat([states_tf, action_tf], axis=1)
            pred_next = self.dynamics(dyn_input, training=True)
            per_sample_mse = tf.reduce_mean(tf.square(pred_next - next_states_tf), axis=1, keepdims=True)
            pred_error = tf.sqrt(per_sample_mse + 1e-8)
            dyn_loss = tf.reduce_mean(per_sample_mse)

            phi_i = self.encoder(states_tf, training=True)
            phi_j = tf.gather(phi_i, pair_idx)
            d_phi = _pair_distance(phi_i, phi_j)

            target_next_i = self.target_encoder(next_states_tf, training=False)
            target_next_j = tf.gather(target_next_i, pair_idx)
            next_distance = _pair_distance(target_next_i, target_next_j)

            reward_i = rewards_tf
            reward_j = tf.gather(reward_i, pair_idx)
            reward_diff = tf.abs(reward_i - reward_j)
            pe_j = tf.gather(pred_error, pair_idx)
            metric_target = reward_diff + float(beta_pe) * 0.5 * (pred_error + pe_j) + float(gamma) * next_distance
            if not has_pairs:
                metric_target = tf.zeros_like(metric_target)
            if float(metric_clip) > 0.0:
                metric_target = tf.clip_by_value(metric_target, 0.0, float(metric_clip))
            metric_target = tf.stop_gradient(metric_target)
            phi_loss = tf.reduce_mean(tf.square(d_phi - metric_target))
            encoder_loss = float(phi_weight) * phi_loss
            dynamics_loss = float(dyn_weight) * dyn_loss

        encoder_grads = tape.gradient(encoder_loss, self.encoder.trainable_variables)
        dynamics_grads = tape.gradient(dynamics_loss, self.dynamics.trainable_variables)
        del tape

        return {
            "encoder_grads": encoder_grads,
            "dynamics_grads": dynamics_grads,
            "phi_loss": float(phi_loss.numpy()),
            "dyn_loss": float(dyn_loss.numpy()),
            "prediction_error": float(tf.reduce_mean(pred_error).numpy()),
            "mean_d_phi": float(tf.reduce_mean(d_phi).numpy()),
            "mean_metric_target": float(tf.reduce_mean(metric_target).numpy()),
        }

    def apply_gradients(self, encoder_grads=None, dynamics_grads=None):
        if encoder_grads:
            clean = []
            for grad, variable in zip(encoder_grads, self.encoder.trainable_variables):
                if grad is not None:
                    clean.append((grad, variable))
            if clean:
                self.encoder_opt.apply_gradients(clean)
        if dynamics_grads:
            clean = []
            for grad, variable in zip(dynamics_grads, self.dynamics.trainable_variables):
                if grad is not None:
                    clean.append((grad, variable))
            if clean:
                self.dynamics_opt.apply_gradients(clean)
        self.soft_update_target()

    def soft_update_target(self):
        tau = float(self.target_tau)
        if tau <= 0.0:
            return
        source = self.encoder.get_weights()
        target = self.target_encoder.get_weights()
        mixed = [(tau * s) + ((1.0 - tau) * t) for s, t in zip(source, target)]
        self.target_encoder.set_weights(mixed)


def sample_direction(rng, latent_dim):
    return normalize_direction(rng.normal(size=int(latent_dim)).astype(np.float32), latent_dim)


def normalize_direction(direction, latent_dim):
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    latent_dim = int(latent_dim)
    if direction.size != latent_dim:
        fixed = np.zeros(latent_dim, dtype=np.float32)
        fixed[: min(direction.size, latent_dim)] = direction[: min(direction.size, latent_dim)]
        direction = fixed
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        direction = np.zeros(latent_dim, dtype=np.float32)
        direction[0] = 1.0
        return direction
    return (direction / norm).astype(np.float32)


def _sanitize_states(states, state_dim):
    states = np.asarray(states, dtype=np.float32)
    states = states.reshape((-1, int(state_dim)))
    return np.nan_to_num(np.clip(states, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0)


def _pair_distance(left, right):
    return tf.sqrt(tf.reduce_sum(tf.square(left - right), axis=1, keepdims=True) + 1e-8)
