import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


def masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0) * mask
    total = float(probs.sum())
    if total <= 0.0:
        valid = np.flatnonzero(mask > 0)
        fixed = np.zeros_like(probs, dtype=np.float64)
        if valid.size == 0:
            fixed[:] = 1.0 / max(len(fixed), 1)
        else:
            fixed[valid] = 1.0 / valid.size
        return fixed
    return probs / total


class PolicyValueNet(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.d1 = Dense(hidden, activation="relu")
        self.d2 = Dense(hidden, activation="relu")
        self.policy = Dense(action_dim, activation="softmax")
        self.value = Dense(1, activation="linear")
        self.build((None, int(state_dim)))

    def call(self, states, training=False):
        x = self.d1(states)
        x = self.d2(x)
        return self.policy(x), self.value(x)


def _mira_encoder(state_dim, latent_dim, hidden):
    return tf.keras.Sequential([
        Input((int(state_dim),), dtype=tf.float32),
        Dense(hidden, activation="relu", dtype=tf.float32),
        Dense(hidden, activation="relu", dtype=tf.float32),
        Dense(int(latent_dim), activation="linear", dtype=tf.float32),
    ])


class MiraMetricModule:
    """BILE/MIRA metric module adapted to ATENA state/action transitions."""

    def __init__(self, state_dim, action_dim, latent_dim=16, hidden=256, lr=3e-4, tau=0.01):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.tau = float(tau)
        self.encoder = self._encoder(hidden)
        self.target_encoder = self._encoder(hidden)
        self.target_encoder.set_weights(self.encoder.get_weights())
        self.dynamics = self._dynamics(hidden)
        self.encoder_opt = tf.keras.optimizers.Adam(float(lr))
        self.dynamics_opt = tf.keras.optimizers.Adam(float(lr))

    def _encoder(self, hidden):
        return _mira_encoder(self.state_dim, self.latent_dim, hidden)

    def _dynamics(self, hidden):
        return tf.keras.Sequential([
            Input((self.state_dim + self.action_dim,), dtype=tf.float32),
            Dense(hidden, activation="relu", dtype=tf.float32),
            Dense(hidden, activation="relu", dtype=tf.float32),
            Dense(self.state_dim, activation="linear", dtype=tf.float32),
        ])

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

    def embed(self, states):
        states = _states(states, self.state_dim)
        return self.encoder.predict(states, verbose=0)

    def compute_bonus(self, state, next_state, direction, clip_value=1.0):
        return _mira_bonus(
            self.encoder,
            self.state_dim,
            self.latent_dim,
            state, next_state, direction, clip_value,
        )

    def update(self, states, actions, rewards, next_states, gamma=0.99, beta_pe=1.0, metric_clip=10.0):
        states_np = _states(states, self.state_dim)
        next_np = _states(next_states, self.state_dim)
        if states_np.shape[0] < 2:
            return {"phi_loss": 0.0, "dyn_loss": 0.0, "metric_target": 0.0}
        actions_np = np.asarray(actions, dtype=np.int64).reshape(-1)
        one_hot = np.zeros((len(actions_np), self.action_dim), dtype=np.float32)
        valid = (actions_np >= 0) & (actions_np < self.action_dim)
        one_hot[np.arange(len(actions_np))[valid], actions_np[valid]] = 1.0
        rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        pair_idx = np.roll(np.arange(states_np.shape[0], dtype=np.int64), 1)

        states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)
        next_tf = tf.convert_to_tensor(next_np, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(one_hot, dtype=tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards_np, dtype=tf.float32)
        pair_tf = tf.convert_to_tensor(pair_idx, dtype=tf.int32)

        with tf.GradientTape(persistent=True) as tape:
            pred_next = self.dynamics(tf.concat([states_tf, actions_tf], axis=1), training=True)
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(pred_next - next_tf), axis=1, keepdims=True) + 1e-8)
            dyn_loss = tf.reduce_mean(tf.square(pred_next - next_tf))

            phi_i = self.encoder(states_tf, training=True)
            phi_j = tf.gather(phi_i, pair_tf)
            d_phi = tf.sqrt(tf.reduce_sum(tf.square(phi_i - phi_j), axis=1, keepdims=True) + 1e-8)

            target_next_i = self.target_encoder(next_tf, training=False)
            target_next_j = tf.gather(target_next_i, pair_tf)
            next_distance = tf.sqrt(
                tf.reduce_sum(tf.square(target_next_i - target_next_j), axis=1, keepdims=True) + 1e-8
            )
            reward_diff = tf.abs(rewards_tf - tf.gather(rewards_tf, pair_tf))
            pe_j = tf.gather(pred_error, pair_tf)
            metric_target = reward_diff + float(beta_pe) * 0.5 * (pred_error + pe_j) + float(gamma) * next_distance
            if float(metric_clip) > 0:
                metric_target = tf.clip_by_value(metric_target, 0.0, float(metric_clip))
            metric_target = tf.stop_gradient(metric_target)
            phi_loss = tf.reduce_mean(tf.square(d_phi - metric_target))

        enc_grads = tape.gradient(phi_loss, self.encoder.trainable_variables)
        dyn_grads = tape.gradient(dyn_loss, self.dynamics.trainable_variables)
        del tape
        self.encoder_opt.apply_gradients([
            (g, v) for g, v in zip(enc_grads, self.encoder.trainable_variables) if g is not None
        ])
        self.dynamics_opt.apply_gradients([
            (g, v) for g, v in zip(dyn_grads, self.dynamics.trainable_variables) if g is not None
        ])
        self.soft_update()
        return {
            "phi_loss": float(phi_loss.numpy()),
            "dyn_loss": float(dyn_loss.numpy()),
            "metric_target": float(tf.reduce_mean(metric_target).numpy()),
        }

    def soft_update(self):
        src = self.encoder.get_weights()
        tgt = self.target_encoder.get_weights()
        self.target_encoder.set_weights([
            (self.tau * s) + ((1.0 - self.tau) * t) for s, t in zip(src, tgt)
        ])


def sample_direction(rng, latent_dim):
    return normalize_direction(rng.normal(size=int(latent_dim)).astype(np.float32), latent_dim)


def normalize_direction(direction, latent_dim):
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    fixed = np.zeros(int(latent_dim), dtype=np.float32)
    fixed[: min(direction.size, fixed.size)] = direction[: min(direction.size, fixed.size)]
    norm = float(np.linalg.norm(fixed))
    if norm <= 1e-8:
        fixed[0] = 1.0
        return fixed
    return fixed / norm


def _mira_bonus(encoder, state_dim, latent_dim, state, next_state, direction, clip_value):
    states = _states(
        np.asarray([state, next_state], dtype=np.float32),
        state_dim,
    )
    phi = encoder(states, training=False).numpy()
    delta = phi[1] - phi[0]
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-8:
        return 0.0
    direction = normalize_direction(direction, latent_dim)
    reward = float(np.dot(delta / norm, direction))
    if float(clip_value) > 0:
        reward = float(np.clip(reward, -float(clip_value), float(clip_value)))
    return reward



def _states(states, state_dim):
    states = np.asarray(states, dtype=np.float32).reshape((-1, int(state_dim)))
    return np.nan_to_num(np.clip(states, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0)

