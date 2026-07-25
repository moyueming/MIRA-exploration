import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


def masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0) * mask
    total = float(probs.sum())
    if total <= 0.0:
        valid = np.flatnonzero(mask > 0)
        fallback = np.zeros_like(probs, dtype=np.float64)
        if valid.size == 0:
            fallback[:] = 1.0 / max(len(fallback), 1)
        else:
            fallback[valid] = 1.0 / valid.size
        return fallback
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
