from pathlib import Path
import re

import h5py
import numpy as np


class ArrayTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self._values


class NumpyPolicyValueNet:
    def __init__(self, state_dim, action_dim, hidden=256):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.weights = None

    def load_weights(self, path):
        path = Path(path)
        with h5py.File(path, "r") as handle:
            d1, d2, policy, value = self._dense_layer_names(handle)
            weights = {
                "d1_kernel": self._read(handle, d1, "kernel:0"),
                "d1_bias": self._read(handle, d1, "bias:0"),
                "d2_kernel": self._read(handle, d2, "kernel:0"),
                "d2_bias": self._read(handle, d2, "bias:0"),
                "policy_kernel": self._read(handle, policy, "kernel:0"),
                "policy_bias": self._read(handle, policy, "bias:0"),
                "value_kernel": self._read(handle, value, "kernel:0"),
                "value_bias": self._read(handle, value, "bias:0"),
            }
        expected = {
            "d1_kernel": (self.state_dim, self.hidden),
            "d1_bias": (self.hidden,),
            "d2_kernel": (self.hidden, self.hidden),
            "d2_bias": (self.hidden,),
            "policy_kernel": (self.hidden, self.action_dim),
            "policy_bias": (self.action_dim,),
            "value_kernel": (self.hidden, 1),
            "value_bias": (1,),
        }
        mismatches = {
            name: (tuple(weights[name].shape), shape)
            for name, shape in expected.items()
            if tuple(weights[name].shape) != shape
        }
        if mismatches:
            raise ValueError(f"weight shape mismatch: {mismatches}")
        self.weights = weights
        return self

    def __call__(self, states, training=False):
        del training
        if self.weights is None:
            raise ValueError("weights have not been loaded")
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.state_dim)
        hidden1 = np.maximum(
            states @ self.weights["d1_kernel"] + self.weights["d1_bias"],
            0.0,
        )
        hidden2 = np.maximum(
            hidden1 @ self.weights["d2_kernel"] + self.weights["d2_bias"],
            0.0,
        )
        logits = hidden2 @ self.weights["policy_kernel"] + self.weights["policy_bias"]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        policy = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        value = hidden2 @ self.weights["value_kernel"] + self.weights["value_bias"]
        return ArrayTensor(policy), ArrayTensor(value)

    @staticmethod
    def _dense_layer_names(handle):
        names = []
        for name, group in handle.items():
            if not isinstance(group, h5py.Group) or name not in group:
                continue
            weights = group[name]
            if "kernel:0" in weights and "bias:0" in weights:
                names.append(name)
        names.sort(key=NumpyPolicyValueNet._dense_layer_sort_key)
        if len(names) != 4:
            raise ValueError(f"expected four Keras Dense layers, found {names}")
        return names

    @staticmethod
    def _dense_layer_sort_key(name):
        match = re.fullmatch(r"dense(?:_(\d+))?", str(name))
        if match is None:
            return (1, str(name))
        return (0, int(match.group(1) or 0))

    @staticmethod
    def _read(handle, layer, dataset):
        return np.asarray(handle[f"{layer}/{layer}/{dataset}"], dtype=np.float32)
