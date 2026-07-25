import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from recalculate_baseline_corpus_metrics.policy import NumpyPolicyValueNet


class NumpyPolicyValueNetTests(unittest.TestCase):
    def test_loads_keras_h5_and_matches_dense_relu_softmax_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.weights.h5"
            with h5py.File(path, "w") as handle:
                self._dense(handle, "dense", np.eye(2), np.zeros(2))
                self._dense(handle, "dense_1", np.eye(2), np.zeros(2))
                self._dense(handle, "dense_2", np.eye(2), np.zeros(2))
                self._dense(handle, "dense_3", np.ones((2, 1)), np.zeros(1))

            model = NumpyPolicyValueNet(2, 2, hidden=2)
            model.load_weights(path)
            policy, value = model(np.asarray([[1.0, 2.0]], dtype=np.float32), training=False)

        expected = np.exp([1.0, 2.0]) / np.exp([1.0, 2.0]).sum()
        np.testing.assert_allclose(policy.numpy()[0], expected, rtol=1e-6)
        np.testing.assert_allclose(value.numpy(), [[3.0]], rtol=1e-6)

    def test_loads_shifted_keras_dense_layer_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.weights.h5"
            with h5py.File(path, "w") as handle:
                self._dense(handle, "dense_13", np.eye(2), np.zeros(2))
                self._dense(handle, "dense_14", np.eye(2), np.zeros(2))
                self._dense(handle, "dense_15", np.eye(2), np.zeros(2))
                self._dense(handle, "dense_16", np.ones((2, 1)), np.zeros(1))

            model = NumpyPolicyValueNet(2, 2, hidden=2)
            model.load_weights(path)
            policy, value = model(
                np.asarray([[1.0, 2.0]], dtype=np.float32),
                training=False,
            )

        expected = np.exp([1.0, 2.0]) / np.exp([1.0, 2.0]).sum()
        np.testing.assert_allclose(policy.numpy()[0], expected, rtol=1e-6)
        np.testing.assert_allclose(value.numpy(), [[3.0]], rtol=1e-6)

    def test_rejects_weight_shapes_that_do_not_match_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.weights.h5"
            with h5py.File(path, "w") as handle:
                self._dense(handle, "dense", np.eye(3), np.zeros(3))
                self._dense(handle, "dense_1", np.eye(3), np.zeros(3))
                self._dense(handle, "dense_2", np.eye(3), np.zeros(3))
                self._dense(handle, "dense_3", np.ones((3, 1)), np.zeros(1))

            model = NumpyPolicyValueNet(2, 2, hidden=2)
            with self.assertRaisesRegex(ValueError, "weight shape mismatch"):
                model.load_weights(path)

    @staticmethod
    def _dense(handle, name, kernel, bias):
        group = handle.create_group(name).create_group(name)
        group.create_dataset("kernel:0", data=np.asarray(kernel, dtype=np.float32))
        group.create_dataset("bias:0", data=np.asarray(bias, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
