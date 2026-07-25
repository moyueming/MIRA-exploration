import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira.schedule import runtime_args, training_schedule
from mira.swa import RunningWeightAverage
from run import parse_args


class ScheduleContractTests(unittest.TestCase):
    def test_mira_schedule_keeps_original_numeric_contract(self):
        args = parse_args(["--schema", "cyber", "--dataset_number", "3"])
        expected = {
            0: (3e-4, 0.5, 0.01, 1.0),
            400_000: (1.8e-4, 0.5, 0.01, 1.0),
            500_000: (1.5e-4, 0.5, 0.01, 1.0),
            750_000: (7.8e-5, 0.3, 0.0055, 0.625),
            1_000_000: (6e-6, 0.1, 0.001, 0.25),
        }

        for steps, wanted in expected.items():
            with self.subTest(steps=steps):
                schedule = training_schedule(args, steps)
                observed = (
                    schedule["policy_lr"],
                    schedule["alpha"],
                    schedule["entropy_coef"],
                    schedule["auxiliary_reward_scale"],
                )
                for actual, target in zip(observed, wanted):
                    self.assertAlmostEqual(actual, target)
                self.assertAlmostEqual(schedule["mira_lr"], schedule["policy_lr"])

    def test_runtime_scales_only_mira_auxiliary_weights(self):
        args = parse_args(["--schema", "cyber", "--dataset_number", "3"])
        runtime = runtime_args(args, training_schedule(args, 1_000_000))

        self.assertAlmostEqual(runtime.w_column_coverage, 0.0875)
        self.assertAlmostEqual(runtime.w_group_coverage, 0.075)
        self.assertAlmostEqual(runtime.w_structure, 0.0625)
        self.assertAlmostEqual(runtime.w_kl, 1.5)
        self.assertAlmostEqual(runtime.w_compaction, 2.0)


class RunningWeightAverageTests(unittest.TestCase):
    def test_starts_at_point_four_and_averages_without_aliasing(self):
        average = RunningWeightAverage(start_fraction=0.4)
        first = [np.asarray([1.0, 3.0], dtype=np.float32)]
        second = [np.asarray([3.0, 5.0], dtype=np.float32)]

        self.assertFalse(average.update(first, 0.399999))
        self.assertTrue(average.update(first, 0.4))
        first[0][:] = 99.0
        self.assertTrue(average.update(second, 0.8))

        np.testing.assert_allclose(
            average.formal_weights([np.zeros(2, dtype=np.float32)])[0],
            np.asarray([2.0, 4.0], dtype=np.float32),
        )
        self.assertEqual(average.count, 2)


@unittest.skipUnless(
    importlib.util.find_spec("tensorflow") is not None,
    "TensorFlow is not installed in this runtime",
)
class FullMiraModelTests(unittest.TestCase):
    def test_mira_rollout_weights_include_full_metric_module(self):
        from mira.models import MiraMetricModule

        mira = MiraMetricModule(state_dim=4, action_dim=3, latent_dim=2, hidden=8)
        weights = mira.get_weights()

        self.assertEqual(set(weights), {"encoder", "target_encoder", "dynamics"})
        self.assertTrue(weights["encoder"])
        self.assertTrue(weights["target_encoder"])
        self.assertTrue(weights["dynamics"])


if __name__ == "__main__":
    unittest.main()
