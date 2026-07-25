import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def load_path_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, Path(relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


greedy = load_path_module(
    "covertype_greedy_eda",
    "covertype-exploration/baselines/greedy_eda/run.py",
)


class UniverseSpy:
    def __init__(self):
        self.target_reads = 0

    def state_for_set(self, set_id):
        return np.asarray([float(set_id), 1.0], dtype=np.float32)

    def targets_for_set(self, set_id):
        self.target_reads += 1
        raise AssertionError("selector read target membership")


class CovertypeGreedyEdaTests(unittest.TestCase):
    def test_visible_candidates_never_reads_targets_and_preserves_slots(self):
        universe = UniverseSpy()
        env = type(
            "E",
            (),
            {"universe": universe, "candidate_set_ids": [1, 2, -1]},
        )()

        slots, candidates = greedy.visible_candidates(env, [0.2, 0.8, 0.0])

        self.assertEqual(slots, [0, 1])
        self.assertEqual([item.candidate_id for item in candidates], [1, 2])
        self.assertEqual(universe.target_reads, 0)

    def test_execute_selected_step_calls_environment_once(self):
        class Environment:
            calls = 0

            def step(self, action_id):
                self.calls += 1
                return np.zeros(2), {"extrinsic_reward": 0.0}, 1, True

        env = Environment()

        greedy.execute_selected_step(env, 3)

        self.assertEqual(env.calls, 1)

    def test_candidate_interestingness_uses_state_not_target_arrays(self):
        current = np.zeros(84, dtype=np.float32)
        candidate = np.zeros(84, dtype=np.float32)
        current[20] = 1.0
        candidate[21] = 1.0

        value = greedy.candidate_interestingness(
            current,
            candidate,
            current_size=100,
            candidate_size=50,
        )

        self.assertGreater(value, 0.0)

    def test_parser_uses_reproducible_experiment_defaults(self):
        args = greedy.build_parser().parse_args(
            ["--preprocess_name", "by_distribution_path100k_seed1"]
        )

        self.assertEqual(args.baseline, "greedy_eda")
        self.assertEqual(args.episodes, 1000)
        self.assertEqual(args.steps, 250)
        self.assertEqual(args.workers, 12)
        self.assertEqual(args.candidate_slots, 10)
        self.assertEqual(args.selection_top_k, 3)
        self.assertFalse(hasattr(args, "force_preprocess"))

    def test_parser_requires_fixed_preprocessing_name(self):
        with self.assertRaises(SystemExit):
            greedy.build_parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
