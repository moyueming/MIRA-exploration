import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import recalculate_baseline_corpus_metrics as recalculator


class SessionReconstructionTests(unittest.TestCase):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def numpy(self):
            return self.values

    class FakeModel:
        def __call__(self, _state, training=False):
            self.training = training
            return (
                SessionReconstructionTests.FakeTensor([[0.9, 0.1]]),
                SessionReconstructionTests.FakeTensor([[0.0]]),
            )

    class FakeEnv:
        state_dim = 1
        action_dim = 2

        def __init__(self, done_after=12):
            self.done_after = done_after
            self.actions = []

        def reset(self):
            return [0.0]

        def legal_action_mask(self):
            return [0.0, 1.0]

        def sample_legal_action(self):
            return 1

        def step(self, action):
            self.actions.append(f"action-{action}")
            done = len(self.actions) >= self.done_after
            return [0.0], 0.0, done, {}

    def test_load_result_config_returns_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"schema": "cyber", "seed": 0}), encoding="utf-8")

            config = recalculator.load_result_config(path)

        self.assertEqual(config.schema, "cyber")
        self.assertEqual(config.seed, 0)

    def test_rollout_argmax_respects_legal_action_mask(self):
        env = self.FakeEnv()
        model = self.FakeModel()

        actions = recalculator.rollout_argmax(env, model)

        self.assertEqual(actions, ["action-1"] * 12)
        self.assertFalse(model.training)

    def test_rollout_argmax_rejects_non_twelve_action_session(self):
        env = self.FakeEnv(done_after=2)

        with self.assertRaisesRegex(ValueError, "expected 12 actions, got 2"):
            recalculator.rollout_argmax(env, self.FakeModel())

    def test_reconstruct_random_sessions_replays_all_configured_seeds(self):
        seen_seeds = []

        def env_factory(schema, dataset, seed, args, reward_mode):
            seen_seeds.append((schema, dataset, seed, reward_mode, args.method))
            return self.FakeEnv()

        config = {
            "schema": "cyber",
            "dataset_number": 1,
            "seed": 0,
            "method": "random",
            "random_eval_episodes": 16,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

            sessions = recalculator.reconstruct_random_sessions(
                result_dir,
                env_factory=env_factory,
            )

        self.assertEqual(len(sessions), 16)
        self.assertEqual([item[2] for item in seen_seeds], list(range(16)))
        self.assertTrue(all(len(session) == 12 for session in sessions))


if __name__ == "__main__":
    unittest.main()
