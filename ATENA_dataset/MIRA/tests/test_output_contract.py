import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira import evaluate as evaluate_module


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class FakeModel:
    def __call__(self, _state, training=False):
        del training
        return FakeTensor([[0.1, 0.9]]), FakeTensor([[0.0]])


class FakeEnv:
    def __init__(self):
        self.actions = []

    def reset(self):
        return [0.0]

    def legal_action_mask(self):
        return [1.0, 1.0]

    def step(self, action):
        self.actions.append("ACTION-{}".format(action))
        return [0.0], 1.25, True, {}


class OutputContractTests(unittest.TestCase):
    def test_formal_and_online_action_files_are_distinct(self):
        args = SimpleNamespace(
            schema="cyber",
            dataset_number=1,
            seed=0,
            method="MIRA",
        )
        metrics = {
            "Precision": 0.1,
            "T-BLEU-1": 0.2,
            "T-BLEU-2": 0.3,
            "T-BLEU-3": 0.4,
            "EDA-Sim": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            evaluate_module,
            "make_env",
            side_effect=lambda *_args, **_kwargs: FakeEnv(),
        ), patch.object(
            evaluate_module,
            "official_metrics",
            return_value=metrics,
        ):
            result_dir = Path(tmp)
            formal = evaluate_module.evaluate_policy(
                args,
                FakeModel(),
                result_dir,
                100,
                action_filename_prefix="actions_steps",
            )
            online = evaluate_module.evaluate_policy(
                args,
                FakeModel(),
                result_dir,
                100,
                action_filename_prefix="actions_online_steps",
            )

            self.assertTrue((result_dir / "actions_steps100.json").exists())
            self.assertTrue((result_dir / "actions_online_steps100.json").exists())
            self.assertEqual(formal["episode_reward"], 1.25)
            self.assertEqual(online["T-BLEU-3"], 0.4)
            self.assertEqual(
                {key for key in metrics if key in formal},
                set(metrics),
            )

    def test_masked_argmax_does_not_select_illegal_action(self):
        policy = evaluate_module.masked_probs([0.9, 0.1], [0.0, 1.0])

        self.assertEqual(policy.tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
