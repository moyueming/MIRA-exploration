import copy
import json
import logging
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atena_baselines.train as train_module
from atena_baselines.env import AtenaEDAEnv
from atena_baselines.greedy import run_greedy, select_greedy_action
from run_atena_baselines import validate_args


def preview_test_env():
    env = object.__new__(AtenaEDAEnv)
    env.action_space = [None, None, None]
    env.dataset = SimpleNamespace(dataset_df=object())
    env.simulator = SimpleNamespace(simulation_state=SimpleNamespace(
        states_history=["root-state"],
        states_stack=["root-state"],
        displays_history=["root-display"],
    ))
    env.actions = []
    env.step_index = 0
    env.displays = ["root-display"]
    env.display_strings = {"root-display"}
    env.prev_action_kind = "START"
    env.visit_counts = defaultdict(int)
    env.rng = np.random.default_rng(7)
    env.legal_action_mask = MethodType(
        lambda self: np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        env,
    )

    def fake_step(self, action_index):
        self.simulator.simulation_state.states_history.append(f"state-{action_index}")
        self.simulator.simulation_state.states_stack.append(f"state-{action_index}")
        self.simulator.simulation_state.displays_history.append(f"display-{action_index}")
        self.actions.append(f"action-{action_index}")
        self.step_index += 1
        self.displays.append(f"display-{action_index}")
        self.display_strings.add(f"display-{action_index}")
        self.prev_action_kind = "FILTER"
        self.visit_counts[f"state-{action_index}"] += 1
        self.rng.random()
        return np.asarray([action_index], dtype=np.float32), 2.5, False, {"action_index": action_index}

    env.step = MethodType(fake_step, env)
    return env


def observable_state(env):
    state = env.simulator.simulation_state
    return {
        "states_history": list(state.states_history),
        "states_stack": list(state.states_stack),
        "displays_history": list(state.displays_history),
        "actions": list(env.actions),
        "step_index": env.step_index,
        "displays": list(env.displays),
        "display_strings": set(env.display_strings),
        "prev_action_kind": env.prev_action_kind,
        "visit_counts": dict(env.visit_counts),
        "rng_state": copy.deepcopy(env.rng.bit_generator.state),
    }


class PreviewStepTests(unittest.TestCase):
    def test_preview_returns_real_step_reward_without_mutating_environment(self):
        env = preview_test_env()
        before = observable_state(env)
        dataset_before = env.dataset
        dataset_df_before = env.dataset.dataset_df

        reward, info = env.preview_step(1)

        self.assertEqual(reward, 2.5)
        self.assertEqual(info["action_index"], 1)
        self.assertEqual(observable_state(env), before)
        self.assertIs(env.dataset, dataset_before)
        self.assertIs(env.dataset.dataset_df, dataset_df_before)

        _, committed_reward, _, _ = env.step(1)
        self.assertEqual(committed_reward, reward)

    def test_preview_restores_environment_when_step_raises(self):
        env = preview_test_env()
        before = observable_state(env)

        def failing_step(self, action_index):
            self.simulator.simulation_state.states_history.append(f"state-{action_index}")
            self.simulator.simulation_state.states_stack.append(f"state-{action_index}")
            self.simulator.simulation_state.displays_history.append(f"display-{action_index}")
            self.actions.append(f"action-{action_index}")
            self.step_index += 1
            self.displays.append(f"display-{action_index}")
            self.display_strings.add(f"display-{action_index}")
            self.prev_action_kind = "FILTER"
            self.visit_counts[f"state-{action_index}"] += 1
            self.rng.random()
            raise RuntimeError("candidate failed")

        env.step = MethodType(failing_step, env)
        with self.assertRaisesRegex(RuntimeError, "candidate failed"):
            env.preview_step(1)
        self.assertEqual(observable_state(env), before)

    def test_preview_rejects_an_illegal_action(self):
        env = preview_test_env()
        with self.assertRaisesRegex(ValueError, "not legal"):
            env.preview_step(0)

    def test_root_state_disables_back_action(self):
        env = object.__new__(AtenaEDAEnv)
        env.action_space = [None, None, None]
        env.simulator = SimpleNamespace(
            simulation_state=SimpleNamespace(states_stack=["root-state"])
        )
        mask = AtenaEDAEnv.legal_action_mask(env)
        self.assertEqual(mask.tolist(), [0.0, 1.0, 1.0])


class FakeGreedyEnv:
    def __init__(self, mask, rewards):
        self._mask = np.asarray(mask, dtype=np.float32)
        self._rewards = dict(rewards)
        self.previewed = []

    def legal_action_mask(self):
        return self._mask.copy()

    def preview_step(self, action_index):
        self.previewed.append(int(action_index))
        return float(self._rewards[int(action_index)]), {"action_index": int(action_index)}


class NoisyGreedyEnv(FakeGreedyEnv):
    def __init__(self, mask, rewards, preview_error=None):
        super().__init__(mask, rewards)
        self.preview_error = preview_error

    def preview_step(self, action_index):
        logging.getLogger("atena.simulation.display").warning("invalid preview candidate")
        if self.preview_error is not None:
            raise self.preview_error
        return super().preview_step(action_index)


class GreedySelectionTests(unittest.TestCase):
    def test_selects_highest_reward_and_lowest_index_on_tie(self):
        env = FakeGreedyEnv([0, 1, 1, 1], {1: 4.0, 2: 5.0, 3: 5.0})
        action, reward, evaluated = select_greedy_action(env)
        self.assertEqual((action, reward, evaluated), (2, 5.0, 3))
        self.assertEqual(env.previewed, [1, 2, 3])

    def test_skips_non_finite_rewards(self):
        env = FakeGreedyEnv([1, 1, 1], {0: float("nan"), 1: float("inf"), 2: 1.25})
        self.assertEqual(select_greedy_action(env), (2, 1.25, 3))

    def test_fails_when_no_actions_are_legal(self):
        env = FakeGreedyEnv([0, 0], {})
        with self.assertRaisesRegex(RuntimeError, "no legal actions"):
            select_greedy_action(env)

    def test_fails_when_all_rewards_are_non_finite(self):
        env = FakeGreedyEnv([1, 1], {0: float("nan"), 1: float("-inf")})
        with self.assertRaisesRegex(RuntimeError, "finite reward"):
            select_greedy_action(env)

    def test_suppresses_display_warnings_during_candidate_preview(self):
        env = NoisyGreedyEnv([1], {0: 2.0})
        logger = logging.getLogger("atena.simulation.display")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.setLevel, original_level)

        with patch.object(logger, "handle") as handle:
            self.assertEqual(select_greedy_action(env), (0, 2.0, 1))

        handle.assert_not_called()
        self.assertEqual(logger.level, logging.INFO)

    def test_restores_display_logger_level_when_candidate_preview_raises(self):
        env = NoisyGreedyEnv([1], {0: 2.0}, preview_error=ValueError("invalid candidate"))
        logger = logging.getLogger("atena.simulation.display")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.setLevel, original_level)

        with patch.object(logger, "handle") as handle:
            with self.assertRaisesRegex(ValueError, "invalid candidate"):
                select_greedy_action(env)

        handle.assert_not_called()
        self.assertEqual(logger.level, logging.INFO)


class FakeSessionEnv(FakeGreedyEnv):
    def __init__(self):
        super().__init__([1, 1], {0: 1.0, 1: 2.0})
        self.actions = []
        self.steps = 0

    def reset(self):
        self.actions = []
        self.steps = 0
        return np.asarray([0.0], dtype=np.float32)

    def step(self, action_index):
        self.actions.append(f"action-{action_index}")
        self.steps += 1
        done = self.steps >= 2
        return np.asarray([self.steps], dtype=np.float32), self._rewards[action_index], done, {}


class GreedyRunnerTests(unittest.TestCase):
    def greedy_args(self, output_dir):
        return SimpleNamespace(
            method="greedy",
            schema="flights",
            dataset_number=1,
            seed=0,
            output_dir=str(output_dir),
        )

    def test_runner_writes_non_learning_result_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "greedy" / "flights1" / "seed0"
            args = self.greedy_args(tmp)
            with patch("atena_baselines.greedy.make_env", return_value=FakeSessionEnv()) as make_env_mock, patch(
                "atena_baselines.greedy.official_metrics",
                return_value={"Precision": 0.1, "T-BLEU-1": 0.2, "T-BLEU-2": 0.3,
                              "T-BLEU-3": 0.4, "EDA-Sim": 0.5},
            ):
                rows = run_greedy(args, result_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "greedy")
            self.assertEqual(rows[0]["steps"], 0)
            self.assertEqual(rows[0]["candidate_evaluations"], 4)
            self.assertGreaterEqual(rows[0]["runtime_seconds"], 0.0)
            self.assertEqual(
                json.loads((result_dir / "actions_steps0.json").read_text()),
                ["'action-1'", "'action-1'"],
            )
            self.assertTrue((result_dir / "eval_metrics.csv").exists())
            self.assertTrue((result_dir / "final_metrics.json").exists())
            self.assertFalse((result_dir / "train_log.csv").exists())
            self.assertFalse(any(result_dir.glob("*.weights.h5")))
            make_env_mock.assert_called_once()

    def test_suppresses_display_warnings_during_official_metrics(self):
        metrics = {
            "Precision": 0.1,
            "T-BLEU-1": 0.2,
            "T-BLEU-2": 0.3,
            "T-BLEU-3": 0.4,
            "EDA-Sim": 0.5,
        }
        logger = logging.getLogger("atena.simulation.display")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.setLevel, original_level)

        def noisy_official_metrics(*_args):
            logger.warning("invalid display during official metrics")
            return metrics

        with tempfile.TemporaryDirectory() as tmp:
            with patch("atena_baselines.greedy.make_env", return_value=FakeSessionEnv()), patch(
                "atena_baselines.greedy.official_metrics", side_effect=noisy_official_metrics
            ), patch.object(logger, "handle") as handle:
                run_greedy(self.greedy_args(tmp), Path(tmp))

        handle.assert_not_called()
        self.assertEqual(logger.level, logging.INFO)

    def test_method_registration_and_dispatch_skip_probe_and_tensorflow_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.greedy_args(tmp)
            self.assertIn("greedy", train_module.METHODS)
            self.assertEqual(train_module.method_reward_mode("greedy"), "official_compound")
            with patch.object(train_module, "run_greedy", return_value=[{"method": "greedy"}]) as runner, patch.object(
                train_module, "make_env"
            ) as make_env_mock:
                rows = train_module.run_experiment(args)
            self.assertEqual(rows, [{"method": "greedy"}])
            runner.assert_called_once()
            make_env_mock.assert_not_called()

    def test_cli_validation_rejects_invalid_worker_count(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            validate_args(SimpleNamespace(workers=0, dataset_number=1))


if __name__ == "__main__":
    unittest.main()
