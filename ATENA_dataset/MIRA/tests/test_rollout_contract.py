import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira import train as train_module
from mira.rollout_pool import PersistentRolloutExecutor
from run import parse_args


class FakePool:
    def __init__(self):
        self.maps = []
        self.closed = 0
        self.terminated = 0
        self.joined = 0

    def map(self, function, payloads):
        payloads = list(payloads)
        self.maps.append((function, payloads))
        return [function(payload) for payload in payloads]

    def close(self):
        self.closed += 1

    def terminate(self):
        self.terminated += 1

    def join(self):
        self.joined += 1


class FakeContext:
    def __init__(self):
        self.pool = FakePool()
        self.calls = []

    def Pool(self, **kwargs):
        self.calls.append(kwargs)
        return self.pool


class RolloutContractTests(unittest.TestCase):
    def test_worker_seed_formula_and_payload_order_are_frozen(self):
        args = parse_args([
            "--schema", "cyber",
            "--dataset_number", "1",
            "--workers", "28",
        ])
        rng = np.random.default_rng(5)
        payloads = train_module.build_worker_args(
            args=args,
            runtime_args=args,
            policy_weights=[np.asarray([1.0])],
            mira_weights={"encoder": [np.asarray([2.0])]},
            update_idx=7,
            rollout_count=28,
            rng=rng,
            direction_sampler=lambda _rng, _dim: np.asarray([1.0]),
        )

        self.assertEqual([payload[3] for payload in payloads], list(range(196, 224)))

    def test_executor_reuses_one_ordered_spawn_pool(self):
        context = FakeContext()
        context_methods = []

        def context_factory(method):
            context_methods.append(method)
            return context

        executor = PersistentRolloutExecutor(
            workers=28,
            worker_function=str,
            context_factory=context_factory,
        )
        self.assertEqual(executor.map([3, 1, 2]), ["3", "1", "2"])
        self.assertEqual(executor.map([5, 4]), ["5", "4"])
        executor.close()

        self.assertEqual(context_methods, ["spawn"])
        self.assertEqual(len(context.calls), 1)
        self.assertEqual(context.calls[0]["processes"], 3)
        self.assertEqual(context.calls[0]["maxtasksperchild"], 64)
        self.assertEqual(context.pool.closed, 1)
        self.assertEqual(context.pool.terminated, 0)
        self.assertEqual(context.pool.joined, 1)

    def test_executor_terminates_pool_on_exception(self):
        context = FakeContext()
        executor = PersistentRolloutExecutor(
            workers=2,
            worker_function=str,
            context_factory=lambda _method: context,
        )
        executor.map([1, 2])

        executor.__exit__(RuntimeError, RuntimeError("failed"), None)

        self.assertEqual(context.pool.closed, 0)
        self.assertEqual(context.pool.terminated, 1)
        self.assertEqual(context.pool.joined, 1)

    def test_formal_final_row_is_always_last(self):
        rows = [{"steps": 10, "score": 99.0}, {"steps": 20, "score": 1.0}]

        self.assertEqual(train_module.formal_final_row(rows), rows[-1])
        self.assertIsNone(train_module.formal_final_row([]))


if __name__ == "__main__":
    unittest.main()
