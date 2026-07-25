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


from recalculate_baseline_corpus_metrics import runtime


class BaselineRuntimeTests(unittest.TestCase):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def numpy(self):
            return self.values

    class FakeModel:
        loaded_path = None

        def __init__(self, state_dim, action_dim, hidden):
            self.shape = (state_dim, action_dim, hidden)

        def load_weights(self, path):
            self.loaded_path = path

        def __call__(self, _state, training=False):
            return (
                BaselineRuntimeTests.FakeTensor([[0.1, 0.9]]),
                BaselineRuntimeTests.FakeTensor([[0.0]]),
            )

    class FakeEnv:
        state_dim = 3
        action_dim = 2

        def __init__(self):
            self.actions = []

        def reset(self):
            return [0.0, 0.0, 0.0]

        def legal_action_mask(self):
            return [1.0, 1.0]

        def step(self, action):
            self.actions.append(f"action-{action}")
            return [0.0, 0.0, 0.0], 0.0, len(self.actions) == 12, {}

    def test_reconstruct_policy_uses_final_weights_and_evaluation_seed(self):
        calls = []

        def env_factory(schema, dataset, seed, args, reward_mode):
            calls.append((schema, dataset, seed, args.method, reward_mode))
            return self.FakeEnv()

        config = {
            "schema": "cyber",
            "dataset_number": 2,
            "seed": 0,
            "method": "pure_a3c",
            "hidden": 256,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (result_dir / "policy.weights.h5").write_bytes(b"weights")

            actions = runtime.reconstruct_policy_session(
                result_dir,
                env_factory=env_factory,
                model_factory=self.FakeModel,
            )

        self.assertEqual(len(actions), 12)
        self.assertEqual(calls[0][2], 777)
        self.assertTrue(runtime.LAST_LOADED_WEIGHT_PATH.endswith("policy.weights.h5"))

    def test_reconstruct_mira_maps_legacy_config_and_uses_formal_weights(self):
        calls = []

        def env_factory(schema, dataset, seed, args):
            calls.append((
                schema,
                dataset,
                seed,
                args.avp,
                args.w_column_coverage,
                args.w_group_coverage,
                args.w_structure,
            ))
            return self.FakeEnv()

        config = {
            "schema": "flights",
            "dataset_number": 3,
            "seed": 0,
            "method": "mira_v5_swa_final",
            "hidden": 256,
            "official_reference_terms": True,
            "w_v3_column_coverage": 0.35,
            "w_v3_group_coverage": 0.30,
            "w_v3_structure": 0.25,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (result_dir / "policy.weights.h5").write_bytes(b"weights")
            (result_dir / "actions_steps1000008.json").write_text(
                json.dumps(["action-1"] * 12),
                encoding="utf-8",
            )

            actions = runtime.reconstruct_mira_session(
                result_dir,
                env_factory=env_factory,
                model_factory=self.FakeModel,
            )

        self.assertEqual(len(actions), 12)
        self.assertEqual(
            calls,
            [("flights", 3, 777, "1", 0.35, 0.30, 0.25)],
        )
        self.assertTrue(runtime.LAST_LOADED_WEIGHT_PATH.endswith("policy.weights.h5"))

    def test_reconstruct_greedy_does_not_write_result_files(self):
        env = self.FakeEnv()
        config = {
            "schema": "flights",
            "dataset_number": 1,
            "seed": 0,
            "method": "greedy",
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            before = set(result_dir.iterdir())

            actions = runtime.reconstruct_greedy_session(
                result_dir,
                env_factory=lambda *_args, **_kwargs: env,
                action_selector=lambda _env: (1, 0.0, 1),
            )

            after = set(result_dir.iterdir())

        self.assertEqual(len(actions), 12)
        self.assertEqual(before, after)

    def test_reconstruct_official_reads_raw_actions(self):
        payload = [{"raw_action": [0.0] * 6}] * 12
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            source = results / "official_atena_eval" / "cyber" / "dataset1" / "seed0"
            source.mkdir(parents=True)
            (source / "raw_actions.json").write_text(json.dumps(payload), encoding="utf-8")

            actions = runtime.reconstruct_official_session(
                results,
                "cyber",
                1,
                seed=0,
                meta_factory=lambda schema, dataset: (schema, dataset),
                converter=lambda meta, raw: [f"{meta}-{index}" for index, _ in enumerate(raw)],
            )

        self.assertEqual(len(actions), 12)
        self.assertEqual(actions[0], "('cyber', 1)-0")

    def test_write_outputs_creates_new_audit_files(self):
        detail = [{"method": "official_atena", "schema": "cyber", "dataset": 1}]
        summary = [{"method": "official_atena", "Precision": 0.1}]
        manifest = {"validation_passed": True, "random_k": 16}

        with tempfile.TemporaryDirectory() as tmp:
            paths = runtime.write_outputs(Path(tmp), detail, summary, manifest)

            self.assertEqual(len(paths), 3)
            self.assertEqual(
                [path.name for path in paths],
                [
                    "all_methods_official_corpus_detail.csv",
                    "all_methods_official_corpus_summary.csv",
                    "all_methods_official_corpus_manifest.json",
                ],
            )
            self.assertTrue(all(path.exists() for path in paths))
            loaded = json.loads(paths[2].read_text(encoding="utf-8"))

        self.assertTrue(loaded["validation_passed"])


if __name__ == "__main__":
    unittest.main()
