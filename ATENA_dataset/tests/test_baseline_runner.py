import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atena_baselines.train import METHODS
from run_atena_baselines import parse_args


class BaselineRunnerTests(unittest.TestCase):
    def test_only_four_required_baselines_are_registered(self):
        self.assertEqual(METHODS, {"random", "greedy", "dora", "pure_a3c"})

    def test_cli_defaults_and_method_identity(self):
        args = parse_args([
            "--method", "pure_a3c",
            "--schema", "cyber",
            "--dataset_number", "1",
        ])

        self.assertEqual(args.method, "pure_a3c")
        self.assertEqual(args.workers, 16)
        self.assertEqual(args.steps, 1_000_000)
        self.assertEqual(args.episode_length, 12)
        self.assertEqual(args.output_dir, "results")

    def test_official_tokenizer_has_legacy_pandas_iterator(self):
        self.assertTrue(hasattr(pd.Series, "iteritems"))

    def test_baseline_sources_have_no_removed_method_paths(self):
        source = "\n".join(
            (ROOT / "atena_baselines" / name).read_text(encoding="utf-8").lower()
            for name in ("env.py", "models.py", "train.py")
        )
        for marker in (
            "v" + "5",
            "v" + "8",
            "bc_" + "candidate",
            "mirametric" + "module",
            "official_compound_" + "v3",
            "reference_free_" + "session_task",
        ):
            self.assertNotIn(marker, source, marker)
        self.assertFalse((ROOT / "atena_baselines" / "methods.py").exists())

    def test_removed_method_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args([
                "--method", "mira_" + "v" + "8_reference_free",
                "--schema", "cyber",
                "--dataset_number", "1",
            ])


if __name__ == "__main__":
    unittest.main()
