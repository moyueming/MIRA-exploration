import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira.train import _validate_runtime
from run import parse_args


class StandaloneBoundaryTests(unittest.TestCase):
    def test_cli_has_frozen_mira_defaults(self):
        args = parse_args(["--schema", "cyber", "--dataset_number", "3"])

        self.assertEqual(args.method, "MIRA")
        self.assertEqual(args.avp, "0")
        self.assertFalse(hasattr(args, "official_reference_terms"))
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.workers, 16)
        self.assertEqual(args.steps, 1_000_000)
        self.assertEqual(args.episode_length, 12)
        self.assertEqual(args.output_dir, "results")

    def test_clean_cli_is_accepted_by_runtime_validation(self):
        args = parse_args(["--schema", "cyber", "--dataset_number", "3"])

        _validate_runtime(args)

    def test_runtime_source_has_no_historical_reward_identifiers(self):
        paths = [ROOT / "run.py"] + list((ROOT / "mira").glob("*.py"))
        source = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)

        for forbidden in (
            "v" + "3",
            "v" + "4",
            "do" + "ra",
            "coher" + "ency",
            "w_" + "interestingness",
            "reward_" + "mode",
        ):
            self.assertNotIn(forbidden, source, forbidden)

    def test_production_modules_do_not_import_baseline_package(self):
        paths = [ROOT / "run.py"] + list((ROOT / "mira").glob("*.py"))
        self.assertTrue(all(path.exists() for path in paths))
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                any(
                    name == "atena_baselines" or name.startswith("atena_baselines.")
                    for name in imported
                ),
                path,
            )


if __name__ == "__main__":
    unittest.main()
