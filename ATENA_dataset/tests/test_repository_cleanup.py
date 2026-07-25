import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryCleanupTests(unittest.TestCase):
    def test_formal_mira_has_final_layout_and_identity(self):
        formal_root = ROOT / "MIRA"
        self.assertTrue((formal_root / "mira" / "__init__.py").is_file())
        self.assertTrue((formal_root / "run.py").is_file())
        self.assertFalse((ROOT / ("MIRA_" + "v" + "5")).exists())

        sys.path.insert(0, str(formal_root))
        try:
            package = importlib.import_module("mira")
            self.assertEqual(package.METHOD, "MIRA")
        finally:
            sys.path.remove(str(formal_root))
            sys.modules.pop("mira", None)

    def test_obsolete_runner_trees_are_absent(self):
        self.assertTrue((ROOT / "atena_baselines").is_dir())
        self.assertTrue((ROOT / "run_atena_baselines.py").is_file())
        self.assertFalse((ROOT / ("mira" + "_aeda")).exists())
        self.assertFalse((ROOT / ("run_mira" + "_aeda.py")).exists())

    def test_active_baseline_package_is_minimal(self):
        expected = {
            "__init__.py",
            "env.py",
            "evaluate.py",
            "greedy.py",
            "models.py",
            "rollout_pool.py",
            "selection.py",
            "train.py",
        }
        observed = {path.name for path in (ROOT / "atena_baselines").glob("*.py")}
        self.assertEqual(observed, expected)

    def test_root_test_suite_contains_only_active_contracts(self):
        expected = {
            "test_baseline_runner.py",
            "test_greedy_baseline.py",
            "test_recalculate_baseline_corpus_metrics.py",
            "test_recalculate_baseline_engine.py",
            "test_recalculate_baseline_orchestrator.py",
            "test_recalculate_baseline_runtime.py",
            "test_recalculate_baseline_sessions.py",
            "test_recalculate_legacy_pandas_compat.py",
            "test_recalculate_numpy_policy.py",
            "test_repository_cleanup.py",
        }
        observed = {path.name for path in (ROOT / "tests").glob("test_*.py")}
        self.assertEqual(observed, expected)

    def test_script_entrypoints_are_canonical(self):
        expected = {
            "build_final_results.py",
            "check_env.py",
            "evaluate_official_atena.py",
            "evaluate_official_atena_batch.sh",
            "recalculate_baselines.py",
            "run_baselines.sh",
            "run_official_atena_template.sh",
            "setup_official_atena_venv.sh",
            "setup_venv.sh",
            "summarize_results.py",
        }
        scripts = ROOT / "scripts"
        observed = {path.name for path in scripts.iterdir() if path.is_file()}
        self.assertEqual(observed, expected)

    def test_current_repository_has_no_obsolete_identifiers(self):
        forbidden = ("v" + "5", "v" + "8", "mira" + "_aeda")
        suffixes = {".py", ".sh", ".ps1", ".md", ".txt"}
        matches = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(ROOT)
            if relative.parts[0] in {"results", "ATENA-A-EDA"}:
                continue
            source = path.read_text(encoding="utf-8", errors="replace").lower()
            for marker in forbidden:
                if marker in source or marker in path.name.lower():
                    matches.append("{}: {}".format(relative, marker))
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
