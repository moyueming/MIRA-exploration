import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERTYPE_ROOT = ROOT / "covertype-exploration"
FULL_A3C = COVERTYPE_ROOT / "covertype_rl" / "full_a3c.py"
STALE_BASELINE = "ours" + "_bile"


def literal_assignment(module, name):
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment: {name}")


class CovertypeMiraNamingTests(unittest.TestCase):
    def test_runtime_uses_public_mira_baseline_names(self):
        source = FULL_A3C.read_text(encoding="utf-8")
        module = ast.parse(source)

        self.assertEqual(literal_assignment(module, "MIRA_BASELINES"), {"mira", "mira_no_ext"})
        self.assertNotIn(STALE_BASELINE, source)

    def test_wrappers_use_public_names(self):
        expected = {
            "mira": 'args.baseline = "mira"',
            "mira_no_ext": 'args.baseline = "mira_no_ext"',
        }

        for directory, assignment in expected.items():
            wrapper = COVERTYPE_ROOT / "baselines" / directory / "run.py"
            self.assertTrue(wrapper.is_file())
            self.assertIn(assignment, wrapper.read_text(encoding="utf-8"))

        self.assertFalse((COVERTYPE_ROOT / "baselines" / STALE_BASELINE).exists())
        self.assertFalse((COVERTYPE_ROOT / "baselines" / (STALE_BASELINE + "_no_ext")).exists())


if __name__ == "__main__":
    unittest.main()
