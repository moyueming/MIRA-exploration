import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import parse_args


class RewardDefaultsTests(unittest.TestCase):
    def test_only_active_mira_reward_defaults_are_present(self):
        args = parse_args(["--schema", "cyber", "--dataset_number", "1"])

        self.assertEqual(args.avp, "0")
        self.assertAlmostEqual(args.w_kl, 1.5)
        self.assertAlmostEqual(args.w_compaction, 2.0)
        self.assertAlmostEqual(args.w_official_diversity, 2.0)
        self.assertAlmostEqual(args.w_column_coverage, 0.35)
        self.assertAlmostEqual(args.w_group_coverage, 0.30)
        self.assertAlmostEqual(args.w_structure, 0.25)
        for removed_name in (
            "w_" + "interestingness",
            "w_" + "diversity",
            "w_" + "coher" + "ency",
            "reward_" + "mode",
        ):
            self.assertFalse(hasattr(args, removed_name), removed_name)

    def test_avp_accepts_exact_one_and_arbitrary_disabled_values(self):
        for value in ("1", "0", "2", "abc", ""):
            with self.subTest(value=value):
                args = parse_args([
                    "--schema", "cyber", "--dataset_number", "1", "--avp", value,
                ])
                self.assertEqual(args.avp, value)


if __name__ == "__main__":
    unittest.main()
