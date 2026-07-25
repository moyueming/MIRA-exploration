import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira import env as standalone_env


class RewardContractTests(unittest.TestCase):
    def test_mira_reward_formula_is_frozen(self):
        env = object.__new__(standalone_env.AtenaEDAEnv)
        env.w_kl = 1.5
        env.w_compaction = 2.0
        env.w_official_diversity = 3.0
        env.w_column_coverage = 0.35
        env.w_group_coverage = 0.30
        env.w_structure = 0.25

        reward = env._combine_reward({
            "official_kl": 0.2,
            "official_compaction": 0.4,
            "official_diversity": 0.1,
            "column_coverage": 1.0,
            "group_coverage": 0.5,
            "structure": -1.0,
        })

        self.assertAlmostEqual(reward, 1.65)


if __name__ == "__main__":
    unittest.main()
