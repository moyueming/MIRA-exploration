import unittest

from final_figure_registry import (
    ABLATION_METHODS,
    ALL_METHODS,
    DISPLAY_LABELS,
    GREEDY_EDA,
    GREEDY_REWARD_FILES,
    GREEDY_TRACE_FILES,
    MAIN_METHODS,
    METHOD_MARKERS,
    METHOD_STYLES,
)


class GreedyFinalFigureRegistryTests(unittest.TestCase):
    def test_greedy_is_main_and_all_but_not_ablation(self):
        self.assertEqual(GREEDY_EDA, "Greedy")
        self.assertEqual(
            MAIN_METHODS,
            ("MIRA", "DORA", "Greedy", "ATENA", "A3C", "Random"),
        )
        self.assertEqual(
            ALL_METHODS,
            (
                "MIRA",
                "MIRA w/o Ext. Reward",
                "DORA",
                "Greedy",
                "ATENA",
                "ATENA w/o Ext. Reward",
                "A3C",
                "Random",
            ),
        )
        self.assertNotIn(GREEDY_EDA, ABLATION_METHODS)
        self.assertEqual(METHOD_STYLES[GREEDY_EDA], ("#7B3294", "-", 1.15))
        self.assertEqual(METHOD_MARKERS[GREEDY_EDA], "H")
        self.assertEqual(DISPLAY_LABELS[GREEDY_EDA], "Greedy")

    def test_all_greedy_inputs_exist_as_three_seed_groups(self):
        for dataset in ("Galaxy", "Covertype"):
            rewards = GREEDY_REWARD_FILES[dataset]
            traces = GREEDY_TRACE_FILES[dataset]
            self.assertEqual(len(rewards), 3)
            self.assertEqual(len(traces), 3)
            self.assertTrue(all(path.exists() for path in rewards + traces))


if __name__ == "__main__":
    unittest.main()
