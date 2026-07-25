import unittest

import matplotlib.pyplot as plt

from final_figure_registry import (
    ABLATION_METHODS,
    ALL_METHODS,
    DISPLAY_LABELS,
    GREEDY_EDA,
    MAIN_METHODS,
    METHOD_STYLES,
)
from plot_final_compact_2x4 import build_final_compact_figure
from plot_final_cumulative_performance import VARIANTS
from plot_final_episode_reward_by_dataset import build_combined_figure


class GreedyMainFigureMembershipTests(unittest.TestCase):
    def test_cumulative_variants_use_shared_method_groups(self):
        self.assertEqual(VARIANTS["main"], MAIN_METHODS)
        self.assertEqual(VARIANTS["ablation"], ABLATION_METHODS)
        self.assertEqual(VARIANTS["all_methods"], ALL_METHODS)

    def test_compact_adds_greedy_only_to_main_row(self):
        fig, axes = build_final_compact_figure("Galaxy")
        try:
            for ax in axes[0]:
                self.assertEqual(
                    [line.get_label() for line in ax.lines], list(MAIN_METHODS)
                )
                greedy = next(
                    line for line in ax.lines if line.get_label() == GREEDY_EDA
                )
                self.assertEqual(greedy.get_color().upper(), "#7B3294")
                self.assertEqual(greedy.get_linestyle(), "-")
            for ax in axes[1]:
                self.assertEqual(
                    [line.get_label() for line in ax.lines], list(ABLATION_METHODS)
                )
            self.assertEqual(
                [text.get_text() for text in fig.legends[0].texts],
                [DISPLAY_LABELS[method] for method in ALL_METHODS],
            )
        finally:
            plt.close(fig)

    def test_episode_combined_adds_greedy_only_to_left_panel(self):
        fig, axes = build_combined_figure("Galaxy")
        try:
            self.assertEqual(
                [line.get_label() for line in axes[0].lines], list(MAIN_METHODS)
            )
            self.assertEqual(
                [line.get_label() for line in axes[1].lines],
                list(ABLATION_METHODS),
            )
            greedy = next(
                line for line in axes[0].lines if line.get_label() == GREEDY_EDA
            )
            self.assertEqual(greedy.get_color().upper(), "#7B3294")
            self.assertEqual(greedy.get_linestyle(), METHOD_STYLES[GREEDY_EDA][1])
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
