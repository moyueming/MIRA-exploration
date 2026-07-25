import unittest

import matplotlib.pyplot as plt

from plot_galaxy_extrinsic_reward_template import MAIN_METHODS, METHOD_STYLES
from plot_real_episode_reward_two_datasets import (
    SMOOTHING_WINDOWS,
    build_two_dataset_figure,
)


class EpisodeRewardSmoothingFixTests(unittest.TestCase):
    def test_windows_and_all_line_styles_match_the_final_contract(self):
        self.assertEqual(SMOOTHING_WINDOWS, {"Galaxy": 25, "Covertype": 50})
        self.assertTrue(all(style[1] == "-" for style in METHOD_STYLES.values()))

        fig, axes = build_two_dataset_figure(MAIN_METHODS)
        try:
            self.assertTrue(
                all(line.get_linestyle() == "-" for ax in axes for line in ax.lines)
            )
            self.assertTrue(
                all(
                    handle.get_linestyle() == "-"
                    for handle in fig.legends[0].legend_handles
                )
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
