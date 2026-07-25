import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from plot_galaxy_extrinsic_reward_template import (
    ABLATION_METHODS,
    MAIN_METHODS,
    build_figure,
    generate_synthetic_curves,
    render_templates,
)


class EpisodeRewardV2Tests(unittest.TestCase):
    def test_method_groups_use_final_labels(self):
        self.assertEqual(
            MAIN_METHODS,
            ("MIRA", "DORA", "Greedy", "ATENA", "A3C", "Random"),
        )
        self.assertEqual(
            ABLATION_METHODS,
            ("MIRA", "MIRA w/o Ext. Reward", "ATENA", "ATENA w/o Ext. Reward"),
        )

    def test_figure_uses_sd_bands_boxed_axes_and_episode_reward_label(self):
        episodes = np.arange(1, 1001)
        curves = generate_synthetic_curves(episodes)
        fig, ax = build_figure(MAIN_METHODS, curves)
        try:
            self.assertEqual(ax.get_ylabel(), "Episode Reward")
            self.assertEqual(len(ax.lines), len(MAIN_METHODS))
            self.assertEqual(len(ax.collections), len(MAIN_METHODS))
            self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
            self.assertTrue(
                all(
                    spine.get_edgecolor()[:3] == (0.0, 0.0, 0.0)
                    for spine in ax.spines.values()
                )
            )
            self.assertLess(max(line.get_linewidth() for line in ax.lines), 1.6)
            legend = ax.get_legend()
            self.assertTrue(
                all(text.get_fontweight() == "bold" for text in legend.get_texts())
            )
            self.assertTrue(
                all(handle.get_linewidth() >= 2.5 for handle in legend.legend_handles)
            )
        finally:
            plt.close(fig)

    def test_renderer_writes_main_and_ablation_png_pdf_pairs(self):
        with TemporaryDirectory() as directory:
            outputs = render_templates(Path(directory))

            self.assertEqual(set(outputs), {"main", "ablation"})
            for png_path, pdf_path in outputs.values():
                self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG"))
                self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
