import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from plot_galaxy_extrinsic_reward_template import ABLATION_METHODS, MAIN_METHODS
from plot_real_episode_reward_two_datasets import (
    DATASET_FILES,
    build_two_dataset_figure,
    load_seed_curves,
    render_real_figures,
)


class RealEpisodeRewardTests(unittest.TestCase):
    def test_all_real_seed_files_load_as_three_complete_curves(self):
        expected_methods = set(MAIN_METHODS) | set(ABLATION_METHODS)
        self.assertEqual(set(DATASET_FILES), {"Galaxy", "Covertype"})

        for methods in DATASET_FILES.values():
            self.assertIn("Greedy", methods)
            self.assertEqual(set(methods), expected_methods)
            for paths in methods.values():
                self.assertEqual(len(paths), 3)
                self.assertTrue(all(path.exists() for path in paths))
                self.assertEqual(load_seed_curves(paths).shape, (3, 1000))

    def test_atena_labels_match_reward_configuration(self):
        self.assertTrue(all("ATENA_ext" in path.parts for path in DATASET_FILES["Galaxy"]["ATENA"]))
        self.assertTrue(all("ATENA_pure" in path.parts for path in DATASET_FILES["Galaxy"]["ATENA w/o Ext. Reward"]))
        self.assertTrue(all("ATENA-EXT" in path.parts for path in DATASET_FILES["Covertype"]["ATENA"]))
        self.assertTrue(all("ATENA" in path.parts for path in DATASET_FILES["Covertype"]["ATENA w/o Ext. Reward"]))

    def test_main_figure_has_two_boxed_panels_bands_and_shared_legend(self):
        fig, axes = build_two_dataset_figure(MAIN_METHODS)
        try:
            self.assertEqual(len(axes), 2)
            self.assertEqual([ax.get_title() for ax in axes], ["Galaxy", "Covertype"])
            for ax in axes:
                self.assertEqual(ax.get_ylabel(), "Episode Reward")
                self.assertEqual(len(ax.lines), len(MAIN_METHODS))
                self.assertEqual(len(ax.collections), len(MAIN_METHODS))
                self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
            self.assertEqual(
                [text.get_text() for text in fig.legends[0].get_texts()],
                list(MAIN_METHODS),
            )
        finally:
            plt.close(fig)

    def test_renderer_writes_main_and_ablation_png_pdf_pairs(self):
        with TemporaryDirectory() as directory:
            outputs = render_real_figures(Path(directory))

            self.assertEqual(set(outputs), {"main", "ablation"})
            for png_path, pdf_path in outputs.values():
                self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG"))
                self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
