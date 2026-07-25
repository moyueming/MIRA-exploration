import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from plot_final_episode_reward_by_dataset import (
    build_combined_figure,
    render_final_pngs,
)


class FinalEpisodeRewardTests(unittest.TestCase):
    def test_combined_panels_share_y_limits(self):
        for dataset in ("Galaxy", "Covertype"):
            fig, axes = build_combined_figure(dataset)
            try:
                self.assertEqual(axes[0].get_ylim(), axes[1].get_ylim())
                self.assertEqual(
                    [ax.get_title() for ax in axes],
                    ["(a) Main comparison", "(b) Ablation"],
                )
            finally:
                plt.close(fig)

    def test_renderer_writes_three_pngs_per_dataset_and_no_pdfs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = render_final_pngs(root)

            self.assertEqual(set(outputs), {"Galaxy", "Covertype"})
            for dataset, files in outputs.items():
                self.assertEqual(set(files), {"main", "ablation", "combined"})
                folder_name = (
                    "galaxy_final" if dataset == "Galaxy" else "covertype_final"
                )
                folder = root / folder_name
                self.assertEqual(set(folder.glob("*.png")), set(files.values()))
                self.assertEqual(list(folder.glob("*.pdf")), [])
                self.assertTrue(
                    all(path.read_bytes().startswith(b"\x89PNG") for path in files.values())
                )


if __name__ == "__main__":
    unittest.main()
