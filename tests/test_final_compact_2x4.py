import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_compact_two_dataset_2x4_sample import ALL_METHODS, DISPLAY_LABELS, METRICS
from plot_final_compact_2x4 import (
    OUTPUT_NAMES,
    build_final_compact_figure,
    load_real_compact_data,
    render_final_compact_pngs,
)
from plot_final_cumulative_performance import load_metric_seed_curves
from plot_real_episode_reward_two_datasets import DATASET_FILES


class FinalCompact2x4Tests(unittest.TestCase):
    def test_loader_returns_all_real_methods_metrics_and_seed_shapes(self):
        for dataset in ("Galaxy", "Covertype"):
            data = load_real_compact_data(dataset)
            self.assertEqual(set(data), set(ALL_METHODS))
            for method in ALL_METHODS:
                self.assertEqual(set(data[method]), set(METRICS))
                for curves in data[method].values():
                    self.assertEqual(curves.shape, (3, 1000))

    def test_cumulative_metric_is_loaded_without_smoothing(self):
        data = load_real_compact_data("Galaxy")
        expected = load_metric_seed_curves(
            DATASET_FILES["Galaxy"]["MIRA"], "cumulative_extrinsic_reward"
        )
        np.testing.assert_array_equal(data["MIRA"]["cumulative_reward"], expected)

    def test_target_efficiency_is_smoothed_per_seed_with_dataset_window(self):
        for dataset, window in (("Galaxy", 25), ("Covertype", 50)):
            data = load_real_compact_data(dataset)
            raw = []
            for path in DATASET_FILES[dataset]["MIRA"]:
                frame = pd.read_csv(path).sort_values("episode").set_index("episode")
                raw.append(frame["target_efficiency"].reindex(range(1, 1001)).to_numpy(dtype=float))
            expected = np.vstack(
                [
                    pd.Series(seed)
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .to_numpy()
                    for seed in raw
                ]
            )
            np.testing.assert_allclose(data["MIRA"]["target_efficiency"], expected)

    def test_final_figure_preserves_approved_two_by_four_layout(self):
        fig, axes = build_final_compact_figure("Galaxy")
        try:
            self.assertEqual(axes.shape, (2, 4))
            for column in range(4):
                self.assertEqual(
                    axes[0, column].get_ylim(), axes[1, column].get_ylim()
                )
            for row in range(2):
                for column in range(4):
                    self.assertEqual(axes[row, column].get_ylabel(), "")
                    self.assertEqual(axes[row, column].get_xlabel(), "Episode")
                    self.assertEqual(list(axes[row, column].get_xticks()), [0, 250, 500, 750, 1000])
            self.assertEqual(len(fig.legends), 1)
            self.assertEqual(
                [text.get_text() for text in fig.legends[0].texts],
                [DISPLAY_LABELS[method] for method in ALL_METHODS],
            )
        finally:
            plt.close(fig)

    def test_renderer_writes_one_png_per_dataset_and_no_pdfs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = render_final_compact_pngs(root)

            self.assertEqual(set(outputs), {"Galaxy", "Covertype"})
            for dataset, path in outputs.items():
                self.assertEqual(path.name, OUTPUT_NAMES[dataset])
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
                self.assertEqual(list(path.parent.glob("*.pdf")), [])


if __name__ == "__main__":
    unittest.main()
