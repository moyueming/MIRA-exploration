import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_final_cumulative_performance import (
    ALL_METHODS,
    ABLATION_METHODS,
    MAIN_METHODS,
    METRICS,
    OUTPUT_NAMES,
    VARIANTS,
    build_metric_figure,
    load_metric_seed_curves,
    metric_statistics,
    render_final_cumulative_pngs,
)


class FinalCumulativePerformanceTests(unittest.TestCase):
    def test_contract_defines_three_metrics_and_three_method_variants(self):
        self.assertEqual(
            tuple(METRICS),
            (
                "cumulative_extrinsic_reward",
                "cumulative_target_efficiency",
                "cumulative_unique_sets_viewed",
            ),
        )
        self.assertEqual(VARIANTS["main"], MAIN_METHODS)
        self.assertEqual(VARIANTS["ablation"], ABLATION_METHODS)
        self.assertEqual(VARIANTS["all_methods"], ALL_METHODS)
        self.assertEqual(len(ALL_METHODS), 8)

    def test_loader_reads_complete_unsmoothed_seed_curves(self):
        with TemporaryDirectory() as directory:
            paths = []
            episodes = np.arange(1, 1001)
            for seed in range(3):
                path = Path(directory) / f"seed{seed + 1}.csv"
                pd.DataFrame(
                    {
                        "episode": episodes,
                        "cumulative_extrinsic_reward": episodes * (seed + 1),
                    }
                ).to_csv(path, index=False)
                paths.append(path)

            curves = load_metric_seed_curves(
                tuple(paths), "cumulative_extrinsic_reward"
            )

        self.assertEqual(curves.shape, (3, 1000))
        np.testing.assert_array_equal(curves[:, 1], np.array([2.0, 4.0, 6.0]))

    def test_statistics_use_sample_standard_deviation_without_smoothing(self):
        curves = np.array([[0.0, 10.0], [0.0, 20.0], [0.0, 30.0]])
        mean, sample_sd = metric_statistics(curves)
        np.testing.assert_array_equal(mean, np.array([0.0, 20.0]))
        np.testing.assert_array_equal(sample_sd, np.array([0.0, 10.0]))

    def test_unique_sets_uses_post_steps_and_all_lines_are_solid(self):
        fig, ax = build_metric_figure(
            "Galaxy", "cumulative_unique_sets_viewed", MAIN_METHODS
        )
        try:
            method_lines = [line for line in ax.lines if line.get_label() in MAIN_METHODS]
            self.assertEqual(len(method_lines), len(MAIN_METHODS))
            self.assertTrue(all(line.get_drawstyle() == "steps-post" for line in method_lines))
            self.assertTrue(all(line.get_linestyle() == "-" for line in method_lines))
            self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
            self.assertTrue(
                all(text.get_fontweight() == "bold" for text in fig.legends[0].texts)
            )
        finally:
            plt.close(fig)

    def test_renderer_writes_nine_pngs_per_dataset_and_no_pdfs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = render_final_cumulative_pngs(root)

            self.assertEqual(set(outputs), {"Galaxy", "Covertype"})
            for dataset, files in outputs.items():
                self.assertEqual(len(files), 9)
                self.assertEqual(set(files), set(OUTPUT_NAMES[dataset]))
                self.assertTrue(
                    all(path.read_bytes().startswith(b"\x89PNG") for path in files.values())
                )
                self.assertEqual(list(files[next(iter(files))].parent.glob("*.pdf")), [])


if __name__ == "__main__":
    unittest.main()
