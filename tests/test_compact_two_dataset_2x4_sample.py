import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from plot_compact_two_dataset_2x4_sample import (
    ABLATION_METHODS,
    ALL_METHODS,
    DISPLAY_LABELS,
    MAIN_METHODS,
    METRICS,
    build_compact_sample_figure,
    render_compact_samples,
)


class CompactTwoDatasetSampleTests(unittest.TestCase):
    def test_figure_is_two_by_four_with_shared_column_limits(self):
        fig, axes = build_compact_sample_figure("Galaxy")
        try:
            self.assertEqual(tuple(METRICS), ("cumulative_reward", "cumulative_target_efficiency", "cumulative_unique_sets", "target_efficiency"))
            self.assertEqual(axes.shape, (2, 4))
            self.assertEqual(
                [ax.get_title() for ax in axes[0]],
                [f"({chr(ord('a') + column)}) {config['title']}" for column, config in enumerate(METRICS.values())],
            )
            self.assertEqual(
                [ax.get_title() for ax in axes[1]],
                [f"({chr(ord('e') + column)}) {config['title']}" for column, config in enumerate(METRICS.values())],
            )
            for column, config in enumerate(METRICS.values()):
                self.assertEqual(
                    axes[0, column].get_ylim(), axes[1, column].get_ylim()
                )
                self.assertEqual(axes[0, column].get_ylabel(), "")
                self.assertEqual(axes[1, column].get_ylabel(), "")
                for row in range(2):
                    self.assertEqual(axes[row, column].get_xlabel(), "Episode")
                    self.assertEqual(list(axes[row, column].get_xticks()), [0, 250, 500, 750, 1000])
            self.assertNotIn("Main comparison", [text.get_text() for text in fig.texts])
            self.assertNotIn("Ablation", [text.get_text() for text in fig.texts])
        finally:
            plt.close(fig)

    def test_all_content_stays_inside_canvas(self):
        fig, _ = build_compact_sample_figure("Galaxy")
        try:
            fig.canvas.draw()
            canvas = fig.bbox
            tight = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
            self.assertGreaterEqual(tight.x0, canvas.x0)
            self.assertGreaterEqual(tight.y0, canvas.y0)
            self.assertLessEqual(tight.x1, canvas.x1)
            self.assertLessEqual(tight.y1, canvas.y1)
        finally:
            plt.close(fig)

    def test_rows_use_approved_method_groups_and_solid_lines(self):
        fig, axes = build_compact_sample_figure("Covertype")
        try:
            for ax in axes[0]:
                lines = [line for line in ax.lines if line.get_label() in MAIN_METHODS]
                self.assertEqual([line.get_label() for line in lines], list(MAIN_METHODS))
                self.assertTrue(all(line.get_linestyle() == "-" for line in lines))
            for ax in axes[1]:
                lines = [line for line in ax.lines if line.get_label() in ABLATION_METHODS]
                self.assertEqual(
                    [line.get_label() for line in lines], list(ABLATION_METHODS)
                )
                self.assertTrue(all(line.get_linestyle() == "-" for line in lines))
        finally:
            plt.close(fig)

    def test_unique_sets_is_post_step_and_legend_contains_all_methods(self):
        fig, axes = build_compact_sample_figure("Galaxy")
        try:
            self.assertEqual(tuple(METRICS), ("cumulative_reward", "cumulative_target_efficiency", "cumulative_unique_sets", "target_efficiency"))
            for row in range(2):
                method_lines = [line for line in axes[row, 2].lines if not line.get_label().startswith("_")]
                self.assertTrue(
                    all(line.get_drawstyle() == "steps-post" for line in method_lines)
                )
            self.assertEqual(len(fig.legends), 1)
            self.assertEqual(
                [text.get_text() for text in fig.legends[0].texts],
                [DISPLAY_LABELS[method] for method in ALL_METHODS],
            )
            self.assertTrue(
                all(text.get_fontweight() == "bold" for text in fig.legends[0].texts)
            )
        finally:
            plt.close(fig)

    def test_renderer_writes_two_pngs_and_no_pdfs(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            outputs = render_compact_samples(output_dir)

            self.assertEqual(set(outputs), {"Galaxy", "Covertype"})
            self.assertEqual(
                {path.name for path in outputs.values()},
                {
                    "galaxy_compact_2x4_sample.png",
                    "covertype_compact_2x4_sample.png",
                },
            )
            self.assertTrue(
                all(path.read_bytes().startswith(b"\x89PNG") for path in outputs.values())
            )
            self.assertEqual(list(output_dir.glob("*.pdf")), [])


if __name__ == "__main__":
    unittest.main()
