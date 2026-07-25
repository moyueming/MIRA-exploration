import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from plot_remaining_four_forms_sample import build_sample_figure, render_sample


class RemainingFourFormsTests(unittest.TestCase):
    def test_four_panels_use_the_approved_forms(self):
        fig, axes = build_sample_figure()
        try:
            flat = list(axes.flat)
            self.assertEqual(len(flat), 4)
            self.assertEqual(
                [ax.get_title() for ax in flat],
                [
                    "(a) Cumulative reward",
                    "(b) Unique-set discovery",
                    "(c) Last-200 efficiency",
                    "(d) Reward-coverage trade-off",
                ],
            )
            self.assertTrue(
                all(line.get_drawstyle() == "steps-post" for line in flat[1].lines)
            )
            self.assertEqual(flat[2].get_xlabel(), "Last-200 Target Efficiency")
            self.assertEqual(flat[3].get_xlabel(), "Cumulative Unique Sets")
            self.assertEqual(flat[3].get_ylabel(), "Cumulative Reward")
        finally:
            plt.close(fig)

    def test_renderer_writes_only_the_png(self):
        with TemporaryDirectory() as directory:
            output = render_sample(Path(directory))

            self.assertEqual(output.name, "remaining_four_figure_forms_sample.png")
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(list(Path(directory).glob("*.pdf")), [])


if __name__ == "__main__":
    unittest.main()
