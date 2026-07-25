from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from export_operator_distribution_vldb import (
    covertype_methods,
    galaxy_methods,
    plot_operator_distribution,
)
from final_figure_registry import ALL_METHODS, GREEDY_TRACE_FILES


class GreedyEdaOperatorDistributionTests(unittest.TestCase):
    def test_both_datasets_include_greedy_in_all_method_order(self) -> None:
        for dataset, methods in (
            ("Galaxy", galaxy_methods()),
            ("Covertype", covertype_methods()),
        ):
            self.assertEqual(
                [method["display_label"] for method in methods],
                list(ALL_METHODS),
            )
            greedy = next(
                method for method in methods if method["display_label"] == "Greedy"
            )
            self.assertEqual(
                tuple(greedy["trace_files"]),
                GREEDY_TRACE_FILES[dataset],
            )

    def test_plot_can_write_png_without_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.csv"
            pd.DataFrame(
                {
                    "operator": [
                        "by_facet",
                        "by_superset",
                        "by_neighbors",
                        "by_distribution",
                    ]
                }
            ).to_csv(trace, index=False)

            png, pdf = plot_operator_distribution(
                [{"display_label": "Greedy", "trace_files": [trace]}],
                root / "operator_distribution",
                write_pdf=False,
            )

            self.assertTrue(png.exists())
            self.assertIsNone(pdf)
            self.assertFalse((root / "operator_distribution.pdf").exists())


if __name__ == "__main__":
    unittest.main()
