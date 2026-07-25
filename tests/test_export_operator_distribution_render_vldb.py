from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from export_operator_distribution_vldb import plot_operator_distribution


class OperatorDistributionRenderTests(unittest.TestCase):
    def test_plot_writes_nonempty_png_and_pdf(self) -> None:
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
            methods = [{"label": "Method A", "trace_files": [trace]}]

            png, pdf = plot_operator_distribution(
                methods,
                root / "operator_distribution",
            )

            self.assertTrue(png.exists())
            self.assertTrue(pdf.exists())
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
