from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from export_operator_distribution_vldb import (
    OPERATOR_KEYS,
    normalize_operator,
    operator_ratio_matrix,
)


class OperatorDistributionTests(unittest.TestCase):
    def test_normalize_operator_accepts_only_four_families(self) -> None:
        self.assertEqual(normalize_operator("by_facet"), "facet")
        self.assertEqual(normalize_operator("BY_SUPERSET"), "superset")
        self.assertEqual(normalize_operator("neighbor"), "neighbor")
        self.assertEqual(normalize_operator("distribution"), "distribution")

        with self.assertRaisesRegex(ValueError, "Unknown operator"):
            normalize_operator("other")

    def test_operator_ratio_matrix_normalizes_each_available_seed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed1 = root / "seed1.csv"
            seed2 = root / "seed2.csv"
            pd.DataFrame(
                {"operator": ["by_facet", "by_facet", "by_superset"]}
            ).to_csv(seed1, index=False)
            pd.DataFrame(
                {"operator": ["by_neighbors", "by_distribution"]}
            ).to_csv(seed2, index=False)

            matrix = operator_ratio_matrix([seed1, None, seed2])

        self.assertEqual(
            OPERATOR_KEYS, ("facet", "superset", "neighbor", "distribution")
        )
        self.assertEqual(matrix.shape, (2, 4))
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(2))
        np.testing.assert_allclose(matrix[0], [2 / 3, 1 / 3, 0, 0])
        np.testing.assert_allclose(matrix[1], [0, 0, 1 / 2, 1 / 2])


if __name__ == "__main__":
    unittest.main()
