from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from export_operator_distribution_vldb import (
    OPERATOR_KEYS,
    normalize_operator,
    operator_ratio_matrix,
)


def test_normalize_operator_accepts_only_four_families() -> None:
    assert normalize_operator("by_facet") == "facet"
    assert normalize_operator("BY_SUPERSET") == "superset"
    assert normalize_operator("neighbor") == "neighbor"
    assert normalize_operator("distribution") == "distribution"

    with pytest.raises(ValueError, match="Unknown operator"):
        normalize_operator("other")


def test_operator_ratio_matrix_normalizes_each_available_seed(tmp_path: Path) -> None:
    seed1 = tmp_path / "seed1.csv"
    seed2 = tmp_path / "seed2.csv"
    pd.DataFrame(
        {"operator": ["by_facet", "by_facet", "by_superset"]}
    ).to_csv(seed1, index=False)
    pd.DataFrame(
        {"operator": ["by_neighbors", "by_distribution"]}
    ).to_csv(seed2, index=False)

    matrix = operator_ratio_matrix([seed1, None, seed2])

    assert OPERATOR_KEYS == ("facet", "superset", "neighbor", "distribution")
    assert matrix.shape == (2, 4)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(matrix[0], [2 / 3, 1 / 3, 0, 0])
    np.testing.assert_allclose(matrix[1], [0, 0, 1 / 2, 1 / 2])
