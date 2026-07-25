import unittest

import numpy as np

from export_operator_distribution_vldb import (
    OPERATOR_LABELS,
    covertype_methods,
    galaxy_methods,
    operator_ymax,
)
from final_figure_registry import ALL_METHODS


EXPECTED_METHOD_LABELS = list(ALL_METHODS)


class OperatorDistributionLabelTests(unittest.TestCase):
    def test_both_datasets_use_approved_method_labels(self) -> None:
        for methods in (galaxy_methods(), covertype_methods()):
            self.assertEqual(
                [method["display_label"] for method in methods],
                EXPECTED_METHOD_LABELS,
            )

    def test_atena_labels_match_trace_configuration(self) -> None:
        galaxy = {method["display_label"]: method for method in galaxy_methods()}
        covertype = {method["display_label"]: method for method in covertype_methods()}
        self.assertTrue(all("ATENA_ext" in path.parts for path in galaxy["ATENA"]["trace_files"] if path is not None))
        self.assertTrue(all("ATENA_pure" in path.parts for path in galaxy["ATENA w/o Ext. Reward"]["trace_files"] if path is not None))
        self.assertTrue(all("ATENA-EXT" in path.parts for path in covertype["ATENA"]["trace_files"]))
        self.assertTrue(all("ATENA" in path.parts for path in covertype["ATENA w/o Ext. Reward"]["trace_files"]))

    def test_operator_legend_uses_by_names(self) -> None:
        self.assertEqual(
            OPERATOR_LABELS,
            ("by_facet", "by_superset", "by_neighbors", "by_distribution"),
        )

    def test_ymax_leaves_room_above_complete_sd_errorbar(self) -> None:
        means = np.array([[0.88685977]])
        deviations = np.array([[0.12568972]])
        upper_errorbar = float(np.max(means + deviations))

        ymax = operator_ymax(means, deviations)

        self.assertGreaterEqual(ymax, upper_errorbar + 0.04)


if __name__ == "__main__":
    unittest.main()
