import unittest

from export_operator_distribution_vldb import format_method_tick_label


class OperatorDistributionTickLabelTests(unittest.TestCase):
    def test_long_ablation_labels_wrap_without_renaming(self) -> None:
        self.assertEqual(
            format_method_tick_label("MIRA w/o Ext. Reward"),
            "MIRA w/o\nExt. Reward",
        )
        self.assertEqual(
            format_method_tick_label("ATENA w/o Ext. Reward"),
            "ATENA w/o\nExt. Reward",
        )
        self.assertEqual(format_method_tick_label("DORA"), "DORA")


if __name__ == "__main__":
    unittest.main()
