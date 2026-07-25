import unittest

from export_operator_distribution_vldb import covertype_methods, galaxy_methods


class OperatorDistributionPathTests(unittest.TestCase):
    def test_all_configured_trace_paths_exist(self) -> None:
        for dataset_methods in (galaxy_methods(), covertype_methods()):
            for method in dataset_methods:
                for path in method["trace_files"]:
                    if path is not None:
                        self.assertTrue(path.exists(), str(path))


if __name__ == "__main__":
    unittest.main()
