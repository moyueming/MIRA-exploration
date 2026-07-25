import unittest
from pathlib import Path


class GreedyEdaCommandDocumentationTests(unittest.TestCase):
    def test_readmes_document_both_entry_points_and_information_boundary(self):
        root = Path("README.md").read_text(encoding="utf-8")
        covertype = Path("covertype-exploration/README.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("RL-launcher-greedy-eda.py", root)
        self.assertIn("baselines/greedy_eda/run.py", covertype)
        for document in (root, covertype):
            lowered = document.lower()
            self.assertIn("target-blind", lowered)
            self.assertIn("non-learning", lowered)
            self.assertIn("evaluation only", lowered)


if __name__ == "__main__":
    unittest.main()
