import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    ROOT / "export_covertype_final_six_png.py",
    ROOT / "export_covertype_final_galaxy_style_png.py",
    ROOT / "export_operator_distribution_vldb" / "__init__.py",
    ROOT / "covertype-exploration" / "plot_covertype_final_ma25_figures.py",
]
STALE_BASELINE = "ours" + "_bile"


class PublicMiraNamingTests(unittest.TestCase):
    def test_public_plotting_sources_do_not_expose_the_old_baseline_name(self):
        stale = [
            str(path.relative_to(ROOT))
            for path in PUBLIC_FILES
            if STALE_BASELINE in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
