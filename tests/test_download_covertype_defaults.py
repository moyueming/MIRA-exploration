import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "covertype-exploration" / "scripts" / "download_covertype.py"
RELEASE_SHA256 = "a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4"


class DownloadCovertypeDefaultTests(unittest.TestCase):
    def test_default_checksum_matches_the_release_dataset(self):
        spec = importlib.util.spec_from_file_location("download_covertype", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module.DEFAULT_SHA256, RELEASE_SHA256)


if __name__ == "__main__":
    unittest.main()
