import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira import avp_loader


class AvpLoaderTests(unittest.TestCase):
    def test_only_exact_string_one_enables_avp(self):
        self.assertTrue(avp_loader.avp_enabled("1"))
        for value in ("0", "2", "abc", "", None, 1, 0):
            with self.subTest(value=value):
                self.assertFalse(avp_loader.avp_enabled(value))

    def test_disabled_values_do_not_import_avp(self):
        with patch.object(avp_loader, "import_module") as importer:
            for value in ("0", "2", "abc", ""):
                with self.subTest(value=value):
                    self.assertEqual(avp_loader.load_avp("cyber", 1, value), {})

        importer.assert_not_called()

    def test_missing_avp_module_behaves_as_disabled(self):
        missing = ModuleNotFoundError(
            "No module named 'mira.avp'",
            name="mira.avp",
        )
        with patch.object(avp_loader, "import_module", side_effect=missing):
            self.assertEqual(avp_loader.load_avp("cyber", 1, "1"), {})

    def test_import_error_inside_installed_avp_is_not_hidden(self):
        nested = ModuleNotFoundError(
            "No module named 'broken_dependency'",
            name="broken_dependency",
        )
        with patch.object(avp_loader, "import_module", side_effect=nested):
            with self.assertRaises(ModuleNotFoundError):
                avp_loader.load_avp("cyber", 1, "1")

    def test_enabled_manifest_reports_active_avp(self):
        manifest = avp_loader.avp_manifest("cyber", 1, "1")

        self.assertEqual(manifest["requested"], "1")
        self.assertTrue(manifest["available"])
        self.assertTrue(manifest["active"])
        self.assertEqual(manifest["schema"], "cyber")
        self.assertEqual(manifest["dataset"], 1)
        self.assertTrue(manifest["terms"])
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_disabled_manifest_does_not_load_terms(self):
        with patch.object(avp_loader, "import_module") as importer:
            manifest = avp_loader.avp_manifest("cyber", 1, "abc")

        importer.assert_not_called()
        self.assertEqual(manifest["requested"], "abc")
        self.assertFalse(manifest["active"])
        self.assertEqual(manifest["terms"], {})

    def test_missing_module_manifest_reports_inactive(self):
        missing = ModuleNotFoundError(
            "No module named 'mira.avp'",
            name="mira.avp",
        )
        with patch.object(avp_loader, "import_module", side_effect=missing):
            manifest = avp_loader.avp_manifest("cyber", 1, "1")

        self.assertFalse(manifest["available"])
        self.assertFalse(manifest["active"])
        self.assertEqual(manifest["terms"], {})


if __name__ == "__main__":
    unittest.main()
