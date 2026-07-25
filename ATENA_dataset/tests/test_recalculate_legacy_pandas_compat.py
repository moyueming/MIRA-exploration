import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from recalculate_baseline_corpus_metrics import runtime
from recalculate_baseline_corpus_metrics.compat import enable_legacy_pandas


class LegacyPandasCompatibilityTests(unittest.TestCase):
    def test_enable_legacy_pandas_restores_series_iteritems(self):
        original = getattr(pd.Series, "iteritems", None)
        if hasattr(pd.Series, "iteritems"):
            delattr(pd.Series, "iteritems")
        try:
            enable_legacy_pandas()
            self.assertIs(pd.Series.iteritems, pd.Series.items)
            self.assertFalse(pd.get_option("future.infer_string"))
        finally:
            if original is None:
                delattr(pd.Series, "iteritems")
            else:
                pd.Series.iteritems = original

    def test_default_mira_environment_enables_legacy_pandas(self):
        original = getattr(pd.Series, "iteritems", None)
        if hasattr(pd.Series, "iteritems"):
            delattr(pd.Series, "iteritems")
        fake_package = ModuleType("mira")
        fake_package.__path__ = []
        fake_env = ModuleType("mira.env")
        fake_env.make_env = lambda *args: args
        try:
            with patch.dict(
                sys.modules,
                {"mira": fake_package, "mira.env": fake_env},
            ):
                result = runtime._default_mira_env_factory(
                    "cyber", 1, 777, SimpleNamespace()
                )
            self.assertEqual(result[:3], ("cyber", 1, 777))
            self.assertIs(pd.Series.iteritems, pd.Series.items)
        finally:
            if original is None:
                if hasattr(pd.Series, "iteritems"):
                    delattr(pd.Series, "iteritems")
            else:
                pd.Series.iteritems = original


if __name__ == "__main__":
    unittest.main()
