import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mira.train import run_experiment
from run import parse_args


@unittest.skipUnless(
    importlib.util.find_spec("tensorflow") is not None,
    "TensorFlow is not installed in this runtime",
)
class StandaloneTensorFlowSmokeTests(unittest.TestCase):
    def test_two_worker_twenty_four_step_run_writes_complete_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args([
                "--schema", "cyber",
                "--dataset_number", "1",
                "--workers", "2",
                "--steps", "24",
                "--eval_interval", "1",
                "--output_dir", tmp,
            ])
            rows = run_experiment(args)
            result_dir = (
                Path(tmp)
                / "MIRA"
                / "cyber1"
                / "seed0"
            )

            self.assertEqual(rows[-1]["steps"], 24)
            for filename in (
                "config.json",
                "avp_manifest.json",
                "eval_metrics.csv",
                "eval_metrics_online.csv",
                "final_metrics.json",
                "final_metrics_online.json",
                "policy.weights.h5",
                "policy.online.weights.h5",
                "actions_steps24.json",
                "actions_online_steps24.json",
            ):
                self.assertTrue((result_dir / filename).exists(), filename)
            manifest = json.loads(
                (result_dir / "avp_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["requested"], "0")
            self.assertTrue(manifest["available"])
            self.assertFalse(manifest["active"])
            self.assertEqual(manifest["terms"], {})


if __name__ == "__main__":
    unittest.main()
