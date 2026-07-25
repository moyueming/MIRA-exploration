import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ServerWorkflowTests(unittest.TestCase):
    def test_batch_script_only_invokes_standalone_runner(self):
        source = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")

        self.assertIn("python MIRA/run.py", source)
        self.assertNotIn("run_atena_baselines.py", source)
        for variable in ("SCHEMAS", "DATASETS", "WORKERS", "SEED", "STEPS", "AVP"):
            self.assertIn(variable, source)
        self.assertIn('AVP="${AVP:-0}"', source)
        self.assertIn('--avp "${AVP}"', source)

    def test_readme_discloses_avp_and_server_contract(self):
        source = (ROOT / "README.md").read_text(encoding="utf-8")

        for phrase in (
            "AVP",
            "--avp 1",
            "AVP=1",
            "mira/avp.py",
            "avp_manifest.json",
            "results/MIRA",
            "python MIRA/run.py",
            "MIRA/scripts/run_all.sh",
        ):
            self.assertIn(phrase, source)

    def test_requirements_are_frozen(self):
        requirements = set(
            (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        )

        self.assertIn("tensorflow==2.9.0", requirements)
        self.assertIn("numpy==1.23.5", requirements)
        self.assertIn("pandas==1.4.4", requirements)


if __name__ == "__main__":
    unittest.main()
