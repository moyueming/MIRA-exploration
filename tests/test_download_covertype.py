import hashlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "covertype-exploration" / "scripts" / "download_covertype.py"
HEADER = (
    "Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,"
    "Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,"
    "Hillshade_9am,Hillshade_Noon,Hillshade_3pm,"
    "Horizontal_Distance_To_Fire_Points,"
    + ",".join(f"Wilderness_Area{i}" for i in range(1, 5))
    + ","
    + ",".join(f"Soil_Type{i}" for i in range(1, 41))
    + ",Cover_Type"
)


class DownloadCovertypeTests(unittest.TestCase):
    def test_cli_downloads_archive_and_adds_the_canonical_header(self):
        raw_rows = b"1,2,3\n4,5,6\n"
        expected = (HEADER + "\n").encode("ascii") + raw_rows
        expected_sha256 = hashlib.sha256(expected).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            archive = temp / "covtype.zip"
            output = temp / "covertype.csv"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("covtype.data", raw_rows)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--url",
                    archive.as_uri(),
                    "--output",
                    str(output),
                    "--expected-sha256",
                    expected_sha256,
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), expected)
            self.assertIn(expected_sha256, result.stdout)


if __name__ == "__main__":
    unittest.main()
