import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from render_greedy_eda_final_figures import (
    EXCLUDED_ABLATION_OUTPUTS,
    TARGET_OUTPUTS,
    render_greedy_eda_final_figures,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GreedyFinalRenderTests(unittest.TestCase):
    def test_renderer_writes_exact_targets_and_preserves_ablation_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            before = {}
            for dataset, folder_name in (
                ("Galaxy", "galaxy_final"),
                ("Covertype", "covertype_final"),
            ):
                folder = root / folder_name
                folder.mkdir(parents=True)
                for name in EXCLUDED_ABLATION_OUTPUTS[dataset]:
                    path = folder / name
                    path.write_bytes(f"protected:{dataset}:{name}".encode("ascii"))
                    before[path] = digest(path)

            outputs = render_greedy_eda_final_figures(root)

            for dataset, folder_name in (
                ("Galaxy", "galaxy_final"),
                ("Covertype", "covertype_final"),
            ):
                self.assertEqual(set(outputs[dataset]), set(TARGET_OUTPUTS[dataset]))
                self.assertEqual(len(outputs[dataset]), 10)
                self.assertTrue(
                    all(
                        path.read_bytes().startswith(b"\x89PNG")
                        for path in outputs[dataset].values()
                    )
                )
                self.assertEqual(list((root / folder_name).glob("*.pdf")), [])

            self.assertEqual(
                {path: digest(path) for path in before},
                before,
            )


if __name__ == "__main__":
    unittest.main()
