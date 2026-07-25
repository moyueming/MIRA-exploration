import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "covertype-exploration"))

from covertype_rl.greedy_preprocessing import (  # noqa: E402
    EXPECTED_METADATA,
    OFFICIAL_GREEDY_INPUTS,
    REQUIRED_FILES,
    official_greedy_input,
    validate_official_greedy_preprocessing,
)


def make_universe(root, seed=1):
    mapping = official_greedy_input(seed)
    universe_dir = Path(root) / "preprocessed" / mapping.preprocess_name
    universe_dir.mkdir(parents=True)
    labels = [f"action_{index}" for index in range(202)]
    metadata = {**EXPECTED_METADATA, "seed": seed, "action_labels": labels}
    for filename in REQUIRED_FILES:
        path = universe_dir / filename
        if filename == "metadata.json":
            path.write_text(json.dumps(metadata), encoding="utf-8")
        else:
            path.write_bytes(b"existing fixed artifact")
    return mapping, universe_dir, labels, metadata


class CovertypeGreedyPreprocessingTests(unittest.TestCase):
    def test_official_seed_target_preprocessing_mapping(self):
        self.assertEqual(set(OFFICIAL_GREEDY_INPUTS), {1, 2, 3})
        for seed in (1, 2, 3):
            mapping = official_greedy_input(seed)
            self.assertEqual(mapping.target_set, f"fixed_seed_{seed}")
            self.assertEqual(
                mapping.preprocess_name, f"by_distribution_path100k_seed{seed}"
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            official_greedy_input(4)

    def test_validates_existing_official_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping, universe_dir, labels, metadata = make_universe(tmp, seed=2)
            validated = validate_official_greedy_preprocessing(
                tmp,
                seed=2,
                target_set=mapping.target_set,
                preprocess_name=mapping.preprocess_name,
                action_labels=labels,
            )
            self.assertEqual(validated.universe_dir, universe_dir.resolve())
            self.assertEqual(validated.metadata, metadata)

    def test_rejects_mismatched_official_mapping(self):
        cases = (
            ("fixed_seed_2", "by_distribution_path100k_seed1", "Target set mismatch"),
            ("fixed_seed_1", "by_distribution_path100k_seed2", "Preprocessing mismatch"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for target_set, preprocess_name, message in cases:
                with self.subTest(target_set=target_set, preprocess_name=preprocess_name):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_official_greedy_preprocessing(
                            tmp,
                            seed=1,
                            target_set=target_set,
                            preprocess_name=preprocess_name,
                            action_labels=[],
                        )

    def test_missing_preprocessing_fails_without_creating_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "does-not-exist"
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                validate_official_greedy_preprocessing(
                    root,
                    seed=1,
                    target_set="fixed_seed_1",
                    preprocess_name="by_distribution_path100k_seed1",
                    action_labels=[],
                )
            self.assertFalse(root.exists())

    def test_rejects_missing_file_metadata_and_action_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping, universe_dir, labels, metadata = make_universe(tmp)
            (universe_dir / "set_graph.npy").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "set_graph.npy"):
                validate_official_greedy_preprocessing(
                    tmp, seed=1, target_set=mapping.target_set,
                    preprocess_name=mapping.preprocess_name, action_labels=labels,
                )
            (universe_dir / "set_graph.npy").write_bytes(b"fixed")
            metadata["n_sets"] = 99_999
            (universe_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "n_sets"):
                validate_official_greedy_preprocessing(
                    tmp, seed=1, target_set=mapping.target_set,
                    preprocess_name=mapping.preprocess_name, action_labels=labels,
                )
            metadata["n_sets"] = 100_000
            (universe_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Action labels"):
                validate_official_greedy_preprocessing(
                    tmp, seed=1, target_set=mapping.target_set,
                    preprocess_name=mapping.preprocess_name,
                    action_labels=list(reversed(labels)),
                )


if __name__ == "__main__":
    unittest.main()