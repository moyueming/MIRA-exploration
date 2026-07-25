import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from recalculate_baseline_corpus_metrics import engine


class BaselineRecalculationEngineTests(unittest.TestCase):
    def test_deterministic_summary_keeps_corpus_bleu_and_means_saved_scores(self):
        saved_rows = []
        for index in range(8):
            saved_rows.append({
                "Precision": index / 10.0,
                "T-BLEU-1": 0.0,
                "T-BLEU-2": 0.0,
                "T-BLEU-3": 0.0,
                "EDA-Sim": index / 20.0,
            })
        corpus = {
            "Precision": 99.0,
            "T-BLEU-1": 0.41,
            "T-BLEU-2": 0.31,
            "T-BLEU-3": 0.21,
            "EDA-Sim": 99.0,
        }

        result = engine.deterministic_summary(saved_rows, corpus)

        self.assertAlmostEqual(result["Precision"], 0.35)
        self.assertAlmostEqual(result["EDA-Sim"], 0.175)
        self.assertEqual(result["T-BLEU-1"], 0.41)
        self.assertEqual(result["T-BLEU-2"], 0.31)
        self.assertEqual(result["T-BLEU-3"], 0.21)

    def test_random_summary_means_all_k_corpus_rows(self):
        rows = []
        for value in (0.0, 0.5, 1.0):
            rows.append({name: value for name in engine.METRICS})

        result = engine.random_summary(rows)

        self.assertEqual(result, {name: 0.5 for name in engine.METRICS})

    def test_ordered_sessions_requires_all_eight_datasets(self):
        with self.assertRaisesRegex(ValueError, "missing dataset keys"):
            engine.ordered_sessions({("cyber", 1): ["BACK"] * 12})

        sessions = {
            key: [f"{key[0]}-{key[1]}"] * 12
            for key in engine.EXPECTED_DATASETS
        }
        ordered = engine.ordered_sessions(sessions)

        self.assertEqual(len(ordered), 8)
        self.assertEqual(ordered[0][0], ("cyber", 1))
        self.assertEqual(ordered[-1][0], ("flights", 4))

    def test_random_episode_saved_scores_replace_runtime_precision_and_eda(self):
        runtime = {name: 0.25 for name in engine.METRICS}
        saved = [
            dict(runtime, Precision=0.0, **{"EDA-Sim": 0.2}),
            dict(runtime, Precision=1.0, **{"EDA-Sim": 0.6}),
        ]

        result = engine.apply_saved_scalar_metrics(runtime, saved)

        self.assertEqual(result["Precision"], 0.5)
        self.assertEqual(result["EDA-Sim"], 0.4)
        self.assertEqual(result["T-BLEU-1"], 0.25)


if __name__ == "__main__":
    unittest.main()
