import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from recalculate_baseline_corpus_metrics import EXPECTED_DATASETS, METRICS
from recalculate_baseline_corpus_metrics import orchestrator


def metric_row(value):
    return {name: float(value) for name in METRICS}


class BaselineOrchestratorTests(unittest.TestCase):
    def test_calculate_deterministic_validates_eight_and_uses_corpus_bleu(self):
        sessions = {key: [str(key)] * 12 for key in EXPECTED_DATASETS}
        saved = {key: metric_row(index / 10.0) for index, key in enumerate(sorted(EXPECTED_DATASETS))}
        validated = []

        def single_metric(key, actions):
            validated.append((key, len(actions)))
            return saved[key]

        corpus = metric_row(0.9)
        detail, summary = orchestrator.calculate_deterministic(
            "greedy",
            sessions,
            saved,
            single_metric=single_metric,
            corpus_metric=lambda _sessions: corpus,
        )

        self.assertEqual(len(validated), 8)
        self.assertEqual(len(detail), 8)
        self.assertEqual(summary["method"], "greedy")
        self.assertEqual(summary["T-BLEU-3"], 0.9)
        self.assertAlmostEqual(summary["Precision"], 0.35)

    def test_recomputed_deterministic_uses_official_runtime_metrics_only(self):
        sessions = {key: [str(key)] * 12 for key in EXPECTED_DATASETS}
        observed = {
            key: metric_row((index + 1) / 10.0)
            for index, key in enumerate(sorted(EXPECTED_DATASETS))
        }
        corpus = dict(metric_row(0.9), Precision=99.0, **{"EDA-Sim": 99.0})

        detail, summary = orchestrator.calculate_recomputed_deterministic(
            "pure_a3c",
            sessions,
            single_metric=lambda key, _actions: observed[key],
            corpus_metric=lambda _sessions: corpus,
        )

        self.assertEqual(len(detail), 8)
        self.assertEqual(detail[0]["Precision"], observed[("cyber", 1)]["Precision"])
        self.assertEqual(
            summary,
            {
                "method": "pure_a3c",
                "Precision": 0.45,
                "T-BLEU-1": 0.9,
                "T-BLEU-2": 0.9,
                "T-BLEU-3": 0.9,
                "EDA-Sim": 0.45,
            },
        )

    def test_calculate_random_validates_128_sessions_and_averages_k_rows(self):
        sessions = {
            key: [[f"{key}-{episode}"] * 12 for episode in range(16)]
            for key in EXPECTED_DATASETS
        }
        saved = {
            key: [metric_row(episode / 15.0) for episode in range(16)]
            for key in EXPECTED_DATASETS
        }
        calls = {"single": 0, "corpus": 0}

        def single_metric(key, actions):
            calls["single"] += 1
            episode = int(actions[0].rsplit("-", 1)[1])
            return saved[key][episode]

        def corpus_metric(episode_sessions):
            calls["corpus"] += 1
            episode = int(next(iter(episode_sessions.values()))[0].rsplit("-", 1)[1])
            return metric_row(episode / 15.0)

        detail, summary = orchestrator.calculate_random(
            sessions,
            saved,
            single_metric=single_metric,
            corpus_metric=corpus_metric,
        )

        self.assertEqual(calls, {"single": 128, "corpus": 16})
        self.assertEqual(len(detail), 8)
        self.assertEqual(summary["method"], "random")
        for name in METRICS:
            self.assertAlmostEqual(summary[name], 0.5)

    def test_recomputed_random_uses_sixteen_official_corpus_rows(self):
        sessions = {
            key: [[f"{key}-{episode}"] * 12 for episode in range(16)]
            for key in EXPECTED_DATASETS
        }
        calls = {"single": 0, "corpus": 0}

        def single_metric(_key, actions):
            calls["single"] += 1
            episode = int(actions[0].rsplit("-", 1)[1])
            return metric_row(episode / 15.0)

        def corpus_metric(episode_sessions):
            calls["corpus"] += 1
            episode = int(next(iter(episode_sessions.values()))[0].rsplit("-", 1)[1])
            return dict(
                metric_row(episode / 15.0), Precision=99.0, **{"EDA-Sim": 99.0}
            )

        detail, summary = orchestrator.calculate_recomputed_random(
            sessions,
            single_metric=single_metric,
            corpus_metric=corpus_metric,
        )

        self.assertEqual(calls, {"single": 128, "corpus": 16})
        self.assertEqual(len(detail), 8)
        self.assertEqual(summary["method"], "random")
        for name in METRICS:
            self.assertAlmostEqual(summary[name], 0.5)


if __name__ == "__main__":
    unittest.main()
