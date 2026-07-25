import sys
import tempfile
import unittest
from unittest import mock
from types import ModuleType, SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import recalculate_baseline_corpus_metrics as recalculator
from recalculate_baseline_corpus_metrics import application


class RecalculationHelperTests(unittest.TestCase):
    def test_mean_metric_rows_averages_all_five_metrics(self):
        rows = [
            {name: 0.0 for name in recalculator.METRICS},
            {name: 1.0 for name in recalculator.METRICS},
        ]

        self.assertEqual(
            recalculator.mean_metric_rows(rows),
            {name: 0.5 for name in recalculator.METRICS},
        )

    def test_mean_metric_rows_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "zero metric rows"):
            recalculator.mean_metric_rows([])

    def test_validate_dataset_keys_requires_exactly_all_eight(self):
        with self.assertRaisesRegex(ValueError, "missing dataset keys"):
            recalculator.validate_dataset_keys({("cyber", 1)})

        recalculator.validate_dataset_keys(recalculator.EXPECTED_DATASETS)

    def test_validate_metric_row_reports_context_and_metric(self):
        expected = {name: 0.0 for name in recalculator.METRICS}
        observed = dict(expected, Precision=0.5)

        with self.assertRaisesRegex(ValueError, "pure_a3c/cyber1: Precision"):
            recalculator.validate_metric_row(
                expected,
                observed,
                "pure_a3c/cyber1",
            )

    def test_validate_metric_row_allows_legacy_eda_numeric_drift(self):
        expected = {name: 0.0 for name in recalculator.METRICS}
        observed = dict(expected, **{"EDA-Sim": 0.019})

        recalculator.validate_metric_row(expected, observed, "official_atena/flights3")

    def test_all_method_order_and_legacy_mira_result_layout(self):
        self.assertEqual(
            application.METHOD_ORDER,
            ("official_atena", "random", "pure_a3c", "dora", "greedy", "MIRA"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            legacy = results / "MIRA" / "mira" / "cyber2" / "seed0"
            legacy.mkdir(parents=True)
            self.assertEqual(
                application._mira_result_dir(results, ("cyber", 2), 0),
                legacy,
            )

    def test_corpus_metric_computes_only_official_tree_bleu_metrics(self):
        sessions = {key: [str(key)] * 12 for key in recalculator.EXPECTED_DATASETS}
        instances = [object()]

        with mock.patch.object(
            application,
            "_official_instances",
            return_value=instances,
        ) as build_instances, mock.patch.object(
            application,
            "_tree_bleu",
            side_effect=lambda n, observed: n / 10.0 if observed is instances else -1.0,
        ) as tree_bleu:
            observed = application._corpus_metric(sessions)

        self.assertEqual(
            observed,
            {"T-BLEU-1": 0.1, "T-BLEU-2": 0.2, "T-BLEU-3": 0.3},
        )
        build_instances.assert_called_once_with(sessions)
        self.assertEqual(
            tree_bleu.call_args_list,
            [mock.call(1, instances), mock.call(2, instances), mock.call(3, instances)],
        )

    def test_official_instances_returns_one_instance_per_session(self):
        metrics = ModuleType("atena.evaluation.metrics")
        metrics.EvalInstance = lambda dataset_meta, actions: SimpleNamespace(
            dataset_meta=dataset_meta,
            actions=actions,
        )
        dataset = ModuleType("atena.simulation.dataset")
        dataset.DatasetMeta = lambda schema, name: (schema, name)
        env = ModuleType("atena_baselines.env")
        env.dataset_enum = lambda schema, number: (schema, f"dataset{number}")

        atena = ModuleType("atena")
        atena.__path__ = []
        evaluation = ModuleType("atena.evaluation")
        evaluation.__path__ = []
        simulation = ModuleType("atena.simulation")
        simulation.__path__ = []
        baselines = ModuleType("atena_baselines")
        baselines.__path__ = []
        atena.evaluation = evaluation
        atena.simulation = simulation
        evaluation.metrics = metrics
        simulation.dataset = dataset
        baselines.env = env

        modules = {
            "atena": atena,
            "atena.evaluation": evaluation,
            "atena.evaluation.metrics": metrics,
            "atena.simulation": simulation,
            "atena.simulation.dataset": dataset,
            "atena_baselines": baselines,
            "atena_baselines.env": env,
        }
        sessions = {
            ("flights", 2): ["flights"],
            ("cyber", 1): ["cyber"],
        }
        with mock.patch.dict(sys.modules, modules):
            instances = application._official_instances(sessions)

        self.assertEqual(
            [instance.actions for instance in instances],
            [["cyber"], ["flights"]],
        )


if __name__ == "__main__":
    unittest.main()
