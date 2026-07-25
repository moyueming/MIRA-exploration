import unittest

import numpy as np

from baselines.greedy_eda.galaxy import (
    build_parser,
    execute_selected_operation,
    legal_operation_ids,
    preview_candidates,
)


class Dataset:
    def __init__(self, set_id, state, interestingness, nonempty=True):
        self.set_id = set_id
        self.state = state
        self.interestingness = interestingness
        self.data = [1] if nonempty else []


class EncoderSpy:
    def __init__(self):
        self.reward_flags = []

    def encode_dataset(self, dataset, parent_dataset=None, get_reward=True):
        self.reward_flags.append(get_reward)
        return np.asarray(dataset.state), 999.0, dataset.interestingness


class GalaxyGreedyEdaTests(unittest.TestCase):
    def test_preview_candidates_disables_reward_and_preserves_dataset_indices(self):
        encoder = EncoderSpy()
        datasets = [
            Dataset(10, [1.0, 0.0], 0.2, nonempty=False),
            Dataset(11, [0.0, 1.0], 0.8),
        ]

        indices, candidates = preview_candidates(datasets, object(), encoder)

        self.assertEqual(indices, [1])
        self.assertEqual([item.candidate_id for item in candidates], [11])
        self.assertEqual(encoder.reward_flags, [False])

    def test_legal_operation_ids_uses_existing_validity_rule(self):
        actions = ["by_superset", "by_distribution", "by_neighbors-&-x"]

        legal = legal_operation_ids(
            object(),
            actions,
            validity_fn=lambda _dataset, action: action != "by_superset",
        )

        self.assertEqual(legal, [1, 2])

    def test_execute_selected_operation_calls_pipeline_once(self):
        calls = []

        def execute_fn(pipeline, dataset, action_type):
            calls.append((pipeline, dataset, action_type))
            return [Dataset(1, [1.0], 0.0)]

        result = execute_selected_operation("pipeline", "dataset", "by_facet", execute_fn)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result), 1)

    def test_parser_uses_reproducible_experiment_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.baseline, "greedy_eda")
        self.assertEqual(args.episodes, 1000)
        self.assertEqual(args.steps, 250)
        self.assertEqual(args.workers, 12)


if __name__ == "__main__":
    unittest.main()
