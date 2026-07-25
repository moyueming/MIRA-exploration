import unittest

import numpy as np

from baselines.greedy_eda.covertype import (
    WorkerExplorationMemory,
    choose_count_balanced_candidate,
)
from baselines.greedy_eda.policy import Candidate, choose_candidate, score_candidates


class _FailOnChoiceRng:
    def choice(self, *args, **kwargs):
        raise AssertionError("single-candidate selection must not consume RNG")


class CovertypeGreedySelectorTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            Candidate(30, np.array([1.0, 0.0]), 0.1),
            Candidate(20, np.array([0.7, 0.7]), 0.9),
            Candidate(10, np.array([0.0, 1.0]), 0.5),
        ]
        self.current = np.array([1.0, 0.0])
        self.history = [np.array([1.0, 0.0])]

    def test_extracted_scores_match_existing_candidate_selection(self):
        batch = score_candidates(self.candidates, self.current, self.history)
        selected = choose_candidate(
            self.candidates,
            self.current,
            self.history,
            set(),
            np.random.default_rng(9),
        )

        self.assertAlmostEqual(selected.score, batch.base_scores[selected.index])
        self.assertAlmostEqual(
            selected.components["interestingness"],
            batch.interestingness[selected.index],
        )
        self.assertAlmostEqual(
            selected.components["coherency"], batch.coherency[selected.index]
        )
        self.assertAlmostEqual(
            selected.components["diversity"], batch.diversity[selected.index]
        )

    def test_memory_persists_counts_and_is_isolated_between_workers(self):
        first = WorkerExplorationMemory()
        second = WorkerExplorationMemory()
        first.register_root(7)
        first.register_root(7)
        first.record_visit(9)
        first.record_visit(9)
        first.record_visit(10, valid_transition=False)

        self.assertEqual(first.set_visit_counts, {7: 1, 9: 2})
        self.assertEqual(second.set_visit_counts, {})

    def test_minimum_visit_count_has_priority_over_base_score(self):
        batch = score_candidates(self.candidates, self.current, self.history)
        highest = int(np.argmax(batch.base_scores))
        memory = WorkerExplorationMemory(
            {self.candidates[highest].candidate_id: 4}
        )

        selected = choose_count_balanced_candidate(
            self.candidates,
            self.current,
            self.history,
            memory,
            np.random.default_rng(2),
            top_k=1,
        )

        self.assertNotEqual(selected.index, highest)
        self.assertEqual(selected.components["worker_visit_count"], 0)

    def test_rank_sampling_is_reproducible(self):
        selections = []
        for _ in range(2):
            rng = np.random.default_rng(41)
            selections.append(
                [
                    choose_count_balanced_candidate(
                        self.candidates,
                        self.current,
                        self.history,
                        WorkerExplorationMemory(),
                        rng,
                    ).candidate_id
                    for _ in range(20)
                ]
            )

        self.assertEqual(selections[0], selections[1])
        self.assertGreater(len(set(selections[0])), 1)

    def test_single_eligible_candidate_does_not_consume_rng(self):
        memory = WorkerExplorationMemory({20: 1, 10: 1})
        selected = choose_count_balanced_candidate(
            self.candidates,
            self.current,
            self.history,
            memory,
            _FailOnChoiceRng(),
        )

        self.assertEqual(selected.candidate_id, 30)

    def test_rejects_empty_candidates_and_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "no legal candidates"):
            choose_count_balanced_candidate(
                [],
                self.current,
                self.history,
                WorkerExplorationMemory(),
                np.random.default_rng(0),
            )
        with self.assertRaisesRegex(ValueError, "top_k"):
            choose_count_balanced_candidate(
                self.candidates,
                self.current,
                self.history,
                WorkerExplorationMemory(),
                np.random.default_rng(0),
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
