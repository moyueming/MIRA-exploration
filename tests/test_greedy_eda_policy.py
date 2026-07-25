import unittest

import numpy as np

from baselines.greedy_eda.policy import Candidate, choose_candidate, choose_operation


class GreedyEdaPolicyTests(unittest.TestCase):
    def test_choose_candidate_prefers_target_blind_score_and_penalizes_repeat(self):
        candidates = [
            Candidate(10, np.array([1.0, 0.0]), 0.1),
            Candidate(11, np.array([0.7, 0.7]), 0.9),
        ]

        selected = choose_candidate(
            candidates,
            current_state=np.array([1.0, 0.0]),
            history_states=[np.array([1.0, 0.0])],
            visited_ids={10},
            rng=np.random.default_rng(7),
        )

        self.assertEqual(selected.candidate_id, 11)
        self.assertEqual(
            set(selected.components),
            {"interestingness", "coherency", "diversity", "repeat_penalty"},
        )

    def test_choose_candidate_sanitizes_non_finite_values_and_seeded_ties(self):
        candidates = [
            Candidate(1, np.array([np.nan, 0.0]), np.inf),
            Candidate(2, np.array([0.0, np.nan]), np.inf),
        ]

        first = choose_candidate(
            candidates,
            np.zeros(2),
            [],
            set(),
            np.random.default_rng(3),
        )
        second = choose_candidate(
            candidates,
            np.zeros(2),
            [],
            set(),
            np.random.default_rng(3),
        )

        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertTrue(np.isfinite(first.score))

    def test_choose_candidate_rejects_empty_candidates(self):
        with self.assertRaisesRegex(ValueError, "no legal candidates"):
            choose_candidate([], np.zeros(1), [], set(), np.random.default_rng(0))

    def test_choose_operation_balances_family_then_action_and_rejects_empty_mask(self):
        rng = np.random.default_rng(5)

        selected = choose_operation(
            [2, 3, 4],
            {2: "facet", 3: "facet", 4: "neighbors"},
            {"facet": 2, "neighbors": 0},
            {2: 0, 3: 1, 4: 4},
            rng,
        )

        self.assertEqual(selected, 4)
        with self.assertRaisesRegex(ValueError, "no legal operations"):
            choose_operation([], {}, {}, {}, rng)


if __name__ == "__main__":
    unittest.main()
