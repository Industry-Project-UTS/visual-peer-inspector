import unittest

import numpy as np

from visual_peer_inspector.scoring.peer import (
    b0_scores,
    b1_scores,
    deterministic_ranking,
    p1_scores,
)


class GlobalPeerScoringTests(unittest.TestCase):
    def test_b0_and_b1_match_hand_calculated_fixture(self):
        features = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)

        b0 = b0_scores(features)
        b1 = b1_scores(features)

        self.assertEqual(b0.status, "scored")
        self.assertTrue(np.allclose(b0.scores, [0.61731657, 0.61731657, 0.0]))
        self.assertTrue(np.allclose(b1.scores, [0.64644661, 0.64644661, 0.29289322]))

    def test_every_scorer_requires_three_candidates(self):
        features = np.asarray([[1, 0], [0, 1]], dtype=float)

        self.assertEqual(b0_scores(features).status, "insufficient_peers")
        self.assertEqual(b1_scores(features).status, "insufficient_peers")

    def test_scores_are_invariant_to_candidate_order(self):
        features = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
        order = [2, 0, 3, 1]
        expected = b0_scores(features).scores
        shuffled = b0_scores(features[order]).scores
        restored = np.empty_like(shuffled)
        for new_index, old_index in enumerate(order):
            restored[old_index] = shuffled[new_index]

        self.assertTrue(np.allclose(expected, restored))

    def test_ranking_uses_stable_identifiers_for_ties(self):
        ranking = deterministic_ranking([0.5, 0.7, 0.7], ["C3", "C2", "C1"])

        self.assertEqual(ranking, [2, 1, 0])


class PatchPeerScoringTests(unittest.TestCase):
    def test_identical_patch_sets_have_zero_mismatch(self):
        patches = np.eye(4, dtype=float)
        masks = np.ones((2, 2), dtype=bool)

        result = p1_scores([patches, patches, patches], [masks, masks, masks])

        self.assertEqual(result.status, "scored")
        self.assertTrue(np.allclose(result.scores, 0.0))
        self.assertTrue(np.allclose(result.heatmaps[0], 0.0))

    def test_local_change_raises_the_expected_heatmap_cell(self):
        base = np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)
        changed = base.copy()
        changed[3] = [0, 1]
        masks = np.ones((2, 2), dtype=bool)

        result = p1_scores([changed, base, base], [masks, masks, masks], top_fraction=0.25)

        self.assertGreater(result.scores[0], result.scores[1])
        self.assertEqual(np.unravel_index(np.nanargmax(result.heatmaps[0]), (2, 2)), (1, 1))

    def test_padding_is_excluded_from_patch_comparison(self):
        patches = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        valid = np.asarray([[True, True], [False, False]])

        result = p1_scores([patches, patches, patches], [valid, valid, valid])

        self.assertTrue(np.isnan(result.heatmaps[0][1]).all())
        self.assertTrue(np.allclose(result.scores, 0.0))


if __name__ == "__main__":
    unittest.main()
