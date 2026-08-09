from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_peer_inspector import average_precision, peer_mean_scores, permutation_error


class BaselineTests(unittest.TestCase):
    def test_outlier_receives_highest_score(self) -> None:
        vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
        scores = peer_mean_scores(vectors)
        self.assertEqual(max(range(len(scores)), key=scores.__getitem__), 2)

    def test_score_is_not_changed_by_object_order(self) -> None:
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.2, 0.8], [0.0, 1.0]]
        scores = peer_mean_scores(vectors)
        self.assertLess(permutation_error(vectors, scores), 1e-12)

    def test_average_precision_uses_full_ranking(self) -> None:
        labels = [True, False, True, False]
        self.assertAlmostEqual(average_precision(labels, [0, 1, 2, 3]), 5 / 6)

    def test_rejects_a_single_object(self) -> None:
        with self.assertRaises(ValueError):
            peer_mean_scores([[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
