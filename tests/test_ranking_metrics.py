import unittest

from visual_peer_inspector.evaluation.ranking import (
    average_views_within_scene,
    clustered_mean_interval,
    paired_clustered_difference,
    ranking_metrics,
)


class RankingMetricTests(unittest.TestCase):
    def test_missed_anomaly_remains_in_average_precision_denominator(self):
        metrics = ranking_metrics(
            [True, False, False],
            positive_total=2,
            ranked_object_ids=["O1", None, "O3"],
            positive_object_ids=["O1", "O2"],
        )

        self.assertEqual(metrics.top1, 1.0)
        self.assertEqual(metrics.average_precision, 0.5)
        self.assertEqual(metrics.recall_at_2, 0.5)
        self.assertEqual(metrics.exact_set, 0.0)

    def test_unmatched_top_candidate_is_not_a_positive(self):
        metrics = ranking_metrics(
            [False, True, False],
            positive_total=1,
            ranked_object_ids=[None, "O1", "O2"],
            positive_object_ids=["O1"],
        )

        self.assertEqual(metrics.top1, 0.0)
        self.assertEqual(metrics.reciprocal_rank, 0.5)
        self.assertEqual(metrics.average_precision, 0.5)

    def test_recall_at_two_uses_the_retrievable_positive_count(self):
        metrics = ranking_metrics(
            [True, True, True],
            positive_total=3,
            ranked_object_ids=["O1", "O2", "O3"],
            positive_object_ids=["O1", "O2", "O3"],
        )

        self.assertEqual(metrics.recall_at_2, 1.0)
        self.assertEqual(metrics.exact_set, 1.0)

    def test_views_are_averaged_before_dataset_summary(self):
        base = {
            "opaque_scene_id": "S1",
            "opaque_asset_group_id": "A1",
            "encoder_key": "dino",
            "route": "supplied_mask",
            "method": "B0",
            "anomaly_count": 1,
            "reciprocal_rank": 1.0,
            "recall_at_2": 1.0,
            "exact_set": 1.0,
        }
        rows = [
            {**base, "top1": 1.0, "average_precision": 1.0},
            {**base, "top1": 0.0, "average_precision": 0.5},
        ]

        result = average_views_within_scene(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["eligible_view_count"], 2)
        self.assertEqual(result[0]["average_precision"], 0.75)

    def test_clustered_interval_is_deterministic(self):
        rows = [
            {"opaque_asset_group_id": "A1", "average_precision": 0.5},
            {"opaque_asset_group_id": "A2", "average_precision": 1.0},
        ]

        first = clustered_mean_interval(rows, metric="average_precision", repetitions=100, seed=7)
        second = clustered_mean_interval(rows, metric="average_precision", repetitions=100, seed=7)

        self.assertEqual(first, second)

    def test_paired_difference_uses_candidate_minus_baseline(self):
        baseline = [
            {"opaque_scene_id": "S1", "opaque_asset_group_id": "A1", "average_precision": 0.5},
            {"opaque_scene_id": "S2", "opaque_asset_group_id": "A2", "average_precision": 0.6},
        ]
        candidate = [
            {"opaque_scene_id": "S1", "opaque_asset_group_id": "A1", "average_precision": 0.7},
            {"opaque_scene_id": "S2", "opaque_asset_group_id": "A2", "average_precision": 0.5},
        ]

        result = paired_clustered_difference(
            baseline,
            candidate,
            metric="average_precision",
            repetitions=100,
            seed=7,
        )

        self.assertAlmostEqual(result["mean_difference"], 0.05)
        self.assertEqual(result["paired_scene_count"], 2)
        self.assertEqual(result["cluster_count"], 2)


if __name__ == "__main__":
    unittest.main()
