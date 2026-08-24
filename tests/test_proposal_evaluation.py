import unittest

from visual_peer_inspector.evaluation.proposals import (
    clustered_bootstrap_intervals,
    evaluate_proposal_dataset,
    match_boxes,
    summarize_proposal_evaluation,
)


def reference_row(scene: str, view: str, object_id: str, box):
    return {
        "opaque_scene_id": scene,
        "view_id": view,
        "opaque_object_id": object_id,
        "visible": box is not None,
        "box_xyxy": box,
    }


def proposal_row(scene: str, view: str, boxes):
    return {
        "opaque_scene_id": scene,
        "view_id": view,
        "status": "ok",
        "proposal_count": len(boxes),
        "boxes_xyxy": boxes,
    }


class ProposalMatchingTests(unittest.TestCase):
    def test_matching_prioritises_eligible_cardinality(self):
        references = [[0, 0, 2, 10], [0, 0, 3, 10]]
        proposals = [[0, 0, 3, 10], [0, 0, 5, 10]]

        matches = match_boxes(references, proposals, minimum_iou=0.5)

        self.assertEqual(
            [(item.reference_index, item.proposal_index) for item in matches],
            [(0, 0), (1, 1)],
        )

    def test_matching_is_deterministic_for_ties(self):
        references = [[0, 0, 10, 10], [0, 0, 10, 10]]
        proposals = [[0, 0, 10, 10], [0, 0, 10, 10]]

        first = match_boxes(references, proposals)
        second = match_boxes(references, proposals)

        self.assertEqual(first, second)


class ProposalDatasetTests(unittest.TestCase):
    def test_perfect_predictions_pass_all_image_metrics(self):
        references = [
            reference_row("S1", "V00", "O1", [0, 0, 10, 10]),
            reference_row("S1", "V00", "O2", [20, 20, 30, 30]),
        ]
        proposals = [
            proposal_row("S1", "V00", [[0, 0, 10, 10], [20, 20, 30, 30]])
        ]

        result = evaluate_proposal_dataset(
            reference_rows=references, proposal_rows=proposals
        )
        summary = summarize_proposal_evaluation(result["image_records"])

        self.assertEqual(summary["exact_count_rate"], 1.0)
        self.assertEqual(summary["micro_recall_at_0_50"], 1.0)
        self.assertEqual(summary["mean_matched_iou"], 1.0)
        self.assertEqual(result["failure_records"], [])

    def test_no_proposals_records_every_visible_object_as_missed(self):
        references = [
            reference_row("S1", "V00", "O1", [0, 0, 10, 10]),
            reference_row("S1", "V00", "O2", None),
        ]
        proposals = [proposal_row("S1", "V00", [])]

        result = evaluate_proposal_dataset(
            reference_rows=references, proposal_rows=proposals
        )
        summary = summarize_proposal_evaluation(result["image_records"])

        self.assertEqual(summary["reference_count"], 1)
        self.assertEqual(summary["miss_count"], 1)
        self.assertEqual(summary["micro_recall_at_0_50"], 0.0)
        self.assertEqual(len(result["failure_records"]), 1)

    def test_spurious_proposals_are_not_trimmed_to_reference_count(self):
        references = [reference_row("S1", "V00", "O1", [0, 0, 10, 10])]
        proposals = [
            proposal_row(
                "S1",
                "V00",
                [[0, 0, 10, 10], [40, 40, 50, 50], [60, 60, 70, 70]],
            )
        ]

        result = evaluate_proposal_dataset(
            reference_rows=references, proposal_rows=proposals
        )
        summary = summarize_proposal_evaluation(result["image_records"])

        self.assertEqual(summary["proposal_count"], 3)
        self.assertEqual(summary["extra_count"], 2)
        self.assertAlmostEqual(summary["micro_precision_at_0_50"], 1 / 3)

    def test_secondary_threshold_uses_a_fresh_assignment(self):
        references = [
            reference_row("S1", "V00", "O1", [0, 0, 2, 10]),
            reference_row("S1", "V00", "O2", [0, 0, 3, 10]),
        ]
        proposals = [
            proposal_row("S1", "V00", [[0, 0, 3, 10], [0, 0, 5, 10]])
        ]

        result = evaluate_proposal_dataset(
            reference_rows=references, proposal_rows=proposals
        )
        row = result["image_records"][0]

        self.assertEqual(row["matched_at_0_50"], 2)
        self.assertEqual(row["matched_at_0_75"], 1)

    def test_merge_and_split_candidates_are_reported(self):
        references = [
            reference_row("S1", "V00", "O1", [0, 0, 10, 10]),
            reference_row("S1", "V00", "O2", [10, 0, 20, 10]),
            reference_row("S1", "V01", "O1", [0, 0, 20, 10]),
        ]
        proposals = [
            proposal_row("S1", "V00", [[0, 0, 20, 10]]),
            proposal_row("S1", "V01", [[0, 0, 9, 10], [9, 0, 20, 10]]),
        ]

        result = evaluate_proposal_dataset(
            reference_rows=references, proposal_rows=proposals
        )

        by_view = {row["view_id"]: row for row in result["image_records"]}
        self.assertEqual(by_view["V00"]["merge_candidate_count"], 1)
        self.assertEqual(by_view["V01"]["split_candidate_count"], 1)

    def test_mismatched_image_coverage_fails_closed(self):
        references = [reference_row("S1", "V00", "O1", [0, 0, 10, 10])]
        proposals = [proposal_row("S1", "V01", [[0, 0, 10, 10]])]

        with self.assertRaisesRegex(ValueError, "coverage differs"):
            evaluate_proposal_dataset(
                reference_rows=references, proposal_rows=proposals
            )

    def test_asset_clustered_intervals_are_reproducible(self):
        rows = []
        for index, exact in enumerate((1, 0, 1, 1)):
            rows.append(
                {
                    "opaque_scene_id": f"S{index}",
                    "opaque_asset_group_id": f"A{index // 2}",
                    "reference_count": 1,
                    "proposal_count": 1,
                    "count_error": 0,
                    "miss_count": 1 - exact,
                    "extra_count": 1 - exact,
                    "matched_at_0_25": exact,
                    "matched_at_0_50": exact,
                    "matched_at_0_75": exact,
                    "exact_count": exact,
                    "full_image_success": exact,
                    "mean_matched_iou": float(exact),
                    "matched_ious": [float(exact)] if exact else [],
                    "merge_candidate_count": 0,
                    "split_candidate_count": 0,
                }
            )

        first = clustered_bootstrap_intervals(rows, repetitions=50, seed=7)
        second = clustered_bootstrap_intervals(rows, repetitions=50, seed=7)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
