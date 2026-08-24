import json
import unittest

import numpy as np

from visual_peer_inspector.evaluation.masks import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    binary_mask_dice,
    binary_mask_iou,
    boundary_precision_recall_f1,
    evaluate_mask_route_image,
    mask_spill_outside_prompt_box,
    paired_asset_clustered_bootstrap,
    separate_mask_evidence,
    summarize_mask_routes,
)


def square(y1=2, y2=8, x1=2, x2=8, size=16):
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class BinaryMaskMetricTests(unittest.TestCase):
    def test_iou_and_dice_cover_perfect_disjoint_and_empty_masks(self):
        first = square()
        second = square(8, 14, 8, 14)
        empty = np.zeros_like(first)

        self.assertEqual(binary_mask_iou(first, first), 1.0)
        self.assertEqual(binary_mask_dice(first, first), 1.0)
        self.assertEqual(binary_mask_iou(first, second), 0.0)
        self.assertEqual(binary_mask_dice(first, second), 0.0)
        self.assertEqual(binary_mask_iou(empty, empty), 1.0)
        self.assertEqual(binary_mask_dice(empty, empty), 1.0)

    def test_non_binary_and_shape_mismatch_fail_closed(self):
        invalid = np.zeros((4, 4), dtype=np.uint8)
        invalid[0, 0] = 2

        with self.assertRaisesRegex(ValueError, "binary"):
            binary_mask_iou(invalid, np.zeros_like(invalid))
        with self.assertRaisesRegex(ValueError, "same shape"):
            binary_mask_dice(np.zeros((4, 4)), np.zeros((5, 5)))

    def test_boundary_f1_uses_the_declared_three_pixel_tolerance(self):
        reference = np.zeros((16, 16), dtype=bool)
        prediction = np.zeros_like(reference)
        reference[4, 4] = True
        prediction[7, 7] = True

        within = boundary_precision_recall_f1(reference, prediction)
        strict = boundary_precision_recall_f1(
            reference, prediction, tolerance_pixels=2
        )

        self.assertEqual(within.f1, 1.0)
        self.assertEqual(strict.f1, 0.0)
        self.assertEqual(within.tolerance_pixels, 3)

    def test_empty_boundary_cases_are_explicit(self):
        empty = np.zeros((8, 8), dtype=bool)
        present = square(1, 3, 1, 3, size=8)

        self.assertEqual(boundary_precision_recall_f1(empty, empty).f1, 1.0)
        self.assertEqual(boundary_precision_recall_f1(present, empty).f1, 0.0)

    def test_invalid_boundary_tolerance_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            boundary_precision_recall_f1(square(), square(), tolerance_pixels=-1)

    def test_mask_spill_uses_half_open_prompt_coordinates(self):
        prediction = square(1, 5, 1, 5, size=8)
        spill = mask_spill_outside_prompt_box(prediction, [1, 1, 4, 4])

        self.assertEqual(spill.foreground_pixels, 16)
        self.assertEqual(spill.outside_prompt_pixels, 7)
        self.assertAlmostEqual(spill.outside_prompt_fraction, 7 / 16)


class MaskRouteEvaluationTests(unittest.TestCase):
    def test_route_requires_a_visible_reference_even_with_no_predictions(self):
        with self.assertRaisesRegex(ValueError, "visible reference"):
            evaluate_mask_route_image(
                opaque_scene_id="S1",
                view_id="V00",
                route="automatic_box",
                reference_masks={},
                prediction_rows=[],
            )

    def test_misses_and_unmatched_predictions_remain_in_the_summary(self):
        references = {"O1": square(), "O2": square(8, 14, 8, 14)}
        records = evaluate_mask_route_image(
            opaque_scene_id="S1",
            view_id="V00",
            route="automatic_box",
            reference_masks=references,
            prediction_rows=[
                {
                    "prediction_id": "P1",
                    "matched_reference_id": "O1",
                    "mask": references["O1"],
                    "prompt_box_xyxy": [2, 2, 8, 8],
                },
                {
                    "prediction_id": "P2",
                    "matched_reference_id": None,
                    "mask": square(0, 2, 0, 2),
                    "prompt_box_xyxy": [0, 0, 1, 1],
                },
            ],
        )
        summary = summarize_mask_routes(records)[0]

        self.assertEqual(len(records), 3)
        self.assertEqual(summary["reference_count"], 2)
        self.assertEqual(summary["prediction_count"], 2)
        self.assertEqual(summary["matched_reference_count"], 1)
        self.assertEqual(summary["missed_reference_count"], 1)
        self.assertEqual(summary["unmatched_prediction_count"], 1)
        self.assertEqual(summary["detection_aware_recall_at_0_75"], 0.5)
        self.assertEqual(summary["mean_matched_iou"], 1.0)
        self.assertEqual(summary["detection_aware_mean_iou"], 0.5)

    def test_empty_matched_prediction_is_a_zero_quality_outcome(self):
        reference = square()
        records = evaluate_mask_route_image(
            opaque_scene_id="S1",
            view_id="V00",
            route="reference_box",
            reference_masks={"O1": reference},
            prediction_rows=[
                {
                    "prediction_id": "P1",
                    "matched_reference_id": "O1",
                    "mask": np.zeros_like(reference),
                    "prompt_box_xyxy": [2, 2, 8, 8],
                }
            ],
        )

        self.assertEqual(records[0]["record_kind"], "matched_reference")
        self.assertEqual(records[0]["mask_iou"], 0.0)
        self.assertEqual(records[0]["boundary_f1"], 0.0)
        self.assertEqual(records[0]["passes_mask_iou_threshold"], 0)

    def test_duplicate_and_unknown_matches_fail_closed(self):
        reference = square()
        duplicate = [
            {
                "prediction_id": prediction_id,
                "matched_reference_id": "O1",
                "mask": reference,
                "prompt_box_xyxy": [2, 2, 8, 8],
            }
            for prediction_id in ("P1", "P2")
        ]

        with self.assertRaisesRegex(ValueError, "More than one"):
            evaluate_mask_route_image(
                opaque_scene_id="S1",
                view_id="V00",
                route="automatic_box",
                reference_masks={"O1": reference},
                prediction_rows=duplicate,
            )
        with self.assertRaisesRegex(ValueError, "unknown reference"):
            evaluate_mask_route_image(
                opaque_scene_id="S1",
                view_id="V00",
                route="automatic_box",
                reference_masks={"O1": reference},
                prediction_rows=[
                    {
                        "prediction_id": "P1",
                        "matched_reference_id": "O2",
                        "mask": reference,
                        "prompt_box_xyxy": [2, 2, 8, 8],
                    }
                ],
            )

    def test_safe_output_has_no_per_image_identifiers(self):
        reference = square()
        records = evaluate_mask_route_image(
            opaque_scene_id="S_PRIVATE",
            view_id="V_PRIVATE",
            route="reference_box",
            reference_masks={"O_PRIVATE": reference},
            prediction_rows=[
                {
                    "prediction_id": "P_PRIVATE",
                    "matched_reference_id": "O_PRIVATE",
                    "mask": reference,
                    "prompt_box_xyxy": [2, 2, 8, 8],
                }
            ],
        )

        separated = separate_mask_evidence(records)
        safe_text = json.dumps(separated["safe_route_summaries"])
        private_text = json.dumps(separated["private_records"])

        self.assertNotIn("S_PRIVATE", safe_text)
        self.assertNotIn("V_PRIVATE", safe_text)
        self.assertNotIn("O_PRIVATE", safe_text)
        self.assertNotIn("P_PRIVATE", safe_text)
        self.assertIn("S_PRIVATE", private_text)


class PairedBootstrapTests(unittest.TestCase):
    def test_default_ten_thousand_replicates_are_deterministic(self):
        rows = [
            {
                "opaque_scene_id": "S1",
                "opaque_asset_group_id": "A1",
                "baseline": 1.0,
                "candidate": 0.8,
            },
            {
                "opaque_scene_id": "S2",
                "opaque_asset_group_id": "A1",
                "baseline": 0.8,
                "candidate": 0.7,
            },
            {
                "opaque_scene_id": "S3",
                "opaque_asset_group_id": "A2",
                "baseline": 0.6,
                "candidate": 0.6,
            },
        ]

        first = paired_asset_clustered_bootstrap(
            rows, baseline_key="baseline", candidate_key="candidate"
        )
        second = paired_asset_clustered_bootstrap(
            rows, baseline_key="baseline", candidate_key="candidate"
        )

        self.assertEqual(first, second)
        self.assertEqual(first["repetitions"], DEFAULT_BOOTSTRAP_REPETITIONS)
        self.assertEqual(first["scene_count"], 3)
        self.assertEqual(first["cluster_count"], 2)
        self.assertAlmostEqual(first["mean_difference"], 0.1)

    def test_repeated_view_rows_are_averaged_within_scene(self):
        rows = [
            {
                "opaque_scene_id": "S1",
                "opaque_asset_group_id": "A1",
                "baseline": 1.0,
                "candidate": 0.5,
            },
            {
                "opaque_scene_id": "S1",
                "opaque_asset_group_id": "A1",
                "baseline": 0.0,
                "candidate": 0.5,
            },
        ]

        result = paired_asset_clustered_bootstrap(
            rows,
            baseline_key="baseline",
            candidate_key="candidate",
            repetitions=10,
        )

        self.assertEqual(result["scene_count"], 1)
        self.assertEqual(result["mean_difference"], 0.0)

    def test_scene_cannot_cross_asset_clusters(self):
        rows = [
            {
                "opaque_scene_id": "S1",
                "opaque_asset_group_id": cluster,
                "baseline": 1.0,
                "candidate": 0.5,
            }
            for cluster in ("A1", "A2")
        ]

        with self.assertRaisesRegex(ValueError, "multiple asset clusters"):
            paired_asset_clustered_bootstrap(
                rows,
                baseline_key="baseline",
                candidate_key="candidate",
                repetitions=10,
            )


if __name__ == "__main__":
    unittest.main()
