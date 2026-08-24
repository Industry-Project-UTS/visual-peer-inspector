import unittest

import numpy as np

from visual_peer_inspector.models.sam2 import (
    CONTRACT_PRODUCER_KIND,
    Sam2Configuration,
    binary_mask_sha256,
    encode_binary_mask,
)
from visual_peer_inspector.workflows.mask_evaluation_ds0 import (
    CONTRACT_TEST_MODE,
    MODEL_EVIDENCE_MODE,
    _qualitative_case_rows,
    _validate_sam_run_record,
    _validated_evaluator_provenance,
    _validated_prediction,
)


class MaskEvaluationInputValidationTests(unittest.TestCase):
    def _evaluator_provenance(self):
        return {
            "git_commit": "a" * 40,
            "git_dirty": True,
            "snapshot_sha256": "b" * 64,
            "precommit_candidate": True,
            "notebook_sha256": "c" * 64,
            "kaggle_notebook_slug": (
                "esreekarreddy/toysad-8k-ds0-proposal-and-mask-evaluation"
            ),
            "kaggle_notebook_version": "1",
            "code_dataset_version": "19",
        }

    def _contract_record(self):
        configuration = Sam2Configuration()
        return {
            "status": "completed",
            "inference_stage": "full",
            "completion_gate_passed": True,
            "reference_prompt_count": 1640,
            "automatic_prompt_count": 1616,
            "prompt_package_sha256": "a" * 64,
            "run_mode": "contract_smoke",
            "producer_kind": CONTRACT_PRODUCER_KIND,
            "checkpoint_sha256": None,
            "summary": {
                "scene_count": 20,
                "image_count": 400,
                "mask_count": 3256,
            },
            "repeat_agreement": {
                "mask_digest_agreement_rate": 1.0,
                "score_agreement_rate": 1.0,
            },
            "configuration": configuration.to_dict(),
            "configuration_sha256": configuration.sha256,
        }

    def test_contract_output_is_accepted_only_in_contract_mode(self):
        record = self._contract_record()

        _validate_sam_run_record(
            record,
            evaluation_mode=CONTRACT_TEST_MODE,
            expected_prompt_package_sha256="a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "model_evidence"):
            _validate_sam_run_record(
                record,
                evaluation_mode=MODEL_EVIDENCE_MODE,
                expected_prompt_package_sha256="a" * 64,
            )

    def test_frozen_rle_and_digest_must_agree(self):
        mask = np.zeros((256, 256), dtype=bool)
        mask[8:24, 7:30] = True
        item = {
            "box_xyxy": [7, 8, 30, 24],
            "foreground_pixels": int(mask.sum()),
            "mask_status": "non_empty",
            "mask_sha256": binary_mask_sha256(mask),
            "mask_rle": encode_binary_mask(mask),
            "predicted_quality_score": 0.85,
        }

        validated = _validated_prediction(item, expected_box=item["box_xyxy"])
        self.assertTrue(np.array_equal(validated["mask"], mask))
        item["mask_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "recorded SHA-256"):
            _validated_prediction(item, expected_box=item["box_xyxy"])

    def test_prompt_geometry_mismatch_fails_before_evaluation(self):
        mask = np.zeros((256, 256), dtype=bool)
        item = {
            "box_xyxy": [1, 1, 2, 2],
            "foreground_pixels": 0,
            "mask_status": "empty",
            "mask_sha256": binary_mask_sha256(mask),
            "mask_rle": encode_binary_mask(mask),
            "predicted_quality_score": 0.1,
        }

        with self.assertRaisesRegex(ValueError, "frozen prompt"):
            _validated_prediction(item, expected_box=[2, 2, 3, 3])

    def test_unexpected_rle_dimensions_fail_before_decode(self):
        item = {
            "box_xyxy": [1, 1, 2, 2],
            "foreground_pixels": 0,
            "mask_status": "empty",
            "mask_sha256": "0" * 64,
            "mask_rle": {"size": [1_000_000, 1_000_000], "counts": [10**12]},
            "predicted_quality_score": 0.1,
        }

        with self.assertRaisesRegex(ValueError, "unexpected dimensions"):
            _validated_prediction(item, expected_box=item["box_xyxy"])

    def test_evaluator_provenance_accepts_the_pinned_schema(self):
        provenance = self._evaluator_provenance()

        self.assertEqual(_validated_evaluator_provenance(provenance), provenance)

    def test_evaluator_provenance_rejects_missing_and_invalid_values(self):
        provenance = self._evaluator_provenance()
        provenance.pop("notebook_sha256")
        with self.assertRaisesRegex(ValueError, "accepted schema"):
            _validated_evaluator_provenance(provenance)

        provenance = self._evaluator_provenance()
        provenance["snapshot_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(ValueError, "snapshot_sha256"):
            _validated_evaluator_provenance(provenance)


class MaskQualitativeSelectionTests(unittest.TestCase):
    def test_selection_covers_strong_typical_and_challenging_ranks(self):
        records = [
            {
                "route": "automatic_box",
                "record_kind": "matched_reference",
                "mask_iou": index / 10,
                "opaque_scene_id": f"S{index:06d}",
                "view_id": f"V{index:02d}",
                "opaque_object_id": "O00",
            }
            for index in range(10)
        ]

        selected = _qualitative_case_rows(records)

        self.assertEqual([group for group, _row in selected], [
            "Strong",
            "Strong",
            "Typical",
            "Typical",
            "Challenging",
            "Challenging",
        ])
        self.assertEqual([row["mask_iou"] for _group, row in selected[:2]], [0.9, 0.8])
        self.assertEqual([row["mask_iou"] for _group, row in selected[-2:]], [0.0, 0.1])

    def test_selection_rejects_too_few_matched_masks(self):
        with self.assertRaisesRegex(ValueError, "six"):
            _qualitative_case_rows([])


if __name__ == "__main__":
    unittest.main()
