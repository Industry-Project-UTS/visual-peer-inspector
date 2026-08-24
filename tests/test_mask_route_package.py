import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from visual_peer_inspector.data import mask_route_package
from visual_peer_inspector.models.sam2 import (
    binary_mask_sha256,
    encode_binary_mask,
)


def _mask_item(prompt_key, mask, *, predicted=False):
    item = {
        "prompt_key": prompt_key,
        "box_xyxy": [4, 5, 20, 21],
        "foreground_pixels": int(mask.sum()),
        "mask_status": "non_empty",
        "mask_sha256": binary_mask_sha256(mask),
        "mask_rle": encode_binary_mask(mask),
    }
    if predicted:
        item["predicted_quality_score"] = 0.91
    return item


class MaskRoutePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary.name) / "routes"
        mask = np.zeros((256, 256), dtype=bool)
        mask[5:21, 4:20] = True
        self.rows = [
            {
                "schema_version": 1,
                "opaque_scene_id": "S000001",
                "view_id": "V00",
                "image_sha256": "a" * 64,
                "routes": {
                    "supplied_mask": [_mask_item("R001", mask)],
                    "reference_box": [
                        _mask_item("R001", mask, predicted=True)
                    ],
                    "automatic_box": [
                        _mask_item("P001", mask, predicted=True),
                        _mask_item("P002", mask, predicted=True),
                    ],
                },
            }
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def _small_contract(self):
        return mock.patch.multiple(
            mask_route_package,
            EXPECTED_SCENES=1,
            EXPECTED_IMAGES=1,
            EXPECTED_ROUTE_COUNTS={
                "supplied_mask": 1,
                "reference_box": 1,
                "automatic_box": 2,
            },
        )

    def _build(self):
        with self._small_contract():
            return mask_route_package.build_mask_route_package(
                rows=self.rows,
                output_root=self.output_root,
                provenance={
                    "evaluator_snapshot_sha256": "a" * 64,
                    "sam_run_record_sha256": "b" * 64,
                    "sam_prompt_package_sha256": "c" * 64,
                    "proposal_results_sha256": "d" * 64,
                    "supplied_mask_record_sha256": "e" * 64,
                },
            )

    def test_complete_label_free_package_verifies(self):
        record = self._build()

        with self._small_contract():
            verified = mask_route_package.verify_mask_route_package(
                self.output_root
            )

        self.assertEqual(record, verified)
        self.assertTrue(verified["model_facing_allowed"])
        self.assertEqual(
            verified["route_counts"],
            {
                "supplied_mask": 1,
                "reference_box": 1,
                "automatic_box": 2,
            },
        )
        package_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in self.output_root.iterdir()
            if path.suffix in {".json", ".jsonl"}
        ).lower()
        self.assertNotIn("matched_object", package_text)
        self.assertNotIn("category", package_text)
        self.assertNotIn("anomaly", package_text)

    def test_protected_fields_fail_before_publication(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["category"] = "private"

        with self._small_contract(), self.assertRaisesRegex(
            ValueError, "Protected key"
        ):
            mask_route_package.build_mask_route_package(
                rows=rows,
                output_root=self.output_root,
                provenance={
                    "evaluator_snapshot_sha256": "a" * 64,
                    "sam_run_record_sha256": "b" * 64,
                    "sam_prompt_package_sha256": "c" * 64,
                    "proposal_results_sha256": "d" * 64,
                    "supplied_mask_record_sha256": "e" * 64,
                },
            )

        self.assertFalse(self.output_root.exists())

    def test_manifest_tampering_is_detected(self):
        self._build()
        manifest = self.output_root / "mask_route_manifest.jsonl"
        manifest.write_bytes(manifest.read_bytes() + b"\n")

        with self._small_contract(), self.assertRaisesRegex(
            ValueError, "manifest hash differs"
        ):
            mask_route_package.verify_mask_route_package(self.output_root)

    def test_package_root_symlink_is_rejected(self):
        self._build()
        symlink = Path(self.temporary.name) / "route-link"
        symlink.symlink_to(self.output_root, target_is_directory=True)

        with self._small_contract(), self.assertRaisesRegex(
            ValueError, "missing or unsafe"
        ):
            mask_route_package.verify_mask_route_package(symlink)


if __name__ == "__main__":
    unittest.main()
