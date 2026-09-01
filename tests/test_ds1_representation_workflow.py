from __future__ import annotations

import unittest

import numpy as np

from visual_peer_inspector.representation.crops import prepare_masked_crop
from visual_peer_inspector.workflows.representation_ds1 import (
    _encode_prepared_in_batches,
    _patch_validity_with_small_object_fallback,
    _score_rows,
)
from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.data.representation_package_ds1 import (
    RepresentationCandidate,
)


class DS1RepresentationWorkflowTests(unittest.TestCase):
    def test_scene_encoder_uses_bounded_batches(self) -> None:
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=bool)
        prepared = [prepare_masked_crop(image, mask) for _ in range(5)]

        class FakeEncoder:
            def __init__(self) -> None:
                self.calls = 0

            def encode(self, crops):
                self.calls += 1
                features = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(crops), 1))
                return EncodedBatch(features, (), None, 2)

        encoder = FakeEncoder()
        features, patches, grid, dimension, calls = _encode_prepared_in_batches(
            encoder, prepared, batch_size=2
        )

        self.assertEqual(features.shape, (5, 2))
        self.assertEqual(patches, ())
        self.assertIsNone(grid)
        self.assertEqual(dimension, 2)
        self.assertEqual(calls, 3)
        self.assertEqual(encoder.calls, 3)

    def test_small_object_uses_one_deterministic_patch(self) -> None:
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        mask[2, 2] = True
        mask[2, 29] = True
        mask[29, 2] = True
        mask[29, 29] = True
        crop = prepare_masked_crop(image, mask)

        valid, used_fallback = _patch_validity_with_small_object_fallback(crop, 16)

        self.assertTrue(used_fallback)
        self.assertEqual(int(valid.sum()), 1)

    def test_geometry_rows_are_emitted_only_by_the_anchor(self) -> None:
        candidates = [
            RepresentationCandidate(
                candidate_id=f"C{index}",
                opaque_scene_id="S000001",
                view_id="V00",
                route="supplied_mask",
                prompt_key=f"R{index:03d}",
                image_path=None,
                image_sha256="0" * 64,
                box_xyxy=(index, index, index + 4, index + 4),
                foreground_pixels=16,
                mask_sha256="1" * 64,
                mask_rle={},
                predicted_quality_score=None,
            )
            for index in range(3)
        ]
        features = np.eye(3, dtype=np.float64)

        anchor_rows, _ = _score_rows(
            candidates=candidates,
            encoder_key="dinov2_small",
            global_features=features,
            patch_features=(),
            valid_patch_masks=(),
        )
        challenger_rows, _ = _score_rows(
            candidates=candidates,
            encoder_key="dinov2_base",
            global_features=features,
            patch_features=(),
            valid_patch_masks=(),
        )

        self.assertEqual({row["method"] for row in anchor_rows}, {"B0", "B1", "P1", "shape_B1", "layout_B1"})
        self.assertEqual({row["method"] for row in challenger_rows}, {"B0", "B1", "P1"})


if __name__ == "__main__":
    unittest.main()
