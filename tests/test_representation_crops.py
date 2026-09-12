import unittest

import numpy as np

from visual_peer_inspector.representation.crops import (
    CropConfiguration,
    prepare_masked_crop,
    project_crop_heatmap,
)


class RepresentationCropTests(unittest.TestCase):
    def _fixture(self):
        image = np.zeros((32, 40, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(40, dtype=np.uint8)
        image[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None]
        image[:, :, 2] = 120
        mask = np.zeros((32, 40), dtype=bool)
        mask[6:25, 11:30] = True
        mask[10:13, 14:17] = False
        return image, mask

    def test_crop_matches_the_golden_transform(self):
        image, mask = self._fixture()
        crop = prepare_masked_crop(image, mask)

        self.assertEqual(crop.source_box_xyxy, (11, 6, 30, 25))
        self.assertEqual(crop.padded_box_xyxy, (8, 3, 33, 28))
        self.assertEqual(crop.rgb.shape, (224, 224, 3))
        self.assertEqual(int(crop.foreground.sum()), 28_171)
        self.assertEqual(
            crop.sha256,
            "ab311f35ea966ab6c8631ce7cc19b6a2b931c5f3f67a62e6f6f680979c0fda87",
        )

    def test_patch_validity_excludes_mostly_background_patches(self):
        image, mask = self._fixture()
        crop = prepare_masked_crop(image, mask)

        valid = crop.patch_validity(16, minimum_fraction=0.50)

        self.assertEqual(valid.shape, (16, 16))
        self.assertEqual(int(valid.sum()), 142)

    def test_crop_rejects_empty_and_tiny_masks(self):
        image, _mask = self._fixture()
        empty = np.zeros(image.shape[:2], dtype=bool)
        tiny = empty.copy()
        tiny[0, :3] = True

        with self.assertRaisesRegex(ValueError, "too little"):
            prepare_masked_crop(image, empty)
        with self.assertRaisesRegex(ValueError, "too little"):
            prepare_masked_crop(image, tiny)

    def test_configuration_hash_changes_with_geometry(self):
        first = CropConfiguration()
        second = CropConfiguration(padding_fraction=0.20)

        self.assertNotEqual(first.sha256, second.sha256)

    def test_patch_heatmap_stays_inside_the_offset_object(self):
        image, mask = self._fixture()
        crop = prepare_masked_crop(image, mask)
        projected = project_crop_heatmap(np.full((16, 16), 0.7), crop, mask)
        self.assertEqual(projected.shape, mask.shape)
        self.assertTrue(np.isnan(projected[~mask]).all())
        np.testing.assert_allclose(projected[mask], 0.7, atol=1e-6)

    def test_patch_heatmap_preserves_direction_and_clips_padded_edges(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        mask = np.zeros((40, 60), dtype=bool)
        mask[:20, :20] = True
        crop = prepare_masked_crop(image, mask)
        self.assertLess(crop.padded_box_xyxy[0], 0)
        heatmap = np.tile(np.linspace(0, 1, 16), (16, 1))
        projected = project_crop_heatmap(heatmap, crop, mask)
        self.assertLess(projected[10, 3], projected[10, 16])
        self.assertTrue(np.isfinite(projected[mask]).all())
        self.assertTrue(np.isnan(projected[~mask]).all())

    def test_invalid_patches_are_not_presented_as_measured_evidence(self):
        image, mask = self._fixture()
        crop = prepare_masked_crop(image, mask)
        heatmap = np.full((16, 16), np.nan)
        heatmap[:, 8:] = 0.9
        projected = project_crop_heatmap(heatmap, crop, mask)
        self.assertTrue(np.isnan(projected[18, 12]))
        self.assertAlmostEqual(float(projected[18, 26]), 0.9, places=5)
        self.assertTrue(
            np.isnan(project_crop_heatmap(np.full((16, 16), np.nan), crop, mask)).all()
        )

    def test_projection_rejects_invalid_geometry_and_mask(self):
        image, mask = self._fixture()
        crop = prepare_masked_crop(image, mask)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            project_crop_heatmap(np.zeros((2, 2, 3)), crop, mask)
        with self.assertRaisesRegex(ValueError, "binary"):
            project_crop_heatmap(np.zeros((16, 16)), crop, mask.astype(float) + 0.5)


if __name__ == "__main__":
    unittest.main()
