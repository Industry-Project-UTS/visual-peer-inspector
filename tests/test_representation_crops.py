import unittest

import numpy as np

from visual_peer_inspector.representation.crops import (
    CropConfiguration,
    prepare_masked_crop,
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


if __name__ == "__main__":
    unittest.main()
