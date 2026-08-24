import unittest

import numpy as np

from visual_peer_inspector.models.visual_encoder import (
    ContractVisualEncoder,
    ENCODER_REGISTRY,
    encoder_registry_sha256,
)


class VisualEncoderContractTests(unittest.TestCase):
    def test_registry_contains_the_anchor_and_dinov3_challengers(self):
        keys = {item.key for item in ENCODER_REGISTRY}

        self.assertIn("dinov2_small", keys)
        self.assertIn("dinov2_small_registers", keys)
        self.assertIn("dinov3_small", keys)
        self.assertIn("dinov3_small_plus", keys)
        self.assertIn("resnet50_control", keys)
        self.assertEqual(len(encoder_registry_sha256()), 64)

    def test_contract_encoder_is_finite_normalized_and_repeatable(self):
        encoder = ContractVisualEncoder()
        crops = [np.full((224, 224, 3), value, dtype=np.uint8) for value in (30, 90, 180)]

        first = encoder.encode(crops)
        second = encoder.encode(crops)

        self.assertEqual(first.global_features.shape, (3, 6))
        self.assertEqual(first.patch_features[0].shape, (256, 6))
        self.assertEqual(first.patch_grid_size, 16)
        self.assertTrue(np.all(np.isfinite(first.global_features)))
        self.assertTrue(np.allclose(np.linalg.norm(first.global_features, axis=1), 1.0))
        self.assertTrue(np.array_equal(first.global_features, second.global_features))


if __name__ == "__main__":
    unittest.main()
