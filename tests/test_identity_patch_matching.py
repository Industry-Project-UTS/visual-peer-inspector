from __future__ import annotations

import numpy as np

from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.representation.crops import CropConfiguration, PreparedCrop
from visual_peer_inspector.representation.identity_patch_matching import (
    DESCRIPTOR_VARIANTS,
    descriptors_from_batch,
    foreground_patch_descriptor,
    match_descriptor_variant,
)


def prepared_crop(mask: np.ndarray) -> PreparedCrop:
    return PreparedCrop(
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        foreground=mask,
        source_box_xyxy=(0, 0, 224, 224),
        padded_box_xyxy=(0, 0, 224, 224),
        configuration_sha256=CropConfiguration().sha256,
    )


def test_foreground_patch_descriptor_uses_only_selected_tokens() -> None:
    mask = np.zeros((224, 224), dtype=bool)
    mask[:112, :112] = True
    patches = np.zeros((256, 3), dtype=np.float32)
    patches[:, 1] = 1.0
    for row in range(8):
        for column in range(8):
            patches[row * 16 + column] = (1.0, 0.0, 0.0)

    descriptor, fallback, selected = foreground_patch_descriptor(
        patches,
        prepared_crop(mask),
        minimum_coverage=0.50,
    )

    assert fallback is False
    assert selected == 64
    np.testing.assert_allclose(descriptor, (1.0, 0.0, 0.0))


def test_empty_patch_selection_has_deterministic_fallback() -> None:
    mask = np.zeros((224, 224), dtype=bool)
    mask[5:8, 5:8] = True
    patches = np.eye(256, 3, dtype=np.float32)

    descriptor, fallback, selected = foreground_patch_descriptor(
        patches,
        prepared_crop(mask),
        minimum_coverage=0.50,
    )

    assert fallback is True
    assert selected == 1
    assert np.isfinite(descriptor).all()


def test_batch_builds_every_declared_descriptor() -> None:
    mask = np.ones((224, 224), dtype=bool)
    batch = EncodedBatch(
        global_features=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        patch_features=(np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (256, 1)),),
        patch_grid_size=16,
        feature_dimension=3,
    )

    descriptors, fallbacks = descriptors_from_batch(batch, [prepared_crop(mask)])

    assert set(descriptors[0]) == {variant.name for variant in DESCRIPTOR_VARIANTS}
    assert all(np.isclose(np.linalg.norm(value), 1.0) for value in descriptors[0].values())
    assert sum(fallbacks.values()) == 0


def test_descriptor_matching_is_order_invariant() -> None:
    views = [
        {
            "view": "V00",
            "objects": [
                {"candidate_id": "a", "descriptors": {"local_cov25": [1.0, 0.0]}},
                {"candidate_id": "b", "descriptors": {"local_cov25": [0.0, 1.0]}},
            ],
        },
        {
            "view": "V07",
            "objects": [
                {"candidate_id": "x", "descriptors": {"local_cov25": [0.0, 1.0]}},
                {"candidate_id": "y", "descriptors": {"local_cov25": [1.0, 0.0]}},
            ],
        },
    ]

    first = match_descriptor_variant(
        views,
        descriptor_name="local_cov25",
        unmatched_cost=0.35,
    )
    second = match_descriptor_variant(
        list(reversed(views)),
        descriptor_name="local_cov25",
        unmatched_cost=0.35,
    )

    assert first == second
    groups = {
        frozenset((member["view_id"], member["candidate_id"]) for member in track["members"])
        for track in first["tracks"]
    }
    assert groups == {
        frozenset({("V00", "a"), ("V07", "y")}),
        frozenset({("V00", "b"), ("V07", "x")}),
    }
