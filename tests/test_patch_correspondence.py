from __future__ import annotations

import numpy as np

from visual_peer_inspector.representation.crops import CropConfiguration, PreparedCrop
from visual_peer_inspector.representation.patch_correspondence import (
    PATCH_CONFIGURATIONS,
    PatchAssociationConfiguration,
    associate_patch_views,
    foreground_patch_evidence,
    foreground_patch_tokens,
    reciprocal_patch_distance,
    sinkhorn_patch_distance,
    symmetric_patch_distance,
)


def _crop(mask: np.ndarray) -> PreparedCrop:
    return PreparedCrop(
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        foreground=mask,
        source_box_xyxy=(0, 0, 224, 224),
        padded_box_xyxy=(0, 0, 224, 224),
        configuration_sha256=CropConfiguration().sha256,
    )


def _candidate(candidate_id: str, global_values: list[float], patch_values: list[list[float]]) -> dict:
    return {
        "candidate_id": candidate_id,
        "global_embedding": global_values,
        "patch_tokens": patch_values,
    }


def test_foreground_tokens_exclude_background_patches() -> None:
    mask = np.zeros((224, 224), dtype=bool)
    mask[:112, :112] = True
    patches = np.zeros((256, 3), dtype=np.float32)
    patches[:, 1] = 1.0
    for row in range(8):
        for column in range(8):
            patches[row * 16 + column] = (1.0, 0.0, 0.0)

    selected, fallback = foreground_patch_tokens(patches, _crop(mask))

    assert fallback is False
    assert selected.shape == (64, 3)
    np.testing.assert_allclose(selected, np.tile((1.0, 0.0, 0.0), (64, 1)))


def test_foreground_patch_weights_follow_mask_occupancy() -> None:
    mask = np.zeros((224, 224), dtype=bool)
    mask[:14, :14] = True
    mask[14:28, :7] = True
    patches = np.tile((1.0, 0.0), (256, 1))

    selected, weights, fallback = foreground_patch_evidence(
        patches,
        _crop(mask),
        minimum_coverage=0.25,
    )

    assert fallback is False
    assert selected.shape == (2, 2)
    np.testing.assert_allclose(weights, [2 / 3, 1 / 3])


def test_patch_distance_is_symmetric_and_order_invariant() -> None:
    left = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
    right = [[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]]

    forward = symmetric_patch_distance(left, right, retained_fraction=1.0)
    reverse = symmetric_patch_distance(list(reversed(right)), left, retained_fraction=1.0)

    assert np.isclose(forward, 0.0)
    assert np.isclose(forward, reverse)


def test_patch_distance_separates_nonmatching_token_sets() -> None:
    reference = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
    match = [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]]
    other = [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]

    matching_distance = symmetric_patch_distance(reference, match, retained_fraction=0.75)
    other_distance = symmetric_patch_distance(reference, other, retained_fraction=0.75)

    assert matching_distance < other_distance


def test_reciprocal_and_sinkhorn_distances_separate_matches() -> None:
    reference = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    match = [[0.95, 0.05], [0.0, 1.0], [1.0, 0.0]]
    other = [[-1.0, 0.0], [-0.9, -0.1], [0.0, -1.0]]

    reciprocal_match = reciprocal_patch_distance(reference, match)
    reciprocal_other = reciprocal_patch_distance(reference, other)
    sinkhorn_match = sinkhorn_patch_distance(reference, match)
    sinkhorn_other = sinkhorn_patch_distance(reference, other)

    assert reciprocal_match["patch_distance"] < reciprocal_other["patch_distance"]
    assert sinkhorn_match["patch_distance"] < sinkhorn_other["patch_distance"]
    assert 0.0 <= sinkhorn_match["patch_match_fraction"] <= 1.0
    assert 0.0 <= sinkhorn_match["patch_entropy"] <= 1.0


def test_association_uses_every_candidate_once_with_one_per_view() -> None:
    configuration = PatchAssociationConfiguration(
        "test_consensus",
        patch_metric="reciprocal",
        global_weight=0.5,
        maximum_cost=0.4,
        solver="consensus",
    )
    views = [
        {
            "view": "V00",
            "objects": [
                _candidate("a", [1.0, 0.0], [[1.0, 0.0], [0.9, 0.1]]),
                _candidate("b", [0.0, 1.0], [[0.0, 1.0], [0.1, 0.9]]),
            ],
        },
        {
            "view": "V07",
            "objects": [
                _candidate("x", [0.0, 1.0], [[0.1, 0.9], [0.0, 1.0]]),
                _candidate("y", [1.0, 0.0], [[0.9, 0.1], [1.0, 0.0]]),
            ],
        },
        {
            "view": "V14",
            "objects": [
                _candidate("m", [1.0, 0.0], [[1.0, 0.0], [0.8, 0.2]]),
                _candidate("n", [0.0, 1.0], [[0.0, 1.0], [0.2, 0.8]]),
            ],
        },
    ]

    result = associate_patch_views(views, configuration)
    members = [
        (item["view_id"], item["candidate_id"])
        for track in result["tracks"]
        for item in track["members"]
    ]

    assert len(members) == len(set(members)) == 6
    assert all(
        len({item["view_id"] for item in track["members"]}) == len(track["members"])
        for track in result["tracks"]
    )
    groups = {
        frozenset((item["view_id"], item["candidate_id"]) for item in track["members"])
        for track in result["tracks"]
    }
    assert groups == {
        frozenset({("V00", "a"), ("V07", "y"), ("V14", "m")}),
        frozenset({("V00", "b"), ("V07", "x"), ("V14", "n")}),
    }
    assert len(result["accepted_links"]) == 4
    assert all(link["margin"] >= 0.0 for link in result["accepted_links"])


def test_association_is_input_order_invariant() -> None:
    configuration = PATCH_CONFIGURATIONS[1]
    views = [
        {"view": "V00", "objects": [_candidate("a", [1.0, 0.0], [[1.0, 0.0]])]},
        {"view": "V07", "objects": [_candidate("b", [1.0, 0.0], [[1.0, 0.0]])]},
    ]

    first = associate_patch_views(views, configuration)
    second = associate_patch_views(list(reversed(views)), configuration)

    assert first == second


def test_configuration_screen_is_bounded() -> None:
    assert len(PATCH_CONFIGURATIONS) == 3
    assert {item.patch_metric for item in PATCH_CONFIGURATIONS} == {"reciprocal", "sinkhorn"}
    assert {item.solver for item in PATCH_CONFIGURATIONS} == {"anchor", "consensus"}
