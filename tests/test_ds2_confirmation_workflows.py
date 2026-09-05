from __future__ import annotations

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.workflows.multiview_confirmation_ds2 import (
    VIEW_COUNTS,
    fixed_camera_spread_views,
    fuse_object_ranks,
)
from visual_peer_inspector.workflows.representation_confirmation_ds2 import (
    MODEL_KEY,
    MODEL_REVISION,
)


def test_ds2_contract_and_model_freeze_are_exact() -> None:
    assert DS2_CONFIRM_CONTRACT.scene_count == 300
    assert DS2_CONFIRM_CONTRACT.image_count == 6_000
    assert DS2_CONFIRM_CONTRACT.asset_group_count == 82
    assert DS2_CONFIRM_CONTRACT.membership_sha256 == "bc00e95781a6d52c25c9e40364f8dfa36ad7874cccd759488ac87ce55dec2eac"
    assert MODEL_KEY == "dinov2_small"
    assert MODEL_REVISION == "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def test_camera_spread_subsets_are_fixed_and_nested_at_endpoints() -> None:
    assert VIEW_COUNTS == (1, 2, 3, 5, 10, 20)
    assert fixed_camera_spread_views(1) == ("V00",)
    assert fixed_camera_spread_views(3) == ("V00", "V06", "V13")
    assert fixed_camera_spread_views(5) == ("V00", "V04", "V08", "V12", "V16")
    assert fixed_camera_spread_views(20) == tuple(f"V{index:02d}" for index in range(20))


def test_rank_fusion_uses_stable_objects_and_visibility_weights() -> None:
    rows = []
    ranks = {"O01": (1, 3), "O02": (2, 2), "O03": (3, 1)}
    areas = {"O01": (100, 100), "O02": (100, 100), "O03": (100, 10_000)}
    for view_index, view_id in enumerate(("V00", "V10")):
        for object_id in ("O01", "O02", "O03"):
            rows.append({
                "view_id": view_id,
                "opaque_object_id": object_id,
                "rank": ranks[object_id][view_index],
                "candidate_count": 3,
                "foreground_pixels": areas[object_id][view_index],
            })

    mean_rows, disagreement = fuse_object_ranks(
        rows,
        selected_views=("V00", "V10"),
        fusion_method="mean_rank",
    )
    weighted_rows, _ = fuse_object_ranks(
        rows,
        selected_views=("V00", "V10"),
        fusion_method="visibility_weighted_rank",
    )

    assert disagreement > 0
    assert [row["opaque_object_id"] for row in mean_rows] == ["O01", "O02", "O03"]
    assert weighted_rows[0]["opaque_object_id"] == "O03"
