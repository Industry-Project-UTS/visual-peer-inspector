from __future__ import annotations

import numpy as np
import pytest

from visual_peer_inspector.representation.identity_association import AssociationConfiguration, associate_views


def object_record(name: str, values: list[float]) -> dict[str, object]:
    return {"candidate_id": name, "embedding": values}


def test_reciprocal_matching_is_deterministic_and_one_per_view() -> None:
    views = [
        {"view": "V00", "objects": [object_record("A", [1.0, 0.0]), object_record("B", [0.0, 1.0])]},
        {"view": "V01", "objects": [object_record("C", [0.99, 0.01]), object_record("D", [0.01, 0.99])]},
        {"view": "V02", "objects": [object_record("E", [1.0, 0.0]), object_record("F", [0.0, 1.0])]},
    ]
    configuration = AssociationConfiguration("reciprocal", "reciprocal", 0.2)
    first = associate_views(views, configuration)
    second = associate_views(list(reversed(views)), configuration)
    assert first == second
    assert sorted(len(track["members"]) for track in first["tracks"]) == [3, 3]
    for track in first["tracks"]:
        members = track["members"]
        assert len({member["view_id"] for member in members}) == len(members)


def test_cycle_pruning_rejects_unsupported_pair() -> None:
    views = [
        {"view": "V00", "objects": [object_record("A", [1.0, 0.0])]},
        {"view": "V01", "objects": [object_record("B", [0.98, 0.2])]},
        {"view": "V02", "objects": [object_record("C", [0.0, 1.0])]},
    ]
    result = associate_views(views, AssociationConfiguration("cycle", "cycle_pruned", 0.25))
    assert all(len(track["members"]) == 1 for track in result["tracks"])


def test_rejects_zero_or_inconsistent_embeddings() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        associate_views(
            [
                {"view": "V00", "objects": [object_record("A", [0.0, 0.0])]},
                {"view": "V01", "objects": [object_record("B", [1.0, 0.0])]},
            ],
            AssociationConfiguration("x", "reciprocal", 0.2),
        )
    with pytest.raises(ValueError, match="same dimension"):
        associate_views(
            [
                {"view": "V00", "objects": [object_record("A", [1.0, 0.0])]},
                {"view": "V01", "objects": [object_record("B", [1.0, 0.0, 0.0])]},
            ],
            AssociationConfiguration("x", "reciprocal", 0.2),
        )


def test_support_consensus_reports_finite_diagnostics() -> None:
    views = [
        {"view": "V00", "objects": [object_record("A", [1.0, 0.0])]},
        {"view": "V01", "objects": [object_record("B", [1.0, 0.0])]},
        {"view": "V02", "objects": [object_record("C", [1.0, 0.0])]},
    ]
    result = associate_views(views, AssociationConfiguration("support", "support_consensus", 0.2))
    assert len(result["tracks"]) == 1
    assert all(np.isfinite(value) for value in result["diagnostics"].values())


def test_empty_views_remain_a_valid_failed_proposal_case() -> None:
    views = [{"view": "V00", "objects": []}, {"view": "V10", "objects": []}]
    for strategy in ("current", "reciprocal", "cycle_pruned", "support_consensus"):
        result = associate_views(views, AssociationConfiguration(strategy, strategy, 0.35))
        assert result["tracks"] == []
        assert result["diagnostics"]["linked_node_fraction"] == 0.0
