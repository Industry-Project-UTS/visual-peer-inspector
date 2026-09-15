"""Scientific invariants for identity-separated patch reference banks."""

from __future__ import annotations

from copy import deepcopy
import itertools
import json
import math

import numpy as np
import pytest

from vpi_object_bank.scoring import score_scene


def observation(view, candidate, identity, patches):
    return {"view_id": view, "candidate_id": candidate, "object_id": identity, "patches": np.asarray(patches, dtype=float)}


def record(result, view, candidate):
    return next(item for item in result["observations"] if (item["view_id"], item["candidate_id"]) == (view, candidate))


def object_record(result, identity):
    return next(item for item in result["objects"] if item["object_id"] == identity)


def three_objects(view="V00"):
    return [
        observation(view, "c1", "A", [[1, 0]]),
        observation(view, "c2", "B", [[0, 1]]),
        observation(view, "c3", "C", [[-1, 0]]),
    ]


def test_single_view_bank_matches_forward_lower_median_not_average_median():
    result = score_scene(three_objects(), selected_views=["V00"])
    # A's two peer distances are 1 and 2: the established lower median is 1.
    target = record(result, "V00", "c1")
    assert target["within_view"]["score"] == pytest.approx(1.0)
    assert target["within_view"] == target["object_bank"]
    assert target["within_view"] == target["view_mean_control"]
    assert all(item["within_view"] == item["object_bank"] for item in result["observations"])


@pytest.mark.parametrize("fraction,expected", [(0.02, 2.0), (0.5, 1.5), (1.0, 0.75)])
def test_upper_tail_uses_floor_rounding_and_at_least_one_patch(fraction, expected):
    rows = [
        observation("V00", "a", "A", [[1, 0], [1, 0], [0, 1], [-1, 0]]),
        observation("V00", "b", "B", [[1, 0]]),
        observation("V00", "c", "C", [[1, 0]]),
    ]
    result = score_scene(rows, selected_views=["V00"], top_fraction=fraction)
    assert record(result, "V00", "a")["object_bank"]["patch_scores"] == [0.0, 0.0, 1.0, 2.0]
    assert record(result, "V00", "a")["object_bank"]["score"] == pytest.approx(expected)


def test_baseline_matches_independent_scalar_forward_reference():
    # Unequal patch counts and non-unit rows exercise the same inputs as crops.
    rng = np.random.default_rng(401)
    matrices = [rng.normal(size=(count, 5)) * 3 for count in [7, 4, 6, 3, 8]]
    rows = [observation("V00", f"c{i}", f"O{i}", matrix) for i, matrix in enumerate(matrices)]
    result = score_scene(rows, selected_views=["V00"], top_fraction=0.4)
    units = [[tuple(row / np.linalg.norm(row)) for row in matrix] for matrix in matrices]
    for index, item in enumerate(result["observations"]):
        patch_scores = []
        for target in units[index]:
            per_peer = [
                min(1.0 - sum(x * y for x, y in zip(target, candidate)) for candidate in peer)
                for peer_index, peer in enumerate(units) if peer_index != index
            ]
            patch_scores.append(sorted(per_peer)[(len(per_peer) - 1) // 2])
        tail = sorted(patch_scores, reverse=True)[:max(1, math.floor(len(patch_scores) * 0.4))]
        assert item["within_view"]["patch_scores"] == pytest.approx(patch_scores)
        assert item["within_view"]["score"] == pytest.approx(sum(tail) / len(tail))


def test_entire_target_identity_is_excluded_from_its_reference_banks():
    rows = three_objects()
    before = score_scene(rows, selected_views=["V00", "V01"])
    rows.append(observation("V01", "extra", "A", [[1, 0]]))
    after = score_scene(rows, selected_views=["V00", "V01"])
    assert record(before, "V00", "c1")["object_bank"] == record(after, "V00", "c1")["object_bank"]
    assert record(before, "V00", "c1")["view_mean_control"] == record(after, "V00", "c1")["view_mean_control"]
    assert record(after, "V00", "c1")["object_bank"]["score"] == pytest.approx(1.0)


def test_other_view_of_peer_can_explain_pose_without_self_matching():
    rows = three_objects()
    before = score_scene(rows, selected_views=["V00", "V01"])
    rows.append(observation("V01", "b_rotated", "B", [[1, 0]]))
    after = score_scene(rows, selected_views=["V00", "V01"])
    assert record(before, "V00", "c1")["object_bank"]["score"] == pytest.approx(1.0)
    assert record(after, "V00", "c1")["object_bank"]["score"] == pytest.approx(0.0)
    assert record(before, "V00", "c1")["within_view"] == record(after, "V00", "c1")["within_view"]


def test_distinct_peer_banks_have_equal_weight_despite_repeated_views():
    rows = [
        observation("V00", "a", "A", [[1, 0]]),
        observation("V00", "c", "C", [[-1, 0]]),
        observation("V00", "d", "D", [[-1, 0]]),
    ]
    views = ["V00"]
    for index in range(1, 7):
        view = f"V{index:02d}"
        views.append(view)
        rows.append(observation(view, "b", "B", [[1, 0]]))
    result = score_scene(rows, selected_views=views)
    target = record(result, "V00", "a")["object_bank"]
    # Three physical peers produce [0, 2, 2], regardless of B's six views.
    assert target["peer_count"] == 3
    assert target["score"] == pytest.approx(2.0)


def test_union_patch_distances_cannot_increase_when_a_peer_gains_views():
    rows = [
        observation("V00", "a", "A", [[1, 0], [0, 1], [-1, 0]]),
        observation("V00", "b", "B", [[-1, 0]]),
        observation("V00", "c", "C", [[-1, 0]]),
    ]
    before = score_scene(rows, selected_views=["V00", "V01"], top_fraction=1.0)
    rows.extend([
        observation("V01", "b", "B", [[0, 1]]),
        observation("V01", "c", "C", [[0, 1]]),
    ])
    after = score_scene(rows, selected_views=["V00", "V01"], top_fraction=1.0)
    first = record(before, "V00", "a")["object_bank"]
    second = record(after, "V00", "a")["object_bank"]
    assert first["patch_scores"] == [2.0, 1.0, 0.0]
    assert second["patch_scores"] == [1.0, 0.0, 0.0]
    assert np.all(np.asarray(second["patch_scores"]) <= first["patch_scores"])
    assert second["score"] < first["score"]


def test_view_mean_control_can_increase_or_decrease_as_views_are_added():
    rows = [
        observation("V00", "a", "A", [[1, 0]]),
        observation("V00", "b", "B", [[1, 0]]),
        observation("V00", "c", "C", [[1, 0]]),
    ]
    selected = ["V00", "V01", "V02"]
    initial = score_scene(rows, selected_views=selected)
    rows.extend([
        observation("V01", "b", "B", [[-1, 0]]),
        observation("V01", "c", "C", [[-1, 0]]),
    ])
    after_different_angle = score_scene(rows, selected_views=selected)
    rows.extend([
        observation("V02", "b", "B", [[1, 0]]),
        observation("V02", "c", "C", [[1, 0]]),
    ])
    after_matching_angle = score_scene(rows, selected_views=selected)
    controls = [record(result, "V00", "a")["view_mean_control"]["score"] for result in [initial, after_different_angle, after_matching_angle]]
    assert controls == pytest.approx([0.0, 1.0, 2.0 / 3.0])
    assert all(record(result, "V00", "a")["object_bank"]["score"] == 0.0 for result in [initial, after_different_angle, after_matching_angle])


def test_view_mean_weights_views_equally_and_exposes_reference_token_counts():
    rows = [observation("V00", "a", "A", [[1, 0]])]
    for identity in ["B", "C"]:
        rows.extend([
            observation("V00", identity, identity, [[1, 0]] * 10),
            observation("V01", identity, identity, [[-1, 0]]),
            observation("V02", identity, identity, np.empty((0, 2))),
        ])
    result = score_scene(rows, selected_views=["V00", "V01", "V02"])
    target = record(result, "V00", "a")
    assert target["object_bank"]["score"] == 0.0
    assert target["view_mean_control"]["score"] == pytest.approx(1.0)
    assert target["peer_token_counts"] == {"B": 11, "C": 11}
    assert target["peer_view_counts"] == {"B": 2, "C": 2}
    assert object_record(result, "B")["bank_token_count"] == 11
    assert object_record(result, "A")["view_mean_control"]["mean"] == pytest.approx(1.0)


def test_nonwinners_and_tied_objects_are_all_retained():
    rows = [observation("V00", f"c{i}", f"O{i}", [[1, 0]]) for i in range(5)]
    result = score_scene(rows, selected_views=["V00"])
    assert len(result["observations"]) == len(result["objects"]) == 5
    assert [item["object_bank"]["mean"] for item in result["objects"]] == [0.0] * 5


def test_raw_mean_and_top2_fuse_available_observations_once():
    rows = []
    for view, direction in zip(["V00", "V01", "V02"], [[1, 0], [0, 1], [-1, 0]]):
        rows.extend([
            observation(view, "a", "A", [direction]),
            observation(view, "b", "B", [[1, 0]]),
            observation(view, "c", "C", [[1, 0]]),
        ])
    result = score_scene(rows, selected_views=["V00", "V01", "V02", "V03"])
    aggregate = object_record(result, "A")["object_bank"]
    assert aggregate["mean"] == pytest.approx(1.0)
    assert aggregate["top2"] == pytest.approx(1.5)
    assert aggregate["valid_observations"] == 3
    assert aggregate["aggregation"] == "raw_observation_scores"
    assert result["missing_views"] == ["V03"]


def test_unmatched_objects_retained_but_not_used_for_bank_self_exclusion():
    rows = three_objects()
    before = score_scene(rows, selected_views=["V00", "V01"])
    rows.extend([
        observation("V01", "u1", None, [[1, 0]]),
        observation("V01", "u2", None, [[0, 1]]),
        observation("V01", "u3", None, [[-1, 0]]),
    ])
    result = score_scene(rows, selected_views=["V00", "V01"])
    assert record(before, "V00", "c1")["object_bank"] == record(result, "V00", "c1")["object_bank"]
    assert result["counts"]["unmatched_observations"] == 3
    unmatched = [item for item in result["objects"] if item["identity_status"] == "unmatched"]
    assert len(unmatched) == 3
    for item in unmatched:
        assert item["object_bank"]["mean"] is None
        assert item["view_mean_control"]["mean"] is None
        assert item["within_view"]["mean"] is not None
        assert len(item["members"]) == 1
    for candidate in ["u1", "u2", "u3"]:
        score = record(result, "V01", candidate)["object_bank"]
        assert score["status"] == "unavailable_identity"
        assert "exclusion" in score["reason"]
        assert record(result, "V01", candidate)["peer_token_counts"] == {}


def test_within_view_scores_do_not_depend_on_object_ids():
    assigned = three_objects()
    unknown = deepcopy(assigned)
    for item in unknown:
        item["object_id"] = None
    first = score_scene(assigned, selected_views=["V00"])
    second = score_scene(unknown, selected_views=["V00"])
    assert [item["within_view"] for item in first["observations"]] == [item["within_view"] for item in second["observations"]]


def test_empty_scene_and_missing_views_are_reported_without_nan():
    result = score_scene([], selected_views=["V00", "V01"])
    assert result["observations"] == result["objects"] == []
    assert result["missing_views"] == ["V00", "V01"]
    assert result["counts"]["observations"] == 0
    json.dumps(result, allow_nan=False)


def test_empty_foreground_is_retained_and_not_a_usable_reference():
    rows = three_objects()
    rows.append(observation("V00", "empty", "D", np.empty((0, 2))))
    result = score_scene(rows, selected_views=["V00", "V01"])
    empty = record(result, "V00", "empty")
    assert empty["within_view"]["status"] == "unavailable_patches"
    assert empty["object_bank"]["status"] == "unavailable_patches"
    assert empty["view_mean_control"]["status"] == "unavailable_patches"
    assert object_record(result, "D")["object_bank"]["mean"] is None
    assert result["counts"]["usable_reference_banks"] == 3
    assert result["counts"]["unavailable_patch_observations"] == 1
    json.dumps(result, allow_nan=False)


def test_empty_peer_banks_cannot_satisfy_two_peer_requirement():
    rows = [
        observation("V00", "a", "A", [[1, 0]]),
        observation("V00", "b", "B", [[0, 1]]),
        observation("V00", "c", "C", np.empty((0, 2))),
    ]
    result = score_scene(rows, selected_views=["V00"])
    target = record(result, "V00", "a")
    for method in ["within_view", "object_bank"]:
        assert target[method]["score"] is None
        assert target[method]["status"] == "insufficient_peers"
        assert target[method]["peer_count"] == 1


def test_input_and_selected_view_order_do_not_change_results():
    rows = three_objects()
    rows.append(observation("V01", "a", "A", [[1, 1]]))
    expected = score_scene(rows, selected_views=["V00", "V01"])
    for permutation in itertools.permutations(rows):
        assert score_scene(permutation, selected_views=["V01", "V00"]) == expected


def test_candidate_names_do_not_change_tied_numeric_scores():
    rows = [observation("V00", f"c{i}", f"O{i}", [[1, 0]]) for i in range(4)]
    before = score_scene(rows, selected_views=["V00"])
    for i, item in enumerate(rows):
        item["candidate_id"] = f"z{3-i}"
    after = score_scene(rows, selected_views=["V00"])
    assert {item["object_id"]: item["object_bank"] for item in before["objects"]} == {item["object_id"]: item["object_bank"] for item in after["objects"]}


@pytest.mark.parametrize("extra", [
    observation("V00", "c1", "D", [[1, 0]]),
    observation("V00", "different_candidate", "A", [[1, 0]]),
])
def test_duplicate_candidate_or_identity_in_one_view_is_invalid(extra):
    with pytest.raises(ValueError, match="same view"):
        score_scene(three_objects() + [extra], selected_views=["V00"])


def test_other_selected_view_can_reuse_candidate_id():
    result = score_scene(three_objects() + [observation("V01", "c1", "A", [[1, 0]])], selected_views=["V00", "V01"])
    assert len(object_record(result, "A")["members"]) == 2


def test_outside_selected_views_is_rejected_instead_of_silently_used_or_filtered():
    with pytest.raises(ValueError, match="selected-view scope"):
        score_scene(three_objects() + [observation("V19", "a", "A", [[1, 0]])], selected_views=["V00"])


@pytest.mark.parametrize("views", [[], "V00", ["V00", "V00"], ["V00", "bad/view"]])
def test_invalid_selected_view_contract(views):
    with pytest.raises(ValueError):
        score_scene([], selected_views=views)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan"), float("inf"), True, "0.02"])
def test_invalid_tail_fraction(fraction):
    with pytest.raises(ValueError, match="top_fraction"):
        score_scene([], selected_views=["V00"], top_fraction=fraction)


@pytest.mark.parametrize("patches", [[[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [1, 2], np.empty((2, 0))])
def test_invalid_patch_features_stop_scoring(patches):
    with pytest.raises(ValueError, match="patch"):
        score_scene([observation("V00", "a", "A", patches)], selected_views=["V00"])


def test_mixed_dimensions_are_rejected_even_for_empty_patch_matrix():
    rows = three_objects() + [observation("V01", "a", "A", np.empty((0, 3)))]
    with pytest.raises(ValueError, match="same feature dimension"):
        score_scene(rows, selected_views=["V00", "V01"])


def test_unknown_evaluator_field_cannot_enter_model_facing_contract():
    rows = three_objects()
    rows[0]["anomaly_label"] = 1
    with pytest.raises(ValueError, match="Observation fields"):
        score_scene(rows, selected_views=["V00"])
