"""Exact-assignment tests use synthetic features, never dataset identities."""

from copy import deepcopy
from itertools import combinations
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from vpi_object_mapping import matching


def observation(view, candidate, angle=0.0):
    vector = np.zeros(384)
    vector[:2] = [math.cos(angle), math.sin(angle)]
    return {"view_id": view, "candidate_id": candidate, "embedding": vector}


def signature(hypothesis):
    return tuple(sorted(tuple(sorted((m["view_id"], m["candidate_id"]) for m in t["members"])) for t in hypothesis["tracks"]))


def assert_complete(hypothesis, observations):
    members = [(m["view_id"], m["candidate_id"]) for t in hypothesis["tracks"] for m in t["members"]]
    assert sorted(members) == sorted((o["view_id"], o["candidate_id"]) for o in observations)
    assert len({t["identity_id"] for t in hypothesis["tracks"]}) == len(hypothesis["tracks"])
    for track in hypothesis["tracks"]:
        assert len({m["view_id"] for m in track["members"]}) == len(track["members"])


def brute_force_objectives(rows):
    """Independent restricted-growth enumeration of all physical-object partitions."""
    vectors = [r["embedding"] / np.linalg.norm(r["embedding"]) for r in rows]
    answers = []

    def visit(index, groups):
        if index == len(rows):
            total = sum(.35 - float(np.clip(1-vectors[a]@vectors[b], 0, 2)) for group in groups for a, b in combinations(group, 2))
            answers.append(total)
            return
        for group in groups:
            if all(rows[j]["view_id"] != rows[index]["view_id"] and 1-vectors[j]@vectors[index] <= .35 for j in group):
                group.append(index)
                visit(index + 1, groups)
                group.pop()
        groups.append([index])
        visit(index + 1, groups)
        groups.pop()

    visit(0, [])
    return sorted(answers, reverse=True)


def test_joint_top_three_matches_independent_exhaustive_partition_oracle():
    rows = [observation(v, c, a) for v, c, a in [
        ("V0", "a", 0.), ("V0", "b", .6),
        ("V1", "c", .25), ("V1", "d", .8),
        ("V2", "e", .4), ("V2", "f", 1.0)]]
    result = matching.match_scene(rows, selected_views=["V0", "V1", "V2"])
    assert result["status"] == "optimal"
    assert [h["objective"] for h in result["hypotheses"]] == pytest.approx(brute_force_objectives(rows)[:3])
    assert len({signature(h) for h in result["hypotheses"]}) == 3
    for hypothesis in result["hypotheses"]:
        assert_complete(hypothesis, rows)
        assert hypothesis["optimal"]
        assert hypothesis["objective_gap_from_best"] == pytest.approx(result["primary"]["objective"] - hypothesis["objective"])


def test_pairwise_chain_cannot_become_incompatible_three_view_track():
    rows = [observation("V0", "a", 0), observation("V1", "b", .6), observation("V2", "c", 1.2)]
    result = matching.match_scene(rows, selected_views=["V0", "V1", "V2"])
    assert result["status"] == "optimal"
    assert max(len(t["members"]) for h in result["hypotheses"] for t in h["tracks"]) == 2
    assert result["primary"]["objective"] == pytest.approx(.35 - (1-math.cos(.6)))


def test_all_candidates_retained_even_when_unmatched_or_same_view_duplicates_look_identical():
    rows = [observation("V0", "a"), observation("V0", "b"), observation("V1", "c"), observation("V1", "far", math.pi)]
    result = matching.match_scene(rows, selected_views=["V0", "V1", "empty"])
    assert result["missing_views"] == ["empty"]
    for hypothesis in result["hypotheses"]:
        assert_complete(hypothesis, rows)
        far = next(t for t in hypothesis["tracks"] if any(m["candidate_id"] == "far" for m in t["members"]))
        assert len(far["members"]) == 1 and far["status"] == "unmatched"


def test_identical_appearance_keeps_multiple_equal_object_maps_without_probabilities():
    rows = [observation(v, c) for v in ("V0", "V1", "V2") for c in ("a", "b")]
    result = matching.match_scene(rows, selected_views=["V0", "V1", "V2"])
    assert len(result["hypotheses"]) == 3
    assert all(h["objective_gap_from_best"] == pytest.approx(0) for h in result["hypotheses"])
    assert "probability" not in json.dumps(result)


def test_candidate_and_view_order_permutation_preserves_all_map_choices():
    rows = [observation(v, c) for v in ("V0", "V1", "V2") for c in ("a", "b")]
    first = matching.match_scene(rows, selected_views=["V0", "V1", "V2"])
    second = matching.match_scene(rows[::-1], selected_views=["V2", "V0", "V1"])
    assert first["hypotheses"] == second["hypotheses"]
    assert first["primary"] == second["primary"]


@pytest.mark.parametrize("rows,views", [([], ["V0", "V1"]), ([observation("V0", "a")], ["V0", "V1"]),
                                      ([observation("V0", "a"), observation("V0", "b")], ["V0"])])
def test_empty_or_single_view_has_only_complete_singleton_partition(rows, views):
    result = matching.match_scene(rows, selected_views=views)
    assert result["status"] == "optimal" and len(result["hypotheses"]) == 1
    assert result["alternative_search_status"] == "exhausted"
    assert result["primary"]["objective"] == 0
    assert_complete(result["primary"], rows)


def test_two_view_margin_objective_matches_optional_bipartite_assignment():
    rows = [observation(v, c, a) for v, c, a in [("V0", "a", 0), ("V0", "b", .5), ("V1", "a", .1), ("V1", "b", .8)]]
    result = matching.match_scene(rows, selected_views=["V0", "V1"])
    expected = sum(.35 - (1-math.cos(a)) for a in [.1, .3])
    assert result["primary"]["objective"] == pytest.approx(expected)


def test_enumeration_cap_never_silently_truncates_the_optimization():
    rows = [observation(v, c) for v in ("V0", "V1", "V2") for c in ("a", "b")]
    result = matching.match_scene(rows, selected_views=["V0", "V1", "V2"], max_tracks=4)
    assert result["status"] == "fallback" and result["fallback_reason"] == "candidate_track_limit"
    assert result["hypotheses"] == []
    assert not result["primary"]["optimal"]
    assert all(len(t["members"]) == 1 for t in result["primary"]["tracks"])
    assert_complete(result["primary"], rows)


def test_observation_cap_falls_back_explicitly_without_losing_candidates():
    rows = [observation("V0", f"C{i}") for i in range(65)]
    result = matching.match_scene(rows, selected_views=["V0", "V1"])
    assert result["fallback_reason"] == "observation_limit"
    assert_complete(result["primary"], rows)


def test_nonoptimal_solver_incumbent_is_diagnostic_not_primary(monkeypatch):
    rows = [observation("V0", "a"), observation("V1", "b")]
    def timeout(**kwargs):
        assert kwargs["options"]["mip_rel_gap"] == 0.0
        assert 0 < kwargs["options"]["time_limit"] <= 5
        # Lexicographic candidates: singleton a, pair ab, singleton b.
        return SimpleNamespace(status=1, x=np.asarray([0., 1., 0.]), mip_gap=.2, mip_dual_bound=-.4)
    monkeypatch.setattr(matching, "milp", timeout)
    result = matching.match_scene(rows, selected_views=["V0", "V1"])
    assert result["status"] == "fallback" and result["fallback_reason"] == "solver_status_1"
    assert len(result["primary"]["tracks"]) == 2 and result["primary"]["optimal"] is False
    assert len(result["unproven_incumbent"]["tracks"]) == 1
    assert result["unproven_incumbent"]["optimal"] is False


def test_alternative_timeout_keeps_proven_primary(monkeypatch):
    real = matching.milp
    calls = []
    def timeout_second(**kwargs):
        calls.append(kwargs["options"]["time_limit"])
        return real(**kwargs) if len(calls) == 1 else SimpleNamespace(status=1, x=None)
    monkeypatch.setattr(matching, "milp", timeout_second)
    result = matching.match_scene([observation("V0", "a"), observation("V1", "b")], selected_views=["V0", "V1"])
    assert result["status"] == "optimal" and result["primary"]["optimal"]
    assert len(result["hypotheses"]) == 1
    assert result["alternative_search_status"] == "time_limit" and result["fallback_reason"] is None
    assert calls[1] < calls[0]


def test_deadline_in_enumeration_falls_back_without_solver(monkeypatch):
    ticks = iter([0., 10., 10.])
    monkeypatch.setattr(matching.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(matching, "milp", lambda **kwargs: pytest.fail("Solver started after exhausted deadline"))
    result = matching.match_scene([observation("V0", "a")], selected_views=["V0", "V1"], time_limit_seconds=1)
    assert result["fallback_reason"] == "enumeration_time_limit"


def test_malformed_solver_partition_is_not_accepted(monkeypatch):
    monkeypatch.setattr(matching, "milp", lambda **kwargs: SimpleNamespace(status=0, x=np.asarray([1., 1., 0.])))
    with pytest.raises(ValueError, match="exactly once"):
        matching.match_scene([observation("V0", "a"), observation("V1", "b")], selected_views=["V0", "V1"])


@pytest.mark.parametrize("change", ["label", "reference", "score", "geometry", "wrong_dimension", "zero", "nan", "duplicate", "unselected"])
def test_invalid_or_evaluator_facing_inputs_fail_closed(change):
    rows = [observation("V0", "a"), observation("V1", "b")]
    if change in ("label", "reference", "score", "geometry"):
        rows[0][change] = 1
    elif change == "wrong_dimension":
        rows[0]["embedding"] = np.ones(383)
    elif change == "zero":
        rows[0]["embedding"] *= 0
    elif change == "nan":
        rows[0]["embedding"][0] = np.nan
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["view_id"] = "V19"
    with pytest.raises(ValueError):
        matching.match_scene(rows, selected_views=["V0", "V1"])


@pytest.mark.parametrize("options", [{"selected_views": []}, {"selected_views": ["V0", "V0"]},
                                     {"selected_views": [f"V{i}" for i in range(6)]},
                                     {"time_limit_seconds": 0}, {"time_limit_seconds": float("nan")},
                                     {"max_hypotheses": 4}, {"max_hypotheses": True}, {"max_tracks": 50001}])
def test_invalid_configurations_are_errors_not_fallbacks(options):
    selected_views = options.get("selected_views", ["V0", "V1"])
    others = {key: value for key, value in options.items() if key != "selected_views"}
    with pytest.raises(ValueError):
        matching.match_scene([], selected_views=selected_views, **others)
