"""Frozen scoring and uncertainty contracts, plus an optional label-free pilot check."""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vpi_object_mapping import prediction
from vpi_object_mapping.evaluation import validate_predictions
from vpi_object_bank.storage import digest, read_json, verified_unit


def observations():
    return [{"view_id": view, "candidate_id": f"C{i}", "object_id": None,
             "patches": np.asarray([vector], dtype=float)}
            for view in ("V0", "V1")
            for i, vector in enumerate(([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]))]


def tracks():
    return [{"identity_id": f"T{i}", "status": "linked",
             "members": [{"view_id": view, "candidate_id": f"C{i}"} for view in ("V0", "V1")]}
            for i in range(3)]


def test_rank_map_conserves_candidates_and_applies_frozen_tie_aware_top_two():
    rows = observations()
    result, ranking, ranks = prediction.rank_map(rows, tracks(), ["V0", "V1"])
    assert len(result["observations"]) == 6
    assert len(ranking) == 3
    assert {r["score"] for r in ranking} == {.5}
    assert {r["support"] for r in ranking} == {2}
    assert {r["fusion_observations"] for r in ranking} == {2}
    assert {r["mapping_status"] for r in ranking} == {"linked"}
    assert set(ranks.values()) == {.5}
    assert prediction.selected_members(ranking) is None
    assert all(row["object_id"] is None for row in rows)


def test_unmatched_singletons_remain_hypotheses_and_unavailable_patches_remain_null():
    rows = observations()
    rows[-1]["patches"] = np.empty((0, 3))
    singles = [{"identity_id": f"T{i}", "status": "unmatched",
                "members": [{"view_id": row["view_id"], "candidate_id": row["candidate_id"]}]}
               for i, row in enumerate(rows)]
    result, ranking, _ = prediction.rank_map(rows, singles, ["V0", "V1"])
    assert len(ranking) == len(rows) == 6
    assert {obj["mapping_status"] for obj in ranking} == {"unmatched"}
    last = next(row for row in ranking if row["id"] == "T5")
    assert last["score"] is None and last["support"] == 0
    assert result["observations"][-1]["object_bank"]["score"] is None


@pytest.mark.parametrize("failure", ["missing", "duplicate_member", "duplicate_track_id", "wrong_key", "same_view"])
def test_malformed_maps_fail_before_silently_reassigning_a_candidate(failure):
    mapped = tracks()
    if failure == "missing":
        mapped[0]["members"].pop()
    elif failure == "duplicate_member":
        mapped.append({"identity_id": "duplicate", "status": "unmatched", "members": [deepcopy(mapped[0]["members"][0])]})
    elif failure == "duplicate_track_id":
        mapped[1]["identity_id"] = mapped[0]["identity_id"]
    elif failure == "wrong_key":
        mapped[0]["members"][0]["candidate_id"] = "outside"
    else:
        # Cover every input exactly once but put two candidates from V0 together.
        mapped[0]["members"][1], mapped[1]["members"][0] = mapped[1]["members"][0], mapped[0]["members"][1]
    with pytest.raises(ValueError):
        prediction.rank_map(observations(), mapped, ["V0", "V1"])


def ranking(selected, *, score=1.0, identity="arbitrary"):
    return [{"id": identity, "score": score,
             "members": [{"view_id": v, "candidate_id": c} for v, c in selected]}]


def test_uncertainty_compares_physical_members_not_arbitrary_map_ids():
    primary = ranking([("V0", "C0"), ("V1", "C0")])
    unchanged = ranking([("V1", "C0"), ("V0", "C0")], identity="renamed")
    changed = ranking([("V0", "C0"), ("V1", "C1")])
    report = prediction.uncertainty(primary, [unchanged, changed])
    assert report["selection_overlap"] == pytest.approx([1., 1 / 3])
    assert report["all_available_maps_same_selected_members"] is False
    assert report["alternative_count"] == 2
    assert "not a calibrated confidence" in report["interpretation"]


def test_tied_unavailable_or_absent_alternatives_do_not_imply_certainty():
    primary = ranking([("V0", "C0")])
    tied = primary + ranking([("V0", "C1")], identity="second")
    assert prediction.selected_members(tied) is None
    assert prediction.selected_members(ranking([("V0", "C0")], score=None)) is None
    assert prediction.uncertainty(primary, [tied])["selection_overlap"] == [None]
    assert prediction.uncertainty(primary, [tied])["all_available_maps_same_selected_members"] is False
    assert prediction.uncertainty(primary, [])["all_available_maps_same_selected_members"] is None
    assert prediction.uncertainty(tied, [primary])["status"] == "no_unique_primary_selection"


def test_actual_first_pilot_conservation_and_frozen_baselines_without_truth():
    configuration = os.environ.get("VPI_OBJECT_MAPPING_PILOT_CONFIG")
    if configuration is None:
        pytest.skip("Set VPI_OBJECT_MAPPING_PILOT_CONFIG for the local sealed pilot check")
    cfg = read_json(Path(configuration))
    manifest = read_json(Path(cfg["prepared_root"]) / "input_manifest.json")
    scene = manifest["scenes"][0]
    unit = Path(cfg["prediction_root"]) / "scenes" / scene
    binding = read_json(Path(cfg["prediction_root"]) / "run_binding.json")
    assert verified_unit(unit, digest(binding))
    payload = read_json(unit / "predictions.json")
    previous = read_json(Path(cfg["previous_prediction_root"]) / "scenes" / scene / "predictions.json")
    scoring = read_json(Path(cfg["scoring_root"]) / "scenes" / f"{scene}.json")
    # This validator checks predictions and input candidates; no truth file is read.
    validate_predictions(payload, scene, previous, scoring)
    for setting in payload["settings"]:
        expected = {(o["view_id"], o["candidate_id"]) for o in setting["observations"]}
        for hypothesis in [setting["mapping"]["primary"], *setting["mapping"]["hypotheses"]]:
            members = [(m["view_id"], m["candidate_id"]) for t in hypothesis["tracks"] for m in t["members"]]
            assert len(members) == len(set(members)) and set(members) == expected
            assert all(len(t["members"]) == len({m["view_id"] for m in t["members"]}) for t in hypothesis["tracks"])
        assert len(setting["mapping"]["hypotheses"]) <= 3
        assert all(row["view_id"] in setting["views"] for row in setting["observations"])
    json.dumps(payload, allow_nan=False)
