from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from visual_peer_inspector.workflows.comparison_io import canonical, seal
from visual_peer_inspector.workflows.learned_peer_features import (
    METHODS,
    VIEW_SETS,
    prepare_feature_package,
    prepare_scene_features,
)


def candidate(name: str, offset: float) -> dict:
    embedding = np.zeros(8, dtype=float)
    embedding[int(offset) % 8] = 1.0
    return {
        "candidate_id": name,
        "embedding": embedding.tolist(),
        "geometry": [offset + value / 10 for value in range(7)],
        "scores": {method: offset + index / 100 for index, method in enumerate(METHODS)},
        "heatmaps": {
            "P1_current": [offset + value / 100 for value in range(16)],
            "forward_02_cov70": [offset + value / 90 for value in range(16)],
            "forward_10_cov0": [offset + value / 80 for value in range(16)],
        },
    }


def scene_records(scene: str = "S000001") -> list[dict]:
    rows = []
    for route in ("automatic", "supplied"):
        for index in range(20):
            rows.append(
                {
                    "scene": scene,
                    "route": route,
                    "view": f"V{index:02d}",
                    "feature_cache_hit": True,
                    "patch_fallbacks": {},
                    "seconds": 0.1,
                    "status": "scored",
                    "objects": [candidate("C00", 0.0), candidate("C01", 1.0), candidate("C02", 2.0)],
                }
            )
    return rows


def test_scene_feature_schema_is_finite_and_label_free():
    payload, names = prepare_scene_features(scene_records())
    assert payload["scene"] == "S000001"
    assert len(payload["groups"]) == len(VIEW_SETS) * 2
    assert len(names) > 40
    text = json.dumps(payload).lower()
    assert "label" not in text
    assert "category" not in text
    for setting in payload["groups"]:
        for group in setting["view_groups"]:
            matrix = np.asarray(group["features"])
            assert matrix.shape == (3, len(names))
            assert np.isfinite(matrix).all()
        for tracks in setting["winner_tracks"].values():
            for track in tracks:
                member_views = [member["view_id"] for member in track["members"]]
                assert len(member_views) == len(set(member_views))
                assert np.isfinite(track["cluster_score"])
                assert 0.0 < track["vote_fraction"] <= 1.0


def test_missing_patch_heatmap_is_encoded_as_missing_evidence():
    records = scene_records()
    records[0]["objects"][0]["heatmaps"]["forward_02_cov70"] = None
    payload, names = prepare_scene_features(records)
    feature_index = names.index("heatmap_forward_02_cov70_available")
    setting = next(
        item for item in payload["groups"]
        if item["route"] == "automatic" and item["view_set_id"] == "one-primary"
    )
    assert setting["view_groups"][0]["features"][0][feature_index] == 0.0


def test_missing_upstream_scores_keep_the_candidate_and_mark_availability():
    records = scene_records()
    for item in records[0]["objects"]:
        item["scores"]["forward_02_cov70"] = None
    payload, names = prepare_scene_features(records)
    availability = names.index("available_forward_02_cov70")
    setting = next(
        item for item in payload["groups"]
        if item["route"] == "automatic" and item["view_set_id"] == "one-primary"
    )
    assert len(setting["view_groups"][0]["candidate_ids"]) == 3
    assert all(row[availability] == 0.0 for row in setting["view_groups"][0]["features"])
    assert setting["view_groups"][0]["method_ranks"]["forward_02_cov70"] == [None, None, None]


def test_single_candidate_view_is_preserved_as_insufficient_peer_evidence():
    records = scene_records()
    records[0]["objects"] = records[0]["objects"][:1]
    payload, names = prepare_scene_features(records)
    count_index = names.index("candidate_count")
    setting = next(
        item for item in payload["groups"]
        if item["route"] == "automatic" and item["view_set_id"] == "one-primary"
    )
    assert setting["view_groups"][0]["features"][0][count_index] == 1.0


def test_empty_view_is_preserved_as_detector_failure():
    records = scene_records()
    records[0]["objects"] = []
    payload, names = prepare_scene_features(records)
    setting = next(
        item for item in payload["groups"]
        if item["route"] == "automatic" and item["view_set_id"] == "one-primary"
    )
    assert setting["view_groups"][0]["candidate_ids"] == []
    assert np.asarray(setting["view_groups"][0]["features"]).shape == (0,)
    assert len(names) > 50


def test_feature_preparation_resumes_verified_scene(tmp_path: Path):
    scoring = tmp_path / "scoring"
    scenes = scoring / "scenes"
    scenes.mkdir(parents=True)
    source = scenes / "S000001.json"
    source.write_bytes(canonical(scene_records()))
    seal(scoring, [source], name="prediction_lock.json")
    output = tmp_path / "output"
    first = prepare_feature_package(scoring, output)
    second = prepare_feature_package(scoring, output)
    assert first["scene_count"] == 1
    assert second["reused_scenes"] == 1
    assert first["feature_lock_sha256"] == second["feature_lock_sha256"]


def test_corrupt_feature_checkpoint_is_recomputed(tmp_path: Path):
    scoring = tmp_path / "scoring"
    scenes = scoring / "scenes"
    scenes.mkdir(parents=True)
    source = scenes / "S000001.json"
    source.write_bytes(canonical(scene_records()))
    seal(scoring, [source], name="prediction_lock.json")
    output = tmp_path / "output"
    prepare_feature_package(scoring, output)
    checkpoint = output / "scenes/S000001.json"
    checkpoint.write_text("{}")
    result = prepare_feature_package(scoring, output)
    assert result["reused_scenes"] == 0
    assert json.loads(checkpoint.read_text())["scene"] == "S000001"
