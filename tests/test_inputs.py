"""Synthetic contract checks; no private labels or dataset fixtures."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
from PIL import Image
import pytest

from visual_peer_inspector.models.sam2 import encode_binary_mask
from visual_peer_inspector.workflows.comparison_io import digest_file, read_json, seal, verify_seal
from visual_peer_inspector.workflows.scoring_comparison import METHODS, PROTOCOL
from vpi_object_bank.inputs import prepare_inputs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def candidate(scene, view, route, original):
    return "C" + sha256(f"{scene}/{view}/{route}/{original}".encode()).hexdigest()[:16]


@pytest.fixture
def inputs(tmp_path):
    model, scoring = tmp_path / "model", tmp_path / "scoring"
    scene = "S000001"
    views = ["V00", "V10", "V19"]
    models, scores, supplied, automatic = [], [], [], []
    for view in views:
        image_path = model / "media" / scene / f"{view}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(image_path)
        image_hash = digest_file(image_path)
        for route in ["supplied", "automatic"]:
            model_objects, scored_objects = [], []
            for index in range(2):
                old = f"O{index:02d}" if route == "supplied" else f"det-{index}"
                cid = candidate(scene, view, route, old)
                mask = np.zeros((8, 8), dtype=bool)
                mask[1:3, 1 + 3 * index:3 + 3 * index] = True
                model_objects.append({"candidate_id": cid, "box": [1 + 3 * index, 1, 3 + 3 * index, 3], "mask": encode_binary_mask(mask), "mask_sha256": sha256(mask.astype(np.uint8).tobytes()).hexdigest()})
                scored_objects.append({"candidate_id": cid, "embedding": [1.0] + [0.0] * 383, "geometry": [0.0] * 7, "heatmaps": {}, "scores": {name: float(index) for name in METHODS}})
                if route == "supplied":
                    supplied.append({"box_xyxy": model_objects[-1]["box"], "foreground_pixels": 4, "mask_path": f"opaque/{cid}.png", "mask_sha256": "a" * 64, "opaque_object_id": old, "opaque_scene_id": scene, "schema_version": 1, "view_id": view, "visible": True})
                else:
                    automatic.append({"object_id": old, "opaque_asset_group_id": "A000001", "opaque_scene_id": scene, "reference_identity_id": "O00" if index == 0 else None, "schema_version": "visual-peer-identity-evaluator/v1", "view_id": view})
            models.append({"scene": scene, "view": view, "route": route, "rgb": f"media/{scene}/{view}.png", "rgb_sha256": image_hash, "objects": model_objects})
            scores.append({"scene": scene, "view": view, "route": route, "objects": scored_objects, "feature_cache_hit": True, "patch_fallbacks": {}, "seconds": 0.1, "status": "scored"})
    write_json(model / "index.json", {"schema": 1, "study": PROTOCOL["study"], "scenes": [scene], "smoke_scenes": [scene], "views": views, "routes": ["supplied", "automatic"]})
    write_json(model / "scenes" / f"{scene}.json", models)
    seal(model, [p for p in model.rglob("*") if p.is_file()])
    write_json(scoring / "scenes" / f"{scene}.json", scores)
    write_json(scoring / "run_binding.json", {"input_lock_sha256": digest_file(model / "artifact_lock.json"), "producer": "dinov2-small", "protocol": deepcopy(PROTOCOL)})
    seal(scoring, [scoring / "scenes" / f"{scene}.json"], name="feature_lock.json")
    seal(scoring, [p for p in scoring.rglob("*") if p.is_file()], name="prediction_lock.json", metadata={"scenes": [scene]})
    supplied_path, automatic_path = tmp_path / "supplied.jsonl", tmp_path / "automatic.jsonl"
    write_lines(supplied_path, supplied)
    write_lines(automatic_path, automatic)
    return {"model_root": model, "scoring_root": scoring, "neutral_supplied_jsonl": supplied_path, "neutral_automatic_jsonl": automatic_path, "output_root": tmp_path / "prepared", "view_sets": {"one-primary": ["V00"], "two-primary": ["V00", "V10"]}}


def test_selected_only_export_preserves_unknown_and_cross_route_identity(inputs):
    manifest = prepare_inputs(**inputs)
    verify_seal(inputs["output_root"])
    payload = read_json(inputs["output_root"] / "scenes/S000001.json")
    rows = payload["observations"]
    assert payload["selected_views"] == ["V00", "V10"]
    assert len(rows) == 8 and sum(row["neutral_id"] is None for row in rows) == 2
    assert {row["view"] for row in rows} == {"V00", "V10"}
    assert all(set(row) == {"route", "view", "candidate_id", "neutral_id"} for row in rows)
    shared = {row["neutral_id"] for row in rows if row["route"] == "automatic" and row["neutral_id"] is not None}
    assert len(shared) == 1
    assert shared <= {row["neutral_id"] for row in rows if row["route"] == "supplied"}
    serialized = json.dumps(payload)
    assert "O00" not in serialized and "A000001" not in serialized and "det-" not in serialized
    assert manifest["anomaly_labels_opened"] is False
    assert not any(str(inputs["model_root"]) in json.dumps(value) for value in manifest.values())


def test_unexpected_label_field_is_rejected_before_output(inputs):
    path = inputs["neutral_automatic_jsonl"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["anomaly_label"] = 1
    write_lines(path, rows)
    with pytest.raises(ValueError, match="source schema"):
        prepare_inputs(**inputs)
    assert not inputs["output_root"].exists()


def test_tampered_scoring_file_fails_checksum(inputs):
    path = inputs["scoring_root"] / "scenes/S000001.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare_inputs(**inputs)


def test_valid_seal_with_wrong_model_binding_is_rejected(inputs):
    root = inputs["scoring_root"]
    binding = read_json(root / "run_binding.json")
    binding["input_lock_sha256"] = "0" * 64
    write_json(root / "run_binding.json", binding)
    lock = read_json(root / "prediction_lock.json")
    seal(root, [root / name for name in lock["files"]], name="prediction_lock.json", metadata=lock["metadata"])
    with pytest.raises(ValueError, match="different model input"):
        prepare_inputs(**inputs)


def test_missing_correspondence_is_not_silently_converted_to_unknown(inputs):
    path = inputs["neutral_automatic_jsonl"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_lines(path, rows[1:])
    with pytest.raises(ValueError, match="missing a trusted correspondence"):
        prepare_inputs(**inputs)


def test_two_candidates_cannot_share_one_known_identity_in_same_view(inputs):
    path = inputs["neutral_automatic_jsonl"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["reference_identity_id"] = "O00"
    write_lines(path, rows)
    with pytest.raises(ValueError, match="share an identity in the same view"):
        prepare_inputs(**inputs)


@pytest.mark.parametrize("schedule", [["V00", "V00"], ["V99"], ["V00", "V10", "V19", "V03"], "V00"])
def test_invalid_or_missing_selected_views_are_rejected(inputs, schedule):
    inputs["view_sets"] = {"invalid": schedule}
    with pytest.raises(ValueError):
        prepare_inputs(**inputs)


def test_correspondence_order_does_not_change_exported_observations(inputs, tmp_path):
    prepare_inputs(**inputs)
    first = (inputs["output_root"] / "scenes/S000001.json").read_bytes()
    for name in ["neutral_supplied_jsonl", "neutral_automatic_jsonl"]:
        path = inputs[name]
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    inputs["output_root"] = tmp_path / "permuted"
    prepare_inputs(**inputs)
    assert (inputs["output_root"] / "scenes/S000001.json").read_bytes() == first


def test_existing_output_is_not_overwritten(inputs):
    prepare_inputs(**inputs)
    original = (inputs["output_root"] / "artifact_lock.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_inputs(**inputs)
    assert (inputs["output_root"] / "artifact_lock.json").read_bytes() == original
