from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.frozen_resource import publish_frozen_resource
from visual_peer_inspector.workflows.one_shot_evaluation import (
    EXPECTED_COHORTS,
    CohortExpectation,
    VerifiedBoundary,
    _digest,
    _normalise_official_evaluator,
    _verify_evaluator_manifest,
    attrition_table,
    build_scene_metrics,
    create_evaluation_contract,
    decision_confusion_table,
    error_taxonomy,
    load_evaluation_contract,
    reliability_table,
    run_one_shot_evaluation,
    subgroup_table,
    summary_metrics,
)


def _expectation(scene_count: int = 2) -> CohortExpectation:
    return CohortExpectation(
        cohort_id="fixture-cohort",
        role="internal_lock",
        split_name="internal-lock",
        membership_sha256="a" * 64,
        source_membership_sha256="a" * 64,
        scene_count=scene_count,
        image_count=scene_count * 20,
        asset_group_count=scene_count,
        prediction_lock_sha256="b" * 64,
        prediction_run_record_sha256="c" * 64,
        evaluator_manifest_sha256="d" * 64,
    )


def _track(identity: str, reference_object: str, score: float) -> tuple[dict, dict]:
    candidate = f"candidate-{identity}"
    track = {
        "identity_id": identity,
        "members": [{"view_id": "V00", "candidate_id": candidate}],
        "identity_status": "matched",
        "cycle_consistent": True,
        "fused_unusual_score": score,
    }
    crosswalk = {
        "schema_version": "visual-peer-evaluator-crosswalk/v1",
        "cohort_id": "fixture-cohort",
        "opaque_scene_id": "",
        "view_id": "V00",
        "prediction_object_id": candidate,
        "reference_object_id": reference_object,
    }
    return track, crosswalk


def _metric_fixture() -> tuple[VerifiedBoundary, list[dict], list[dict]]:
    decisions = []
    labels = []
    crosswalk = []
    for scene_number in range(2):
        scene_id = f"scene-{scene_number}"
        labels.append(
            {
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": f"asset-{scene_number}",
                "anomalous_reference_object_ids": ["odd"],
                "category": "toy",
                "anomaly_type": "colour",
                "object_count_band": "3",
            }
        )
        odd_track, odd_crosswalk = _track(
            "track-odd", "odd", 0.9 if scene_number == 0 else 0.1
        )
        normal_track, normal_crosswalk = _track(
            "track-normal", "normal", 0.1 if scene_number == 0 else 0.9
        )
        for item in (odd_crosswalk, normal_crosswalk):
            item["opaque_scene_id"] = scene_id
            crosswalk.append(item)
        for view_count in (2, 3, 5):
            decisions.append(
                {
                    "opaque_scene_id": scene_id,
                    "view_count": view_count,
                    "tracks": [odd_track, normal_track],
                    "review_signal": "automatic" if scene_number == 0 else "review",
                    "margin": 0.8,
                    "agreement": 1.0,
                    "view_failures": [],
                }
            )
    boundary = VerifiedBoundary(
        expectation=_expectation(),
        prediction_root=Path("."),
        prediction_lock={},
        prediction_rows=(),
        multiview_rows=tuple(decisions),
        evaluator_root=Path("."),
        evaluator_manifest={},
    )
    return boundary, labels, crosswalk


def test_contract_self_hash_and_exact_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        EXPECTED_COHORTS,
        "09d",
        {
            "fixture-cohort": {
                "role": "internal_lock",
                "split_name": "internal-lock",
                "membership_sha256": "a" * 64,
                "scene_count": 2,
                "image_count": 40,
                "asset_group_count": 2,
            }
        },
    )
    expectation = asdict(_expectation())
    record = {
        "schema_version": "visual-peer-one-shot-evaluation-contract/v1",
        "status": "locked",
        "stage": "09d",
        "method_record_sha256": "e" * 64,
        "policy_record_sha256": "f" * 64,
        "evaluation_rules": {
            "prediction_reference_mapping": "evaluator_supplied_locked_crosswalk"
        },
        "cohorts": [expectation],
    }
    record["contract_sha256"] = _digest(record)
    path = tmp_path / "contract.json"
    path.write_bytes(json.dumps(record, sort_keys=True).encode())
    contract = load_evaluation_contract(path, expected_stage="09d")
    assert contract.cohorts[0].scene_count == 2
    record["cohorts"][0]["scene_count"] = 3
    path.write_bytes(canonical_json_bytes(record))
    with pytest.raises(ValueError, match="contract_sha256"):
        load_evaluation_contract(path, expected_stage="09d")


def test_official_contract_requires_internal_lock_and_inference_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        EXPECTED_COHORTS,
        "10b",
        {
            "fixture-cohort": {
                "role": "official_test",
                "split_name": "seen-test",
                "membership_sha256": "a" * 64,
                "scene_count": 2,
                "image_count": 40,
                "asset_group_count": 2,
            }
        },
    )
    record = {
        "schema_version": "visual-peer-one-shot-evaluation-contract/v1",
        "status": "locked",
        "stage": "10b",
        "method_record_sha256": "e" * 64,
        "policy_record_sha256": "f" * 64,
        "evaluation_rules": {
            "prediction_reference_mapping": "maximum_total_iou_one_to_one",
            "minimum_box_iou": 0.5,
        },
        "cohorts": [asdict(_expectation())],
    }
    record["contract_sha256"] = _digest(record)
    path = tmp_path / "contract.json"
    path.write_bytes(canonical_json_bytes(record))
    with pytest.raises(ValueError, match="09D and 10A receipt hashes"):
        load_evaluation_contract(path, expected_stage="10b")


def test_contract_creation_pins_verified_non_label_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        EXPECTED_COHORTS,
        "09d",
        {
            "fixture-cohort": {
                "role": "internal_lock",
                "split_name": "internal-lock",
                "membership_sha256": "a" * 64,
                "scene_count": 1,
                "image_count": 1,
                "asset_group_count": 1,
            }
        },
    )

    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value) + b"\n")

    method = {
        "schema_version": "visual-peer-08f-workflow/v1",
        "status": "frozen",
        "chosen_configuration": {"variant": "embedding_only"},
    }
    method["chosen_configuration_sha256"] = _digest(method["chosen_configuration"])
    source_root = tmp_path / "resource-sources"
    source_root.mkdir()
    method_source = source_root / "method_freeze.json"
    application_source = source_root / "application_output_contract.json"
    write_json(method_source, method)
    write_json(application_source, {"schema_version": "visual-peer-application-output/v1"})
    method_root = tmp_path / "method-resource"
    publish_frozen_resource(
        resource_root=method_root,
        stage="cross_view_identity_method_release",
        files={
            "application_output_contract.json": application_source,
            "method_freeze.json": method_source,
        },
        bindings={},
    )
    method_path = method_root / "method_freeze.json"
    policy = {
        "schema_version": "visual-peer-09b-review-policy/v1",
        "status": "frozen",
        "method_record_sha256": file_sha256(method_path),
        "selection": {"universal_accuracy_gate": None},
    }
    policy["policy_record_sha256"] = _digest(policy)
    policy_source = source_root / "review_policy_freeze.json"
    write_json(policy_source, policy)
    policy_root = tmp_path / "policy-resource"
    publish_frozen_resource(
        resource_root=policy_root,
        stage="calibration_review_policy_release",
        files={"review_policy_freeze.json": policy_source},
        bindings={"method_record_sha256": file_sha256(method_path)},
    )
    policy_path = policy_root / "review_policy_freeze.json"
    for path in source_root.iterdir():
        path.unlink()
    source_root.rmdir()

    model_safe = tmp_path / "prediction" / "model-safe"
    model_safe.mkdir(parents=True)
    for name in (
        "prediction_manifest.jsonl",
        "global_embeddings.npy",
        "embedding_index.jsonl",
        "multiview_prediction_manifest.jsonl",
    ):
        (model_safe / name).write_bytes(b"fixture\n")
    artifacts = [
        {
            "path": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(model_safe.iterdir())
    ]
    identity = {
        "split_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "locateanything_sha256": "c" * 64,
        "sam21_sha256": "d" * 64,
        "dinov2_sha256": "e" * 64,
        "preprocessing_sha256": "f" * 64,
        "schema_sha256": "1" * 64,
    }
    lock = {
        "schema_version": 1,
        "status": "locked",
        "cohort_id": "fixture-cohort",
        "cohort_membership_sha256": "a" * 64,
        "scene_count": 1,
        "image_count": 1,
        "identity": identity,
        "identity_sha256": _digest(identity),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_inventory_sha256": _digest(artifacts),
    }
    lock_path = model_safe / "prediction_lock.json"
    write_json(lock_path, lock)
    write_json(
        tmp_path / "prediction" / "safe" / "run_record.json",
        {
            "status": "completed",
            "cohort_id": "fixture-cohort",
            "prediction_only": True,
            "evaluator_opened": False,
            "accuracy_metrics_reported": False,
            "method_record_sha256": file_sha256(method_path),
            "policy_record_sha256": file_sha256(policy_path),
            "identity_sha256": lock["identity_sha256"],
            "prediction_bundle": {"prediction_lock_sha256": file_sha256(lock_path)},
        },
    )
    evaluator_root = tmp_path / "evaluator"
    evaluator_root.mkdir()
    evaluator_files = []
    for name in (
        "crosswalk_private.jsonl",
        "labels_private.jsonl",
        "geometry_private.jsonl",
    ):
        path = evaluator_root / name
        path.write_text("{}\n", encoding="utf-8")
        evaluator_files.append({"path": name, "sha256": file_sha256(path)})
    write_json(
        evaluator_root / "evaluator_manifest.json",
        {
            "schema_version": "visual-peer-evaluator-package/v1",
            "status": "locked",
            "cohort_id": "fixture-cohort",
            "cohort_membership_sha256": "a" * 64,
            "scene_count": 1,
            "image_count": 1,
            "asset_group_count": 1,
            "artifacts": evaluator_files,
            "artifact_inventory_sha256": _digest(evaluator_files),
        },
    )
    destination = tmp_path / "evaluation_contract_09d.json"
    record = create_evaluation_contract(
        stage="09d", mounted_roots=[tmp_path], output_path=destination
    )
    assert record["label_files_parsed_during_contract_creation"] is False
    assert load_evaluation_contract(destination, expected_stage="09d").cohorts[
        0
    ].prediction_lock_sha256 == file_sha256(lock_path)


def test_scene_metrics_retain_wrong_and_reviewed_results() -> None:
    boundary, labels, crosswalk = _metric_fixture()
    rows = build_scene_metrics(boundary, labels, crosswalk)
    assert len(rows) == 6
    first = next(
        row
        for row in rows
        if row["opaque_scene_id"] == "scene-0" and row["view_count"] == 5
    )
    wrong = next(
        row
        for row in rows
        if row["opaque_scene_id"] == "scene-1" and row["view_count"] == 5
    )
    assert (first["top1"], first["target_rank"], first["policy_state"]) == (
        1,
        1,
        "result_ready",
    )
    assert (wrong["top1"], wrong["top2"], wrong["target_rank"]) == (0, 1, 2)
    assert wrong["policy_state"] == "review_required"


def test_aggregate_tables_separate_selective_risk_and_review() -> None:
    boundary, labels, crosswalk = _metric_fixture()
    rows = build_scene_metrics(boundary, labels, crosswalk)
    summary = summary_metrics(rows, include_pooled=False)
    five = next(row for row in summary if row["view_count"] == 5)
    assert five["top1_accuracy"] == 0.5
    assert five["top2_accuracy"] == 1.0
    assert five["selective_accuracy"] == 1.0
    assert five["selective_risk"] == 0.0
    assert five["review_rate"] == 0.5
    assert reliability_table(rows)
    assert subgroup_table(rows)
    confusion = decision_confusion_table(rows)
    assert sum(row["scene_count"] for row in confusion if row["view_count"] == 5) == 2
    categories = {row["error_category"] for row in error_taxonomy(rows)}
    assert {"correct_automatic_result", "wrong_result_sent_to_review"}.issubset(
        categories
    )


def test_two_anomaly_scene_reports_complete_ranking_evidence() -> None:
    tracks_and_crosswalks = [
        _track("odd-one", "odd-one", 0.9),
        _track("normal", "normal", 0.8),
        _track("odd-two", "odd-two", 0.7),
    ]
    tracks = [item[0] for item in tracks_and_crosswalks]
    crosswalk = []
    for _track_row, crosswalk_row in tracks_and_crosswalks:
        crosswalk_row["opaque_scene_id"] = "two-anomaly-scene"
        crosswalk.append(crosswalk_row)
    decisions = tuple(
        {
            "opaque_scene_id": "two-anomaly-scene",
            "view_count": view_count,
            "tracks": tracks,
            "review_signal": "automatic",
            "margin": 0.1,
            "agreement": 1.0,
            "view_failures": [],
        }
        for view_count in (2, 3, 5)
    )
    boundary = VerifiedBoundary(
        expectation=_expectation(scene_count=1),
        prediction_root=Path("."),
        prediction_lock={},
        prediction_rows=(),
        multiview_rows=decisions,
        evaluator_root=Path("."),
        evaluator_manifest={},
    )
    labels = [
        {
            "opaque_scene_id": "two-anomaly-scene",
            "opaque_asset_group_id": "asset-two",
            "anomalous_reference_object_ids": ["odd-one", "odd-two"],
            "category": "toy",
            "anomaly_type": "mixed",
            "object_count_band": "3",
        }
    ]

    rows = build_scene_metrics(boundary, labels, crosswalk)
    five = next(row for row in rows if row["view_count"] == 5)
    assert five["top1"] == 1
    assert five["top2"] == 1
    assert five["retrieved_anomaly_count"] == 2
    assert five["recall_at_2"] == 0.5
    assert five["exact_set_at_anomaly_count"] == 0.0
    assert five["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    summary = summary_metrics(rows, include_pooled=False)
    summary_five = next(row for row in summary if row["view_count"] == 5)
    assert summary_five["mean_average_precision"] == pytest.approx(
        five["average_precision"]
    )
    assert summary_five["mean_recall_at_2"] == 0.5
    assert summary_five["exact_set_accuracy"] == 0.0
    subgroup = subgroup_table(rows)
    view_five = next(
        row
        for row in subgroup
        if row["subgroup_field"] == "view_count" and row["subgroup_value"] == "5"
    )
    assert view_five["mean_average_precision"] == pytest.approx(
        five["average_precision"]
    )


def test_official_adapter_matches_only_evaluator_supplied_anomaly_geometry() -> None:
    expectation = _expectation(scene_count=1)
    prediction = {
        "opaque_scene_id": "scene-0",
        "view_id": "V00",
        "objects": [
            {"object_id": "p0", "box_xyxy": [0, 0, 20, 20]},
            {"object_id": "p1", "box_xyxy": [40, 40, 60, 60]},
            {"object_id": "p2", "box_xyxy": [59, 59, 79, 79]},
        ],
    }
    boundary = VerifiedBoundary(
        expectation=expectation,
        prediction_root=Path("."),
        prediction_lock={},
        prediction_rows=(prediction,),
        multiview_rows=(),
        evaluator_root=Path("."),
        evaluator_manifest={},
    )
    raw_crosswalk = [
        {
            "opaque_scene_id": "scene-0",
            "base_asset_id": "asset-0",
            "category": "toy",
            "object_id_map": {"O00": 0, "O01": 1},
            "view_id_map": {"V00": "0000.png"},
        }
    ]
    raw_labels = [
        {
            "opaque_scene_id": "scene-0",
            "anomaly_type": "colour",
            "object_labels": [
                {"object_id": "O00", "is_anomaly": False},
                {"object_id": "O01", "is_anomaly": True},
            ],
            "anomaly_count": 1,
        }
    ]
    raw_geometry = [
        {
            "opaque_scene_id": "scene-0",
            "views": [
                {
                    "view_id": "V00",
                    "source_filename": "0000.png",
                    "boxes_xyxy": [[0, 0, 20, 20], [40, 40, 60, 60]],
                }
            ],
        }
    ]
    labels, crosswalk, geometry = _normalise_official_evaluator(
        boundary, raw_labels, raw_crosswalk, raw_geometry
    )
    assert labels[0]["anomalous_reference_object_ids"] == ["O01"]
    assert labels[0]["anomaly_type"] == "colour"
    assert {
        (row["prediction_object_id"], row["reference_object_id"]) for row in crosswalk
    } == {("p0", "O00"), ("p1", "O01")}
    assert all(row["prediction_object_id"] != "p2" for row in crosswalk)
    assert crosswalk[0]["correspondence_rule"].endswith("iou_0_50")
    assert len(geometry) == 2
    assert {row["reference_object_id"] for row in geometry} == {"O00", "O01"}


def test_official_evaluator_manifest_uses_release_file_inventory(
    tmp_path: Path,
) -> None:
    artifacts = []
    for name in (
        "crosswalk_private.jsonl",
        "labels_private.jsonl",
        "geometry_private.jsonl",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        artifacts.append(
            {
                "path": name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "visual-peer-official-evaluator-manifest/v1",
        "release_id": "fixture-release",
        "cohort_id": "fixture-cohort",
        "source_membership_sha256": "a" * 64,
        "scene_count": 1,
        "image_count": 20,
        "files": artifacts,
    }
    manifest_path = tmp_path / "evaluator_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    expectation = replace(
        _expectation(scene_count=1),
        evaluator_manifest_sha256=file_sha256(manifest_path),
    )
    assert _verify_evaluator_manifest(manifest_path, expectation) == manifest


def test_attrition_keeps_missed_reference_objects() -> None:
    expectation = _expectation(scene_count=1)
    prediction = {
        "opaque_scene_id": "scene-0",
        "view_id": "V00",
        "objects": [
            {
                "object_id": "p0",
                "box_xyxy": [0, 0, 20, 20],
                "foreground_pixels": 10,
                "mask_rle": {"size": [2, 2], "counts": [0, 1, 3]},
            }
        ],
    }
    decisions = tuple(
        {
            "opaque_scene_id": "scene-0",
            "view_count": count,
            "view_ids": ["V00"],
            "tracks": [
                {
                    "identity_id": "t0",
                    "members": [{"view_id": "V00", "candidate_id": "p0"}],
                    "fused_unusual_score": 1.0,
                }
            ],
        }
        for count in (2, 3, 5)
    )
    boundary = VerifiedBoundary(
        expectation=expectation,
        prediction_root=Path("."),
        prediction_lock={},
        prediction_rows=(prediction,),
        multiview_rows=decisions,
        evaluator_root=Path("."),
        evaluator_manifest={},
    )
    geometry = [
        {
            "opaque_scene_id": "scene-0",
            "view_id": "V00",
            "reference_object_id": "r0",
            "box_xyxy": [0, 0, 20, 20],
        },
        {
            "opaque_scene_id": "scene-0",
            "view_id": "V00",
            "reference_object_id": "r1",
            "box_xyxy": [40, 40, 60, 60],
        },
    ]
    crosswalk = [
        {
            "opaque_scene_id": "scene-0",
            "view_id": "V00",
            "prediction_object_id": "p0",
            "reference_object_id": "r0",
        }
    ]
    rows = attrition_table(boundary, geometry, crosswalk)
    assert all(row["reference_object_view_count"] == 2 for row in rows)
    assert all(row["proposal_retention"] == 0.5 for row in rows)


def test_reused_nonempty_output_is_rejected_before_any_input(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="reused nonempty"):
        run_one_shot_evaluation(
            contract_path=tmp_path / "missing.json",
            mounted_roots=[tmp_path],
            output_root=output,
            expected_stage="09d",
        )
