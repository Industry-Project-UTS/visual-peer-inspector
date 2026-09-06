from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from visual_peer_inspector.workflows import calibration_review_policy as calibration
from visual_peer_inspector.workflows.calibration_review_policy import (
    PREDICTION_SIGNAL_FIELDS,
    ReviewPolicy,
    VerifiedCalibrationBoundary,
    apply_review_policy,
    assign_group_folds,
    build_calibration_signal_record,
    choose_policy,
    cross_validated_calibration_curve,
    exact_resume_receipt_valid,
    freeze_review_policy,
    grouped_cross_validation,
    open_calibration_evaluator,
    policy_grid,
    primary_metric_table,
    select_evaluator_examples,
    wilson_interval,
    write_exact_resume_receipt,
)


def test_evaluator_fields_cannot_change_policy_signals_or_state() -> None:
    decision = {
        "opaque_scene_id": "S001",
        "view_count": 2,
        "view_ids": ["V00", "V10"],
        "tracks": [
            {
                "identity_status": "matched",
                "cycle_consistent": True,
            }
        ],
        "fused_odd_one_out": "I00",
        "margin": 0.30,
        "agreement": 1.0,
        "insufficient_evidence": False,
    }
    objects = [
        {
            "box_xyxy": [1, 1, 5, 5],
            "foreground_pixels": 16,
            "predicted_quality_score": 0.80,
        },
        {
            "box_xyxy": [7, 1, 11, 5],
            "foreground_pixels": 16,
            "predicted_quality_score": 0.85,
        },
    ]
    view_rows = [
        {"status": "ok", "objects": objects},
        {"status": "ok", "objects": objects},
    ]
    evaluator = {
        "reference_object_count": 3,
        "duplicate_fragment_warning": False,
        "acceptable_anomalous_identity_ids": ["I00"],
        "opaque_asset_group_id": "A001",
        "object_count_band": "3",
    }
    changed_evaluator = {
        **evaluator,
        "reference_object_count": 99,
        "duplicate_fragment_warning": True,
        "object_count_band": "5_plus",
    }
    first = build_calibration_signal_record(decision, view_rows, evaluator)
    second = build_calibration_signal_record(decision, view_rows, changed_evaluator)
    assert {field: first[field] for field in PREDICTION_SIGNAL_FIELDS} == {
        field: second[field] for field in PREDICTION_SIGNAL_FIELDS
    }
    policy = ReviewPolicy(0.10, 0.50, 0.25)
    assert apply_review_policy(first, policy) == apply_review_policy(second, policy)


def make_records() -> list[dict]:
    rows = []
    for scene_index in range(12):
        for view_count in (2, 3, 5):
            group = f"A{scene_index // 2:02d}"
            correct = not (scene_index in {2, 7} and view_count != 5)
            rows.append({
                "opaque_scene_id": f"S{scene_index:03d}",
                "view_count": view_count,
                "opaque_asset_group_id": group,
                "fused_odd_one_out": "I00",
                "result_available": True,
                "is_correct": correct,
                "margin": 0.03 if not correct else 0.30 + 0.01 * view_count,
                "agreement": 0.40 if not correct else 0.80,
                "minimum_mask_quality": 0.42 if scene_index == 4 else 0.88,
                "zero_proposal": False,
                "incomplete_proposals": scene_index == 5,
                "duplicate_or_fragmented_proposals": scene_index == 6,
                "empty_mask": False,
                "identity_unmatched_or_ambiguous": scene_index == 8,
                "stage_failure": False,
                "insufficient_evidence": False,
                "category": "toy-a" if scene_index < 6 else "toy-b",
                "anomaly_type": "missing" if scene_index % 2 else "colour",
                "object_count_band": "4",
                "acceptable_anomalous_identity_ids": ["I00" if correct else "I01"],
            })
    rows[-1]["fused_odd_one_out"] = None
    rows[-1]["is_correct"] = None
    rows[-1]["insufficient_evidence"] = True
    rows[-1]["stage_failure"] = True
    return rows


def test_three_state_policy_uses_all_required_candidate_signals() -> None:
    policy = ReviewPolicy(0.10, 0.60, 0.50)
    base = make_records()[0]
    assert apply_review_policy(base, policy) == ("result_ready", ())

    warning = {**base, "duplicate_or_fragmented_proposals": True}
    state, reasons = apply_review_policy(warning, policy)
    assert state == "review_required"
    assert reasons == ("duplicate_or_fragmented_proposals",)

    low_margin = {**base, "margin": 0.01, "agreement": 0.30, "minimum_mask_quality": 0.20}
    state, reasons = apply_review_policy(low_margin, policy)
    assert state == "review_required"
    assert {"small_p1_margin", "view_disagreement", "low_quality_mask"}.issubset(reasons)

    unavailable = {**base, "fused_odd_one_out": None, "insufficient_evidence": True}
    assert apply_review_policy(unavailable, policy)[0] == "insufficient_evidence"


def test_grouped_cross_validation_never_splits_base_assets() -> None:
    rows = make_records()
    assignments = assign_group_folds(rows, fold_count=5)
    assert set(assignments) == {row["opaque_asset_group_id"] for row in rows}
    fold_rows, evaluated = grouped_cross_validation(rows, fold_count=5)
    assert len(fold_rows) == 5
    assert len(evaluated) == len(rows)
    for group in assignments:
        assert {row["cv_fold"] for row in evaluated if row["opaque_asset_group_id"] == group} == {assignments[group]}


def test_policy_selection_and_calibration_report_real_held_out_values() -> None:
    rows = make_records()
    policy, frontier = choose_policy(rows, policy_grid(rows))
    assert isinstance(policy, ReviewPolicy)
    assert sum(bool(row["selected"]) for row in frontier) == 1
    assert any(bool(row["pareto_efficient"]) for row in frontier)

    _folds, evaluated = grouped_cross_validation(rows, fold_count=3)
    calibrated, reliability = cross_validated_calibration_curve(evaluated, bin_count=5)
    assert len(calibrated) == len(rows) - 1
    assert reliability
    assert all(0.0 <= row["calibrated_error_probability"] <= 1.0 for row in calibrated)
    assert "p1_probability" not in calibrated[0]


def test_primary_metrics_define_false_ready_separately_from_wrong_result() -> None:
    rows = make_records()
    policy = ReviewPolicy(0.10, 0.60, 0.50)
    evaluated = []
    for row in rows:
        state, reasons = apply_review_policy(row, policy)
        evaluated.append({**row, "policy_state": state, "review_reasons": list(reasons)})
    table = {row["metric"]: row for row in primary_metric_table(evaluated)}
    assert table["wrong_result_rate"]["denominator"] == len(rows) - 1
    assert table["false_ready_rate"]["denominator"] == sum(
        row["policy_state"] == "result_ready" for row in evaluated
    )
    low, high = wilson_interval(2, 10)
    assert low < 0.2 < high


def test_evaluator_cannot_open_without_verified_prediction_boundary(tmp_path: Path) -> None:
    missing = tmp_path / "calibration_evaluator_correspondence.jsonl"
    with pytest.raises(TypeError, match="verified calibration prediction boundary"):
        open_calibration_evaluator(object(), missing)  # type: ignore[arg-type]


def test_raw_calibration_resource_is_mapped_only_after_prediction_lock(
    tmp_path: Path, monkeypatch
) -> None:
    contract = SimpleNamespace(
        cohort_id="calibration-fixture",
        membership_sha256="a" * 64,
        scene_count=1,
        image_count=20,
        asset_group_count=1,
    )
    monkeypatch.setattr(calibration, "CALIBRATION_CONTRACT", contract)
    prediction_path = tmp_path / "prediction_manifest.jsonl"
    prediction_rows = [
        {
            "opaque_scene_id": "S001",
            "view_id": f"V{index:02d}",
            "objects": [{"object_id": "P00", "box_xyxy": [10, 10, 30, 30]}],
        }
        for index in range(20)
    ]
    prediction_path.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    multiview_path = tmp_path / "multiview_prediction_manifest.jsonl"
    multiview_rows = [
        {
            "opaque_scene_id": "S001",
            "view_count": view_count,
            "tracks": [
                {
                    "identity_id": "I00",
                    "members": [
                        {"view_id": f"V{index:02d}", "candidate_id": "P00"}
                        for index in range(view_count)
                    ],
                }
            ],
        }
        for view_count in (2, 3, 5)
    ]
    multiview_path.write_text(
        "".join(json.dumps(row) + "\n" for row in multiview_rows),
        encoding="utf-8",
    )
    boundary = VerifiedCalibrationBoundary(
        lock_path=tmp_path / "prediction_lock.json",
        lock_sha256="b" * 64,
        lock_record={},
        run_record_path=tmp_path / "run_record.json",
        run_record_sha256="c" * 64,
        method_record_path=tmp_path / "method_freeze.json",
        method_record_sha256="d" * 64,
        chosen_configuration_sha256="e" * 64,
        prediction_manifest_path=prediction_path,
        multiview_manifest_path=multiview_path,
        embedding_path=tmp_path / "global_embeddings.npy",
        embedding_index_path=tmp_path / "embedding_index.jsonl",
    )
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    raw = {
        "crosswalk_private.jsonl": {
            "cohort_id": contract.cohort_id,
            "opaque_scene_id": "S001",
            "base_asset_id": "A001",
            "category": "animal",
            "object_id_map": {"O00": 0},
        },
        "labels_private.jsonl": {
            "cohort_id": contract.cohort_id,
            "opaque_scene_id": "S001",
            "object_labels": [
                {"object_id": "O00", "is_anomaly": True, "raw_label": 1}
            ],
            "anomaly_type": "colour",
        },
        "geometry_private.jsonl": {
            "cohort_id": contract.cohort_id,
            "opaque_scene_id": "S001",
            "views": [
                {"view_id": f"V{index:02d}", "boxes_xyxy": [[10, 10, 30, 30]]}
                for index in range(20)
            ],
        },
    }
    files = []
    for name, row in raw.items():
        path = evaluator / name
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        files.append(
            {"path": name, "sha256": sha256(path.read_bytes()).hexdigest()}
        )
    (evaluator / "evaluator_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "visual-peer-official-evaluator-manifest/v1",
                "cohort_id": contract.cohort_id,
                "source_membership_sha256": contract.membership_sha256,
                "scene_count": 1,
                "image_count": 20,
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    rows, receipt = open_calibration_evaluator(boundary, evaluator)

    assert len(rows) == 3
    assert {row["acceptable_anomalous_identity_ids"][0] for row in rows} == {"I00"}
    assert {row["category"] for row in rows} == {"animal"}
    assert {row["anomaly_type"] for row in rows} == {"colour"}
    assert receipt["raw_reference_mapping_created_after_lock"] is True


def test_freeze_carries_immutable_method_and_pipeline_hashes(tmp_path: Path) -> None:
    chosen = {"score_method": "P1", "fusion_method": "mean_normalized_rank"}
    method = {
        "schema_version": "visual-peer-08f-workflow/v1",
        "status": "frozen",
        "chosen_configuration": chosen,
        "chosen_configuration_sha256": "unused-in-this-focused-test",
    }
    method_path = tmp_path / "method_freeze.json"
    method_path.write_text(json.dumps(method), encoding="utf-8")
    identity = {
        "split_sha256": "1" * 64,
        "configuration_sha256": "2" * 64,
        "locateanything_sha256": "3" * 64,
        "sam21_sha256": "4" * 64,
        "dinov2_sha256": "5" * 64,
        "preprocessing_sha256": "6" * 64,
        "schema_sha256": "7" * 64,
    }
    boundary = VerifiedCalibrationBoundary(
        lock_path=tmp_path / "prediction_lock.json",
        lock_sha256="8" * 64,
        lock_record={"identity": identity, "identity_sha256": "9" * 64},
        run_record_path=tmp_path / "run_record.json",
        run_record_sha256="a" * 64,
        method_record_path=method_path,
        method_record_sha256="b" * 64,
        chosen_configuration_sha256="c" * 64,
        prediction_manifest_path=tmp_path / "prediction_manifest.jsonl",
        multiview_manifest_path=tmp_path / "multiview_prediction_manifest.jsonl",
        embedding_path=tmp_path / "global_embeddings.npy",
        embedding_index_path=tmp_path / "embedding_index.jsonl",
    )
    cv_rows = [{"cv_fold": 0}, {"cv_fold": 1}]
    output = tmp_path / "review_policy_freeze.json"
    frozen = freeze_review_policy(
        boundary=boundary,
        policy=ReviewPolicy(0.1, 0.5, 0.25),
        cv_rows=cv_rows,
        evaluator_access_receipt={"evaluator_sha256": "d" * 64},
        output_path=output,
        review_cost=0.15,
        insufficient_cost=0.25,
    )
    assert frozen["immutable_pipeline_identity"] == identity
    assert frozen["method_record_sha256"] == "b" * 64
    assert frozen["states"].keys() == {"result_ready", "review_required", "insufficient_evidence"}
    assert frozen["all_normal_scene_claim_supported"] is False
    assert frozen["selection"]["raw_p1_is_probability"] is False


def test_evaluator_examples_are_capped_even_when_many_ready_results_are_wrong() -> None:
    rows = make_records()
    for index, row in enumerate(rows):
        row["policy_state"] = "result_ready"
        row["is_correct"] = False
        row["review_reasons"] = []
    selected = select_evaluator_examples(rows, count=16)
    assert len(selected) == 16
    assert all(row["is_correct"] is False for row in selected)


def test_evaluator_example_selection_is_deterministic_and_stratified() -> None:
    rows = make_records()
    for index, row in enumerate(rows):
        if index == 0:
            row["policy_state"] = "result_ready"
            row["is_correct"] = False
            row["review_reasons"] = []
        elif index == 1:
            row["policy_state"] = "insufficient_evidence"
            row["review_reasons"] = ["insufficient_evidence"]
            row["insufficient_evidence"] = True
        else:
            row["policy_state"] = "review_required"
            row["review_reasons"] = ["small_p1_margin"]
    forward = select_evaluator_examples(rows, count=6)
    reverse = select_evaluator_examples(list(reversed(rows)), count=6)
    assert [
        (row["opaque_scene_id"], row["view_count"]) for row in forward
    ] == [
        (row["opaque_scene_id"], row["view_count"]) for row in reverse
    ]
    covered = {
        stratum for row in forward for stratum in row["example_selection_strata"]
    }
    assert {
        "policy_state:result_ready",
        "policy_state:review_required",
        "policy_state:insufficient_evidence",
    }.issubset(covered)


def test_wrong_result_ready_is_represented_when_available() -> None:
    rows = make_records()
    for row in rows:
        row["policy_state"] = "review_required"
        row["is_correct"] = True
        row["review_reasons"] = ["small_p1_margin"]
    rows[-1]["policy_state"] = "result_ready"
    rows[-1]["is_correct"] = False
    rows[-1]["review_reasons"] = []
    selected = select_evaluator_examples(rows, count=2)
    assert any(
        row["policy_state"] == "result_ready" and row["is_correct"] is False
        for row in selected
    )
    assert "wrong_result_ready" in selected[0]["example_selection_strata"]


def test_exact_resume_rejects_changed_inputs_or_outputs(tmp_path: Path) -> None:
    output = tmp_path / "safe" / "metrics.csv"
    output.parent.mkdir(parents=True)
    output.write_text("metric,value\ncoverage,0.5\n", encoding="utf-8")
    receipt = tmp_path / "evaluation_receipt.json"
    hashes = {"prediction_lock_sha256": "a" * 64}
    write_exact_resume_receipt(receipt, input_hashes=hashes, output_paths=[output])
    assert exact_resume_receipt_valid(receipt, hashes)
    assert not exact_resume_receipt_valid(receipt, {"prediction_lock_sha256": "b" * 64})
    output.write_text("metric,value\ncoverage,0.6\n", encoding="utf-8")
    assert not exact_resume_receipt_valid(receipt, hashes)


def test_source_has_no_universal_point_nine_five_acceptance_gate() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/visual_peer_inspector/workflows/calibration_review_policy.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "minimum_accuracy = 0.95" not in lowered
    assert "minimum_recall = 0.95" not in lowered
    assert "acceptance_threshold = 0.95" not in lowered
    assert "not a probability" in lowered
