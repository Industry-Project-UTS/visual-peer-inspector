from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from visual_peer_inspector.data.cohort_contract import CALIBRATION_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.models.sam2 import encode_binary_mask
from visual_peer_inspector.representation.cross_view_identity import (
    IdentityConfiguration,
)
from visual_peer_inspector.review_policy import (
    ReviewPolicy,
    apply_review_policy,
    build_prediction_review_signals,
)
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    FrozenPipelineIdentity,
    StageOutcome,
)
from visual_peer_inspector.workflows.protected_cohort_inference import (
    VIEW_CONTRACTS,
    assert_evaluator_absent,
    build_multiview_predictions,
    build_protected_identity,
    load_method_freeze,
    load_policy_freeze,
    read_stage,
    run_exact_sharded_stage,
    safe_aggregate_tables,
    select_prediction_audit_rows,
    verify_frozen_base_route,
)


def digest(value) -> str:
    from hashlib import sha256

    return sha256(canonical_json_bytes(value)).hexdigest()


def base_identity() -> FrozenPipelineIdentity:
    values = [f"{index:064x}" for index in range(1, 8)]
    return FrozenPipelineIdentity(*values)


def write_freezes(root: Path) -> tuple[Path, Path, Path]:
    configuration = asdict(IdentityConfiguration())
    method = {
        "schema_version": "visual-peer-08f-workflow/v1",
        "status": "frozen",
        "chosen_variant": "embedding_only",
        "chosen_configuration": configuration,
        "chosen_configuration_sha256": digest(configuration),
        "frozen_model_identity_sha256": base_identity().sha256,
    }
    method_path = root / "method_freeze.json"
    method_path.write_text(json.dumps(method), encoding="utf-8")
    application = {
        "schema_version": "visual-peer-application-output/v1",
        "status": "frozen",
        "identity_configuration": configuration,
        "identity_configuration_sha256": digest(configuration),
        "frozen_model_identity_sha256": base_identity().sha256,
    }
    application_path = root / "application_output_contract.json"
    application_path.write_text(json.dumps(application), encoding="utf-8")
    policy_configuration = asdict(ReviewPolicy(0.10, 0.50, 0.25))
    policy = {
        "schema_version": "visual-peer-09b-review-policy/v1",
        "status": "frozen",
        "method_record_sha256": file_sha256(method_path),
        "minimum_margin": 0.10,
        "minimum_agreement": 0.50,
        "minimum_mask_quality": 0.25,
        "policy_configuration": policy_configuration,
        "policy_configuration_sha256": digest(policy_configuration),
    }
    policy_path = root / "review_policy_freeze.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return method_path, application_path, policy_path


def test_method_and_policy_records_are_hash_linked() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        method_path, application_path, policy_path = write_freezes(root)
        method = load_method_freeze(method_path, application_contract_path=application_path)
        policy = load_policy_freeze(policy_path, method_sha256=method["sha256"])

        assert method["view_contracts"] == VIEW_CONTRACTS
        assert method["score_method"] == "P1"
        assert policy["minimum_margin"] == 0.10
        with pytest.raises(ValueError, match="not frozen against"):
            load_policy_freeze(policy_path, method_sha256="0" * 64)


def test_protected_identity_binds_cohort_and_gate_hashes() -> None:
    identity = build_protected_identity(
        base_identity(),
        contract=CALIBRATION_CONTRACT,
        method_sha256="a" * 64,
        policy_sha256=None,
    )

    assert identity.split_sha256 == CALIBRATION_CONTRACT.membership_sha256
    assert identity.sha256 != base_identity().sha256


def test_mounted_base_route_must_match_the_method_freeze() -> None:
    method = {"frozen_model_identity_sha256": base_identity().sha256}
    verify_frozen_base_route(base_identity(), method)

    changed_values = [f"{index:064x}" for index in range(2, 9)]
    changed = FrozenPipelineIdentity(*changed_values)
    with pytest.raises(ValueError, match="differs from the frozen 08E route"):
        verify_frozen_base_route(changed, method)


def test_evaluator_like_mount_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "calibration-rgb"
        model.mkdir()
        assert_evaluator_absent(
            [model, root / "code-snapshot"],
            permitted_roots=(model, root / "code-snapshot"),
        )
        with pytest.raises(ValueError, match="Evaluator-like"):
            assert_evaluator_absent(
                [model, root / "calibration-labels"],
                permitted_roots=(model,),
            )
        broad_notebook_output = root / "apparently-safe-notebook-output"
        (broad_notebook_output / "review-only").mkdir(parents=True)
        (broad_notebook_output / "review-only" / "scene_labels.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Evaluator-like"):
            assert_evaluator_absent(
                [model, broad_notebook_output],
                permitted_roots=(model,),
            )

        combined_mount = root / "combined-mount"
        allowed_child = combined_mount / "model-input"
        forbidden_sibling = combined_mount / "evaluator"
        allowed_child.mkdir(parents=True)
        forbidden_sibling.mkdir(parents=True)
        (forbidden_sibling / "labels_private.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Evaluator-like"):
            assert_evaluator_absent(
                [combined_mount],
                permitted_roots=(allowed_child,),
            )


def test_shards_resume_only_after_exact_hash_verification() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = [
            {"opaque_scene_id": "S001", "view_id": "V00", "value": 1},
            {"opaque_scene_id": "S001", "view_id": "V01", "value": 2},
        ]

        def worker(row):
            return {"status": "ok", "value": row["value"] * 2}

        first = run_exact_sharded_stage(
            stage="locateanything",
            input_records=rows,
            output_root=root,
            identity=base_identity(),
            worker=worker,
            shard_size=1,
            shard_limit=1,
        )
        second = run_exact_sharded_stage(
            stage="locateanything",
            input_records=rows,
            output_root=root,
            identity=base_identity(),
            worker=worker,
            shard_size=1,
            shard_limit=1,
        )
        assert first["processed_this_run"] == 1
        assert second["resumed"] == 1
        output_path = root / "locateanything/shard-00000.jsonl"
        output_path.write_text(output_path.read_text() + "\n", encoding="utf-8")
        repaired = run_exact_sharded_stage(
            stage="locateanything",
            input_records=rows,
            output_root=root,
            identity=base_identity(),
            worker=worker,
            shard_size=1,
            shard_limit=1,
        )
        assert repaired["processed_this_run"] == 1
        assert len(read_stage(root / "locateanything")) == 1


def test_resumed_failure_totals_match_the_first_run() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = [{"opaque_scene_id": "S001", "view_id": "V00", "value": 1}]

        def worker(_row):
            raise RuntimeError("unexpected")

        first = run_exact_sharded_stage(
            stage="sam21",
            input_records=rows,
            output_root=root,
            identity=base_identity(),
            worker=worker,
            shard_size=1,
        )
        resumed = run_exact_sharded_stage(
            stage="sam21",
            input_records=rows,
            output_root=root,
            identity=base_identity(),
            worker=worker,
            shard_size=1,
        )
        assert first["failure_counts"]["stage_failure"] == 1
        assert resumed["failure_counts"] == first["failure_counts"]
        assert resumed["terminal_failures"] == 1


def test_embedding_shard_has_one_output_per_input() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = [{"opaque_scene_id": "S001", "view_id": "V00"}]
        run_exact_sharded_stage(
            stage="dinov2_small",
            input_records=rows,
            output_root=root,
            identity=base_identity(),
            worker=lambda _row: StageOutcome(
                record={"status": "ok", "objects": [{"object_id": "P01"}]},
                global_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            ),
            shard_size=100,
        )
        assert len(read_stage(root / "dinov2_small")) == 1


def synthetic_predictions():
    required_views = sorted({view for contract in VIEW_CONTRACTS.values() for view in contract})
    rows = []
    vectors = []
    index = {}
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    rle = encode_binary_mask(mask)
    for view_id in required_views:
        objects = []
        start = len(vectors)
        for object_index, embedding in enumerate(([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]), 1):
            x1 = 1 + (object_index - 1) * 5
            objects.append({
                "object_id": f"{view_id}-P{object_index:02d}",
                "box_xyxy": [x1, 2, x1 + 4, 8],
                "mask_rle": rle,
                "foreground_pixels": int(mask.sum()),
                "predicted_quality_score": 0.90,
                "p1_rank": object_index,
                "p1_score": float(4 - object_index),
            })
            vectors.append(embedding)
        rows.append({
            "opaque_scene_id": "S001",
            "view_id": view_id,
            "image_path": __file__,
            "status": "ok",
            "objects": objects,
        })
        index[("S001", view_id)] = {"start": start, "count": 3}
    return rows, np.asarray(vectors, dtype=np.float16), index


def test_two_three_five_view_predictions_preserve_review_state(monkeypatch) -> None:
    rows, embeddings, index = synthetic_predictions()

    class Opened:
        size = (16, 16)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("visual_peer_inspector.workflows.protected_cohort_inference.Image.open", lambda _path: Opened())
    deployed_policy = ReviewPolicy(0.1, 0.5, 0.25)
    predictions = build_multiview_predictions(
        rows,
        embeddings,
        index,
        identity_configuration=IdentityConfiguration(unmatched_cost=0.5),
        policy={"policy_configuration": asdict(deployed_policy)},
    )

    assert [row["view_count"] for row in predictions] == [2, 3, 5]
    assert all(row["fused_odd_one_out"] is not None for row in predictions)
    assert all(row["review_signal"] == "automatic" for row in predictions)
    selected_rows = {
        str(row["view_id"]): row for row in rows
    }
    for prediction in predictions:
        signals = build_prediction_review_signals(
            prediction,
            [selected_rows[view_id] for view_id in prediction["view_ids"]],
        )
        state, reasons = apply_review_policy(
            {**prediction, **signals}, deployed_policy
        )
        assert prediction["policy_state"] == state
        assert prediction["review_reasons"] == list(reasons)
    contract_rows, review_rows = safe_aggregate_tables(predictions)
    assert [row["scene_count"] for row in contract_rows] == [1, 1, 1]
    assert [row["result_available_count"] for row in contract_rows] == [1, 1, 1]
    assert [row["result_ready_count"] for row in contract_rows] == [1, 1, 1]
    assert [row["review_required_count"] for row in contract_rows] == [0, 0, 0]
    assert [row["policy_pending_count"] for row in contract_rows] == [0, 0, 0]
    assert sum(row["scene_count"] for row in review_rows if row["review_signal"] == "automatic") == 3


def test_qualitative_selection_covers_available_states_and_failures() -> None:
    view_ids = list(VIEW_CONTRACTS[5])
    decisions = [
        {
            "opaque_scene_id": "S-AUTOMATIC",
            "view_count": 5,
            "view_ids": view_ids,
            "policy_state": "result_ready",
            "review_signal": "automatic",
        },
        {
            "opaque_scene_id": "S-REVIEW",
            "view_count": 5,
            "view_ids": view_ids,
            "policy_state": "review_required",
            "review_signal": "review",
        },
        {
            "opaque_scene_id": "S-INSUFFICIENT",
            "view_count": 5,
            "view_ids": view_ids,
            "policy_state": "insufficient_evidence",
            "review_signal": "insufficient_evidence",
        },
        {
            "opaque_scene_id": "S-FULL-FAILURE",
            "view_count": 5,
            "view_ids": view_ids,
            "policy_state": "insufficient_evidence",
            "review_signal": "insufficient_evidence",
        },
        {
            "opaque_scene_id": "S-PARTIAL-FAILURE",
            "view_count": 5,
            "view_ids": view_ids,
            "policy_state": "review_required",
            "review_signal": "review",
        },
    ]
    score_rows = []
    for decision in decisions:
        for index, view_id in enumerate(view_ids):
            failure = None
            if decision["opaque_scene_id"] == "S-FULL-FAILURE":
                failure = "zero_proposal"
            elif decision["opaque_scene_id"] == "S-PARTIAL-FAILURE" and index == 0:
                failure = "feature_failure"
            score_rows.append(
                {
                    "opaque_scene_id": decision["opaque_scene_id"],
                    "view_id": view_id,
                    "terminal_failure": failure,
                }
            )

    selected = select_prediction_audit_rows(
        score_rows=list(reversed(score_rows)),
        multiview_rows=list(reversed(decisions)),
        count=5,
    )

    selected_strata = {
        value for row in selected for value in row["audit_selection_strata"]
    }
    assert "policy_state:review_required" in selected_strata
    assert "policy_state:insufficient_evidence" in selected_strata
    assert "failure_scope:full_scene" in selected_strata
    assert "terminal_failure:zero_proposal" in selected_strata
    assert "terminal_failure:feature_failure" in selected_strata
    assert [row["opaque_scene_id"] for row in selected] == [
        row["opaque_scene_id"]
        for row in select_prediction_audit_rows(
            score_rows=score_rows,
            multiview_rows=decisions,
            count=5,
        )
    ]


def test_safe_aggregate_separates_available_ready_review_and_pending() -> None:
    rows = [
        {
            "view_count": 5,
            "insufficient_evidence": False,
            "policy_state": "result_ready",
            "review_signal": "automatic",
            "margin": 0.4,
            "agreement": 1.0,
        },
        {
            "view_count": 5,
            "insufficient_evidence": False,
            "policy_state": "review_required",
            "review_signal": "review",
            "margin": 0.1,
            "agreement": 0.4,
        },
        {
            "view_count": 5,
            "insufficient_evidence": False,
            "policy_state": "policy_pending",
            "review_signal": "policy_pending",
            "margin": 0.2,
            "agreement": 0.6,
        },
        {
            "view_count": 5,
            "insufficient_evidence": True,
            "policy_state": "insufficient_evidence",
            "review_signal": "insufficient_evidence",
            "margin": None,
            "agreement": None,
        },
    ]

    contract_rows, _review_rows = safe_aggregate_tables(rows)
    summary = next(row for row in contract_rows if row["view_count"] == 5)

    assert summary["scene_count"] == 4
    assert summary["result_available_count"] == 3
    assert summary["result_ready_count"] == 1
    assert summary["review_required_count"] == 1
    assert summary["policy_pending_count"] == 1
    assert summary["insufficient_evidence_count"] == 1
