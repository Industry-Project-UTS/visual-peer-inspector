from __future__ import annotations

import json
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import file_sha256, membership_sha256
from visual_peer_inspector.frozen_resource import verify_frozen_resource
from visual_peer_inspector.models.sam2 import (
    binary_mask_sha256,
    decode_binary_mask,
    encode_binary_mask,
)
from visual_peer_inspector.workflows.cross_view_identity_freeze import (
    EVALUATOR_SCHEMA_VERSION,
    _combined_heatmap,
    _load_evaluator_rows,
    _prediction_lock,
    load_08e_predictions,
    run_cross_view_identity_freeze,
    verify_identity_prediction_lock,
)
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    FrozenPipelineIdentity,
    lock_predictions,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def make_fixture(
    root: Path, *, camera_prefix: str = "camera"
) -> tuple[Path, Path, Path]:
    prediction_root = root / "predictions" / "model-safe"
    media_root = root / "media-package"
    evaluator_path = root / "private" / "evaluator_correspondence.jsonl"
    prediction_root.mkdir(parents=True)
    vectors = []
    index_rows = []
    manifest_rows = []
    evaluator_rows = []
    bases = ("asset-a", "asset-b", "asset-c")
    identities = ("peer-a", "peer-b", "peer-c")
    centres = ((7, 7), (16, 16), (25, 25))

    for scene_index in range(12):
        scene_id = f"S{scene_index:03d}"
        asset_id = bases[scene_index % len(bases)]
        for view_index in range(5):
            view_id = f"V{view_index:02d}"
            image = np.full((32, 32, 3), 35 + scene_index, dtype=np.uint8)
            image_path = media_root / "media" / scene_id / "rgb" / f"{view_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image, mode="RGB").save(image_path)
            objects = []
            for object_index, (identity_id, (cx, cy)) in enumerate(
                zip(identities, centres)
            ):
                object_id = f"O{view_index}{object_index}"
                x1, y1 = cx - 3, cy - 3
                x2, y2 = cx + 3, cy + 3
                mask = np.zeros((32, 32), dtype=bool)
                mask[y1:y2, x1:x2] = True
                rle = encode_binary_mask(mask)
                vector = np.zeros(4, dtype=np.float16)
                vector[object_index] = np.float16(1.0)
                vector[3] = np.float16(view_index * 0.001)
                vector /= np.linalg.norm(vector.astype(np.float32)).astype(np.float16)
                row_index = len(vectors)
                vectors.append(vector)
                index_rows.append(
                    {
                        "schema_version": 1,
                        "cohort_id": "development-fixture",
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "object_id": object_id,
                        "row_index": row_index,
                        "feature_dimension": 4,
                        "vector_sha256": sha256(
                            np.ascontiguousarray(vector).tobytes(order="C")
                        ).hexdigest(),
                    }
                )
                p1_score = (0.9, 0.4, 0.1)[object_index] + view_index * 0.001
                objects.append(
                    {
                        "object_id": object_id,
                        "box_xyxy": [x1, y1, x2, y2],
                        "mask_rle": rle,
                        "mask_sha256": binary_mask_sha256(mask),
                        "foreground_pixels": int(mask.sum()),
                        "predicted_quality_score": 0.8,
                        "b0_status": "scored",
                        "b0_score": p1_score - 0.05,
                        "p1_status": "scored",
                        "p1_score": p1_score,
                        "p1_rank": object_index + 1,
                        "p1_heatmap": [[p1_score, 0.0], [0.0, p1_score / 2]],
                    }
                )
                evaluator_rows.append(
                    {
                        "schema_version": EVALUATOR_SCHEMA_VERSION,
                        "opaque_scene_id": scene_id,
                        "opaque_asset_group_id": asset_id,
                        "view_id": view_id,
                        "object_id": object_id,
                        "reference_identity_id": identity_id,
                        "camera_diagnostic": f"{camera_prefix}-{view_index}",
                    }
                )
            manifest_rows.append(
                {
                    "schema_version": 1,
                    "cohort_id": "development-fixture",
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "image_sha256": file_sha256(image_path),
                    "status": "complete",
                    "terminal_failure": None,
                    "object_count": len(objects),
                    "objects": objects,
                    "per_view_p1_candidate": objects[0]["object_id"],
                    "per_view_p1_score": objects[0]["p1_score"],
                }
            )

    write_jsonl(prediction_root / "prediction_manifest.jsonl", manifest_rows)
    embeddings = np.stack(vectors).astype(np.float16)
    with (prediction_root / "global_embeddings.npy").open("wb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
    write_jsonl(prediction_root / "embedding_index.jsonl", index_rows)
    identity = FrozenPipelineIdentity(
        split_sha256="1" * 64,
        configuration_sha256="2" * 64,
        locateanything_sha256="3" * 64,
        sam21_sha256="4" * 64,
        dinov2_sha256="5" * 64,
        preprocessing_sha256="6" * 64,
        schema_sha256="7" * 64,
    )
    lock_predictions(
        prediction_root=prediction_root,
        identity=identity,
        cohort_id="development-fixture",
        cohort_membership_sha256="8" * 64,
        scene_count=12,
        image_count=60,
    )
    write_jsonl(evaluator_path, evaluator_rows)
    (evaluator_path.parent / "evaluator_manifest.json").write_text(json.dumps({
        "schema_version": "visual-peer-primary-resource-release/v1",
        "status": "locked", "scene_count": 12, "image_count": 60,
        "membership_sha256": membership_sha256(sorted({row["opaque_scene_id"] for row in evaluator_rows})),
        "correspondence_row_count": len(evaluator_rows),
        "reference_geometry_policy": {
            "decoder": "toysad-grayscale-threshold-v1", "mask_threshold": 128,
            "image_size": [256, 256], "box_format": "half-open-xyxy",
        },
        "correspondence_sha256": file_sha256(evaluator_path),
    }))
    return prediction_root, media_root, evaluator_path


def run_fixture(root: Path, *, camera_prefix: str = "camera") -> tuple[dict, Path]:
    prediction_root, media_root, evaluator = make_fixture(
        root / "inputs", camera_prefix=camera_prefix
    )
    output = root / "output"
    record = run_cross_view_identity_freeze(
        prediction_roots=[prediction_root],
        evaluator_correspondence_path=evaluator,
        media_roots=[media_root],
        output_root=output,
        bootstrap_repetitions=30,
        bootstrap_seed=7,
        qualitative_count=12,
    )
    return record, output


def test_patch_heatmap_is_projected_only_inside_the_selected_object(tmp_path):
    prediction_root, media_root, _ = make_fixture(tmp_path)
    loaded = load_08e_predictions([prediction_root])
    # Keep invalid background patches, as the real P1 output does.
    view = loaded.views_by_scene["S000"][0]
    selected = replace(view.objects[0], p1_heatmap=((float("nan"), 0.9), (0.6, 0.5)))
    view = replace(view, objects=(selected, *view.objects[1:]))
    with Image.open(media_root / "media/S000/rgb/V00.png") as rgb:
        projected = _combined_heatmap(view, rgb, {selected.object_id})
        assert _combined_heatmap(view, rgb, set()) is None
    mask = decode_binary_mask(selected.mask_rle)
    assert projected.shape == mask.shape
    assert np.isfinite(projected[mask]).any()
    assert np.isnan(projected[~mask]).all()
    assert np.nanmax(projected) <= 0.9 + 1e-6


@pytest.mark.parametrize("field,value", [
    ("correspondence_sha256", "0" * 64),
    ("scene_count", 13),
    ("correspondence_row_count", 0),
    ("reference_geometry_policy", None),
    ("reference_geometry_policy", {"mask_threshold": 0}),
    ("membership_sha256", "0" * 64),
])
def test_evaluator_manifest_rejects_changed_resource(tmp_path, field, value):
    _, _, evaluator = make_fixture(tmp_path / "inputs")
    output = tmp_path / "output"
    prediction_root = output / "prediction-only"
    prediction_root.mkdir(parents=True)
    (prediction_root / "fixture.json").write_text('{}')
    _prediction_lock(output, {"scene_count": 12})
    manifest_path = evaluator.parent / "evaluator_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Evaluator"):
        _load_evaluator_rows(
            evaluator,
            prediction_lock_path=prediction_root / "identity_prediction_lock.json",
        )


def test_fixture_run_locks_predictions_before_evaluation_and_renders_evidence(
    tmp_path: Path,
) -> None:
    record, output = run_fixture(tmp_path)

    assert record["status"] == "complete"
    assert record["scene_count"] == 12
    assert record["prediction_count"] == 72
    assert record["qualitative_panel_count"] == 12
    assert record["all_acceptance_gates_passed"] is True
    lock = verify_identity_prediction_lock(
        output / "prediction-only" / "identity_prediction_lock.json"
    )
    assert lock["status"] == "locked_before_evaluator_access"
    receipt = json.loads(
        (output / "private" / "evaluator_access_receipt.json").read_text()
    )
    assert receipt["status"] == "opened_after_identity_prediction_lock"
    assert receipt["identity_prediction_lock_sha256"] == file_sha256(
        output / "prediction-only" / "identity_prediction_lock.json"
    )
    assert (
        len(
            list(
                (output / "review-only" / "figures" / "odd-one-out").glob(
                    "*-prediction.png"
                )
            )
        )
        == 12
    )
    assert (
        len(
            list(
                (output / "review-only" / "figures" / "odd-one-out").glob(
                    "*-sam-evidence.png"
                )
            )
        )
        == 12
    )
    application_rows = [
        json.loads(line)
        for line in (output / "downstream" / "application_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(application_rows) == 36
    assert {row["view_count"] for row in application_rows} == {2, 3, 5}
    assert all(
        row["schema_version"] == "visual-peer-application-output/v1"
        for row in application_rows
    )
    assert all(row["fused_odd_one_out"] is not None for row in application_rows)
    safe_files = {
        path.relative_to(output / "safe").as_posix()
        for path in (output / "safe").rglob("*")
        if path.is_file()
    }
    assert "application_predictions.jsonl" not in safe_files
    assert "qualitative_manifest.csv" not in safe_files
    assert "failure_receipts.jsonl" not in safe_files
    assert not any("odd-one-out" in path for path in safe_files)
    method_resource = output / "publish" / "method-freeze"
    resource_record = verify_frozen_resource(
        method_resource,
        expected_stage="cross_view_identity_method_release",
        expected_files={
            "application_output_contract.json",
            "method_freeze.json",
        },
    )
    assert resource_record["bindings"]["identity_prediction_lock_sha256"] == file_sha256(
        output / "prediction-only" / "identity_prediction_lock.json"
    )


def test_failed_acceptance_gate_withholds_method_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "visual_peer_inspector.workflows.cross_view_identity_freeze.select_publication_results",
        lambda *_args, **_kwargs: (),
    )
    record, output = run_fixture(tmp_path)
    assert record["status"] == "completed_with_failed_gates"
    assert record["all_acceptance_gates_passed"] is False
    assert record["method_resource_manifest_sha256"] is None
    assert not (output / "publish" / "method-freeze").exists()


def test_every_variant_and_view_count_has_clustered_metrics(tmp_path: Path) -> None:
    _, output = run_fixture(tmp_path)
    with (output / "safe" / "clustered_intervals.csv").open(newline="") as handle:
        rows = list(__import__("csv").DictReader(handle))

    observed = {(row["variant"], int(row["view_count"]), row["metric"]) for row in rows}
    for variant in ("embedding_only", "normalized_geometry"):
        for view_count in (2, 3, 5):
            for metric in (
                "observation_coverage",
                "unmatched_rate",
                "ambiguity_rate",
                "pairwise_identity_accuracy",
                "switch_rate",
                "cycle_consistency_rate",
            ):
                assert (variant, view_count, metric) in observed
    two_view_cycles = [
        row
        for row in rows
        if int(row["view_count"]) == 2 and row["metric"] == "cycle_consistency_rate"
    ]
    assert {row["status"] for row in two_view_cycles} == {"not_applicable"}


def test_verified_receipts_resume_without_recomputing(tmp_path: Path) -> None:
    prediction_root, media_root, evaluator = make_fixture(tmp_path / "inputs")
    output = tmp_path / "output"
    first = run_cross_view_identity_freeze(
        prediction_roots=[prediction_root],
        evaluator_correspondence_path=evaluator,
        media_roots=[media_root],
        output_root=output,
        bootstrap_repetitions=20,
        qualitative_count=12,
    )
    second = run_cross_view_identity_freeze(
        prediction_roots=[prediction_root],
        evaluator_correspondence_path=evaluator,
        media_roots=[media_root],
        output_root=output,
        bootstrap_repetitions=20,
        qualitative_count=12,
    )

    assert first["processed_this_run"] == 96
    assert second["processed_this_run"] == 0
    assert second["resumed_from_verified_receipts"] == 96
    assert (
        first["identity_prediction_lock_sha256"]
        == second["identity_prediction_lock_sha256"]
    )


def test_camera_diagnostics_cannot_change_predictions_or_method(tmp_path: Path) -> None:
    first_record, first_output = run_fixture(tmp_path / "first", camera_prefix="front")
    second_record, second_output = run_fixture(
        tmp_path / "second", camera_prefix="random"
    )

    assert (
        first_record["identity_prediction_lock_sha256"]
        == second_record["identity_prediction_lock_sha256"]
    )
    first_freeze = json.loads(
        (first_output / "safe" / "method_freeze.json").read_text()
    )
    second_freeze = json.loads(
        (second_output / "safe" / "method_freeze.json").read_text()
    )
    assert first_freeze == second_freeze
    assert first_freeze["camera_metadata_used_by_chosen_method"] is False
    assert first_freeze["absolute_performance_threshold_used"] is False


def test_input_lock_corruption_is_rejected(tmp_path: Path) -> None:
    prediction_root, _, _ = make_fixture(tmp_path)
    with (prediction_root / "prediction_manifest.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="artifact differs"):
        load_08e_predictions([prediction_root])
