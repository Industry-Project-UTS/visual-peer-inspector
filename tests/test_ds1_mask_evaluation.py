from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import canonical_json_bytes, file_sha256, membership_sha256
from visual_peer_inspector.evaluation import (
    ds1_evaluator_package,
    proposal_reference_ds1,
)
from visual_peer_inspector.models.sam2 import binary_mask_sha256, encode_binary_mask
from visual_peer_inspector.workflows import mask_evaluation_ds1
from visual_peer_inspector.data import ds1_mask_route_package


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _write_artifact_manifest(root: Path) -> None:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    (root / "artifact_manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
            "manifest_content_sha256": sha256(
                b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
            ).hexdigest(),
        }),
        encoding="utf-8",
    )


def _small_contract() -> CohortContract:
    scenes = ["S000001", "S000002"]
    return CohortContract(
        cohort_id="test-ds1",
        release_id="test-release",
        role="development",
        pilot_stage="screen_100",
        membership_sha256=membership_sha256(scenes),
        scene_count=2,
        views_per_scene=2,
        image_count=4,
        asset_group_count=2,
    )


def _inputs(tmp_path: Path):
    scenes = ["S000001", "S000002"]
    references = []
    attributes = []
    crosswalk = []
    geometry_root = tmp_path / "geometry_source"
    attributes_root = tmp_path / "attributes_source"
    mask_root = geometry_root / "masks"
    box = [40, 50, 120, 150]
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 40:120] = 255
    for scene_index, scene in enumerate(scenes):
        attributes.append({
            "opaque_scene_id": scene,
            "opaque_asset_group_id": f"A{scene_index + 1:03d}",
            "category": f"category_{scene_index + 1}",
            "object_count": 1,
            "anomaly_count": 1,
            "anomaly_families": ["size"],
        })
        for view_index in range(2):
            view = f"V{view_index:02d}"
            image_sha256 = sha256(f"{scene}/{view}".encode()).hexdigest()
            relative = Path(scene) / "O01" / f"{view}.png"
            path = mask_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(path)
            references.append({
                "opaque_scene_id": scene,
                "view_id": view,
                "opaque_object_id": "O01",
                "visible": True,
                "box_xyxy": box,
                "mask_path": relative.as_posix(),
                "mask_sha256": file_sha256(path),
                "foreground_pixels": int(np.count_nonzero(mask)),
            })
            for route, prompt in (("reference_box", "R000"), ("automatic_box", "P000")):
                crosswalk.append({
                    "opaque_scene_id": scene,
                    "view_id": view,
                    "image_sha256": image_sha256,
                    "route": route,
                    "prompt_key": prompt,
                    "prompt_box_xyxy": box,
                    "opaque_object_id": "O01",
                    "proposal_index": 0 if route == "automatic_box" else None,
                })
    reference_path = geometry_root / "reference_manifest.jsonl"
    attributes_path = attributes_root / "protected_scene_attributes.jsonl"
    crosswalk_path = tmp_path / "crosswalk.jsonl"
    _write_jsonl(reference_path, references)
    _write_jsonl(attributes_path, attributes)
    _write_jsonl(crosswalk_path, crosswalk)
    geometry_checksums = {
        "reference_manifest.jsonl": file_sha256(reference_path),
        **{
            f"masks/{row['mask_path']}": file_sha256(mask_root / row["mask_path"])
            for row in references
        },
    }
    geometry_checksum_path = geometry_root / "content_checksums.sha256"
    geometry_checksum_path.write_text(
        "".join(
            f"{digest}  {relative}\n"
            for relative, digest in sorted(geometry_checksums.items())
        ),
        encoding="utf-8",
    )
    geometry_record_path = geometry_root / "package_record.json"
    geometry_record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "package_kind": "ds1-oracle-geometry",
                "cohort_id": _small_contract().cohort_id,
                "release_id": _small_contract().release_id,
                "membership_sha256": _small_contract().membership_sha256,
                "scene_count": _small_contract().scene_count,
                "views_per_scene": _small_contract().views_per_scene,
                "image_count": _small_contract().image_count,
                "asset_group_count": _small_contract().asset_group_count,
                "object_view_count": len(references),
                "visible_object_view_count": len(references),
                "mask_threshold": 128,
                "reference_manifest_sha256": file_sha256(reference_path),
                "content_checksums_sha256": file_sha256(geometry_checksum_path),
                "all_checks_passed": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    attributes_checksum_path = attributes_root / "content_checksums.sha256"
    attributes_checksum_path.write_text(
        f"{file_sha256(attributes_path)}  protected_scene_attributes.jsonl\n",
        encoding="utf-8",
    )
    attributes_record_path = attributes_root / "package_record.json"
    attributes_record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "package_kind": "ds1-evaluator-attributes",
                "cohort_id": _small_contract().cohort_id,
                "release_id": _small_contract().release_id,
                "membership_sha256": _small_contract().membership_sha256,
                "scene_count": _small_contract().scene_count,
                "views_per_scene": _small_contract().views_per_scene,
                "image_count": _small_contract().image_count,
                "asset_group_count": _small_contract().asset_group_count,
                "protected_attributes_sha256": file_sha256(attributes_path),
                "content_checksums_sha256": file_sha256(attributes_checksum_path),
                "all_checks_passed": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    crosswalk_record_path = tmp_path / "crosswalk_package_record.json"
    crosswalk_record_path.write_text(json.dumps({
        "schema_version": 1,
        "package_kind": "ds1-sam-evaluator-crosswalk",
        "intended_use": "evaluator_only",
        "model_facing_allowed": False,
        "cohort_id": _small_contract().cohort_id,
        "release_id": _small_contract().release_id,
        "membership_sha256": _small_contract().membership_sha256,
        "scene_count": _small_contract().scene_count,
        "image_count": _small_contract().image_count,
        "row_count": len(crosswalk),
        "prompt_package_record_sha256": "1" * 64,
        "prompt_manifest_sha256": "5" * 64,
        "model_input_manifest_sha256": "6" * 64,
        "geometry_package_record_sha256": file_sha256(geometry_record_path),
        "geometry_manifest_sha256": file_sha256(reference_path),
        "proposal_results_sha256": "4" * 64,
        "crosswalk_sha256": file_sha256(crosswalk_path),
        "all_checks_passed": True,
    }), encoding="utf-8")
    return (
        reference_path,
        geometry_record_path,
        attributes_path,
        attributes_record_path,
        crosswalk_path,
        crosswalk_record_path,
        mask_root,
        references,
    )


def test_ds1_evaluator_package_is_private_and_checksum_locked(tmp_path, monkeypatch):
    contract = _small_contract()
    monkeypatch.setattr(ds1_evaluator_package, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    (
        reference,
        geometry_record,
        attributes,
        attributes_record,
        crosswalk,
        crosswalk_record,
        mask_root,
        _,
    ) = _inputs(tmp_path)
    output = tmp_path / "evaluator"
    record = ds1_evaluator_package.build_ds1_evaluator_package(
        reference_manifest_path=reference,
        geometry_package_record_path=geometry_record,
        protected_attributes_path=attributes,
        attributes_package_record_path=attributes_record,
        evaluator_crosswalk_path=crosswalk,
        crosswalk_package_record_path=crosswalk_record,
        source_mask_root=mask_root,
        output_root=output,
        provenance={"snapshot_sha256": "a" * 64},
    )
    assert record["image_count"] == 4
    assert record["model_facing_allowed"] is False
    assert record["crosswalk_package_record_sha256"] == file_sha256(
        output / "crosswalk_package_record.json"
    )
    copied_crosswalk_record = json.loads(
        (output / "crosswalk_package_record.json").read_text(encoding="utf-8")
    )
    assert copied_crosswalk_record["prompt_manifest_sha256"] == "5" * 64
    assert copied_crosswalk_record["model_input_manifest_sha256"] == "6" * 64
    assert record["geometry_package_record_sha256"] == file_sha256(geometry_record)
    assert record["attributes_package_record_sha256"] == file_sha256(
        attributes_record
    )
    assert record["protected_attributes_sha256"] == file_sha256(attributes)
    assert ds1_evaluator_package.verify_ds1_evaluator_package(output)["all_checks_passed"]
    try:
        ds1_evaluator_package.verify_ds1_evaluator_package(output, intended_use="model_facing")
    except ValueError as error:
        assert "cannot be used" in str(error)
    else:
        raise AssertionError("Evaluator package was accepted as model input")


def test_ds1_evaluator_builder_rejects_tampered_geometry_source(
    tmp_path, monkeypatch
):
    contract = _small_contract()
    monkeypatch.setattr(ds1_evaluator_package, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    (
        reference,
        geometry_record,
        attributes,
        attributes_record,
        crosswalk,
        crosswalk_record,
        mask_root,
        _,
    ) = _inputs(tmp_path)
    rows = [json.loads(line) for line in reference.read_text().splitlines()]
    rows[0]["box_xyxy"] = [41, 50, 120, 150]
    _write_jsonl(reference, rows)

    with pytest.raises(ValueError, match="geometry"):
        ds1_evaluator_package.build_ds1_evaluator_package(
            reference_manifest_path=reference,
            geometry_package_record_path=geometry_record,
            protected_attributes_path=attributes,
            attributes_package_record_path=attributes_record,
            evaluator_crosswalk_path=crosswalk,
            crosswalk_package_record_path=crosswalk_record,
            source_mask_root=mask_root,
            output_root=tmp_path / "evaluator",
            provenance={"snapshot_sha256": "a" * 64},
        )


def test_ds1_evaluator_builder_rejects_tampered_attribute_source(
    tmp_path, monkeypatch
):
    contract = _small_contract()
    monkeypatch.setattr(ds1_evaluator_package, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    (
        reference,
        geometry_record,
        attributes,
        attributes_record,
        crosswalk,
        crosswalk_record,
        mask_root,
        _,
    ) = _inputs(tmp_path)
    rows = [json.loads(line) for line in attributes.read_text().splitlines()]
    rows[0]["category"] = "altered_category"
    _write_jsonl(attributes, rows)

    with pytest.raises(ValueError, match="attribute"):
        ds1_evaluator_package.build_ds1_evaluator_package(
            reference_manifest_path=reference,
            geometry_package_record_path=geometry_record,
            protected_attributes_path=attributes,
            attributes_package_record_path=attributes_record,
            evaluator_crosswalk_path=crosswalk,
            crosswalk_package_record_path=crosswalk_record,
            source_mask_root=mask_root,
            output_root=tmp_path / "evaluator",
            provenance={"snapshot_sha256": "a" * 64},
        )


def test_ds1_evaluator_builder_rejects_package_record_from_another_root(
    tmp_path, monkeypatch
):
    contract = _small_contract()
    monkeypatch.setattr(ds1_evaluator_package, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    (
        reference,
        geometry_record,
        attributes,
        attributes_record,
        crosswalk,
        crosswalk_record,
        mask_root,
        _,
    ) = _inputs(tmp_path)
    other_root = tmp_path / "other_geometry"
    other_root.mkdir()
    other_record = other_root / "package_record.json"
    other_record.write_bytes(geometry_record.read_bytes())

    with pytest.raises(ValueError, match="canonical geometry package"):
        ds1_evaluator_package.build_ds1_evaluator_package(
            reference_manifest_path=reference,
            geometry_package_record_path=other_record,
            protected_attributes_path=attributes,
            attributes_package_record_path=attributes_record,
            evaluator_crosswalk_path=crosswalk,
            crosswalk_package_record_path=crosswalk_record,
            source_mask_root=mask_root,
            output_root=tmp_path / "evaluator",
            provenance={"snapshot_sha256": "a" * 64},
        )


def test_ds1_evaluator_builder_rejects_masks_outside_geometry_package(
    tmp_path, monkeypatch
):
    contract = _small_contract()
    monkeypatch.setattr(ds1_evaluator_package, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    (
        reference,
        geometry_record,
        attributes,
        attributes_record,
        crosswalk,
        crosswalk_record,
        mask_root,
        _,
    ) = _inputs(tmp_path)
    other_mask_root = tmp_path / "other_masks"
    shutil.copytree(mask_root, other_mask_root)

    with pytest.raises(ValueError, match="canonical geometry package"):
        ds1_evaluator_package.build_ds1_evaluator_package(
            reference_manifest_path=reference,
            geometry_package_record_path=geometry_record,
            protected_attributes_path=attributes,
            attributes_package_record_path=attributes_record,
            evaluator_crosswalk_path=crosswalk,
            crosswalk_package_record_path=crosswalk_record,
            source_mask_root=other_mask_root,
            output_root=tmp_path / "evaluator",
            provenance={"snapshot_sha256": "a" * 64},
        )


def test_ds1_evaluator_builder_rejects_crosswalk_geometry_identity_mismatch(
    tmp_path, monkeypatch
):
    contract = _small_contract()
    monkeypatch.setattr(ds1_evaluator_package, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    (
        reference,
        geometry_record,
        attributes,
        attributes_record,
        crosswalk,
        crosswalk_record,
        mask_root,
        _,
    ) = _inputs(tmp_path)
    record = json.loads(crosswalk_record.read_text())
    record["geometry_package_record_sha256"] = "f" * 64
    crosswalk_record.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical package chain"):
        ds1_evaluator_package.build_ds1_evaluator_package(
            reference_manifest_path=reference,
            geometry_package_record_path=geometry_record,
            protected_attributes_path=attributes,
            attributes_package_record_path=attributes_record,
            evaluator_crosswalk_path=crosswalk,
            crosswalk_package_record_path=crosswalk_record,
            source_mask_root=mask_root,
            output_root=tmp_path / "evaluator",
            provenance={"snapshot_sha256": "a" * 64},
        )


def test_ds1_workflow_reports_metrics_and_builds_label_free_routes(tmp_path, monkeypatch):
    contract = _small_contract()
    monkeypatch.setattr(ds1_evaluator_package, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(mask_evaluation_ds1, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(mask_evaluation_ds1, "SUBGROUP_MINIMUM_CLUSTERS", 1)
    monkeypatch.setattr(mask_evaluation_ds1, "SUBGROUP_MINIMUM_SCENES", 1)
    monkeypatch.setattr(mask_evaluation_ds1, "SUBGROUP_MINIMUM_SUPPORT", 1)
    monkeypatch.setattr(ds1_mask_route_package, "DS1_SCREEN_CONTRACT", contract)
    (
        reference,
        geometry_record,
        attributes,
        attributes_record,
        crosswalk,
        crosswalk_record,
        mask_root,
        reference_rows,
    ) = _inputs(tmp_path)
    evaluator = tmp_path / "evaluator"
    ds1_evaluator_package.build_ds1_evaluator_package(
        reference_manifest_path=reference,
        geometry_package_record_path=geometry_record,
        protected_attributes_path=attributes,
        attributes_package_record_path=attributes_record,
        evaluator_crosswalk_path=crosswalk,
        crosswalk_package_record_path=crosswalk_record,
        source_mask_root=mask_root,
        output_root=evaluator,
        provenance={"snapshot_sha256": "b" * 64},
    )

    proposal_rows = []
    sam_rows = []
    box = [40, 50, 120, 150]
    mask = np.zeros((256, 256), dtype=bool)
    mask[50:150, 40:120] = True
    encoded = encode_binary_mask(mask)
    for row in reference_rows:
        scene, view = row["opaque_scene_id"], row["view_id"]
        image_sha256 = sha256(f"{scene}/{view}".encode()).hexdigest()
        proposal_rows.append({
            "opaque_scene_id": scene,
            "view_id": view,
            "image_sha256": image_sha256,
            "status": "ok",
            "proposal_count": 1,
            "boxes_xyxy": [box],
            "runtime_seconds": 0.2,
        })
        item = {
            "box_xyxy": box,
            "foreground_pixels": int(mask.sum()),
            "mask_status": "non_empty",
            "mask_sha256": binary_mask_sha256(mask),
            "mask_rle": encoded,
            "predicted_quality_score": 0.99,
        }
        sam_rows.append({
            "opaque_scene_id": scene,
            "view_id": view,
            "image_sha256": image_sha256,
            "status": "ok",
            "total_runtime_seconds": 0.3,
            "routes": {
                "reference_box": [{**item, "prompt_key": "R000"}],
                "automatic_box": [{**item, "prompt_key": "P000"}],
            },
        })
    proposal_root = tmp_path / "proposals"
    sam_root = tmp_path / "sam"
    proposal_history_rows = [
        {
            **proposal_rows[0],
            "status": "error",
            "proposal_count": 0,
            "boxes_xyxy": [],
            "runtime_seconds": 99.0,
        },
        *proposal_rows,
    ]
    _write_jsonl(
        proposal_root / "private" / "proposal_results.jsonl",
        proposal_history_rows,
    )
    _write_jsonl(sam_root / "private" / "mask_results.jsonl", sam_rows)
    monkeypatch.setattr(
        mask_evaluation_ds1,
        "_verify_upstream_identity",
        lambda *args, **kwargs: (
            {
                "proposal_run_record_sha256": "5" * 64,
                "proposal_artifact_manifest_sha256": "6" * 64,
                "sam_run_record_sha256": "7" * 64,
                "sam_artifact_manifest_sha256": "8" * 64,
                "prompt_package_record_sha256": "1" * 64,
                "prompt_manifest_sha256": "5" * 64,
                "model_input_manifest_sha256": "9" * 64,
                "proposal_configuration_sha256": "a" * 64,
                "sam_configuration_sha256": "b" * 64,
            },
            proposal_rows,
        ),
    )
    output = tmp_path / "evaluation"
    record = mask_evaluation_ds1.run_ds1_mask_evaluation(
        evaluator_root=evaluator,
        proposal_output_root=proposal_root,
        sam_output_root=sam_root,
        output_root=output,
        evaluator_provenance={"snapshot_sha256": "c" * 64},
        bootstrap_repetitions=20,
        bootstrap_seed=7,
    )
    assert record["component_decision"]["status"] == "proceed_with_locateanything_and_sam21"
    assert record["coverage"]["proposal_images"] == 4
    assert record["proposal_summary"]["image_count"] == 4
    assert record["runtime"]["proposal_total_seconds"] == pytest.approx(0.8)
    assert (output / "safe" / "proposal_subgroups.csv").is_file()
    assert (output / "safe" / "mask_bootstrap_intervals.csv").is_file()
    model_text = (output / "model_input" / "mask_routes.jsonl").read_text(encoding="utf-8").lower()
    for protected in ("category", "anomaly", "asset", "object_id", "source"):
        assert protected not in model_text
    verified = ds1_mask_route_package.verify_ds1_mask_route_package(output / "model_input")
    assert set(verified["route_counts"]) == {"supplied_mask", "reference_box", "automatic_box"}
    route_rows = [
        json.loads(line)
        for line in (output / "model_input" / "mask_routes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for route_row in route_rows:
        supplied_keys = {item["peer_key"] for item in route_row["routes"]["supplied_mask"]}
        reference_keys = {item["peer_key"] for item in route_row["routes"]["reference_box"]}
        automatic_keys = {
            item["peer_key"]
            for item in route_row["routes"]["automatic_box"]
            if item["peer_key"] is not None
        }
        assert supplied_keys == reference_keys
        assert automatic_keys <= reference_keys


def test_subgroup_iou_is_detection_aware_for_missed_references(monkeypatch):
    monkeypatch.setattr(mask_evaluation_ds1, "SUBGROUP_MINIMUM_CLUSTERS", 1)
    monkeypatch.setattr(mask_evaluation_ds1, "SUBGROUP_MINIMUM_SCENES", 1)
    monkeypatch.setattr(mask_evaluation_ds1, "SUBGROUP_MINIMUM_SUPPORT", 1)
    proposal_records = [{
        "opaque_scene_id": "S000001",
        "opaque_asset_group_id": "A001",
        "view_id": "V00",
        "category": "category_1",
        "object_count": 1,
        "anomaly_families": ["size"],
        "matched_at_0_50": 1,
        "proposal_count": 1,
        "reference_count": 1,
    }]
    common = {
        "opaque_scene_id": "S000001",
        "opaque_asset_group_id": "A001",
        "view_id": "V00",
        "category": "category_1",
        "object_count": 1,
        "anomaly_family": ["size"],
    }
    mask_records = [
        {
            **common,
            "route": "reference_box",
            "record_kind": "matched_reference",
            "passes_mask_iou_threshold": 1,
            "mask_iou": 1.0,
        },
        {
            **common,
            "route": "automatic_box",
            "record_kind": "missed_reference",
            "passes_mask_iou_threshold": 0,
            "mask_iou": 0.0,
        },
    ]
    intervals, _, _ = mask_evaluation_ds1._subgroup_bootstrap_evidence(
        proposal_records,
        mask_records,
        repetitions=20,
        seed=7,
    )
    automatic_iou = [
        row
        for row in intervals
        if row["component"] == "automatic_box"
        and row["metric"] == "detection_aware_mean_iou"
        and row["dimension"] == "view_id"
    ][0]
    assert automatic_iou["point_estimate"] == 0.0
    assert automatic_iou["ci_lower_95"] == 0.0


def test_mask_route_contract_rejects_unknown_automatic_peer(monkeypatch):
    contract = _small_contract()
    monkeypatch.setattr(ds1_mask_route_package, "DS1_SCREEN_CONTRACT", contract)
    mask = np.zeros((256, 256), dtype=bool)
    mask[50:150, 40:120] = True
    item = {
        "box_xyxy": [40, 50, 120, 150],
        "foreground_pixels": int(mask.sum()),
        "mask_status": "non_empty",
        "mask_sha256": binary_mask_sha256(mask),
        "mask_rle": encode_binary_mask(mask),
    }
    rows = []
    for scene in ("S000001", "S000002"):
        for view in ("V00", "V01"):
            rows.append({
                "schema_version": 1,
                "opaque_scene_id": scene,
                "view_id": view,
                "image_sha256": sha256(f"{scene}/{view}".encode()).hexdigest(),
                "routes": {
                    "supplied_mask": [{**item, "prompt_key": "R000", "peer_key": "R000"}],
                    "reference_box": [{**item, "prompt_key": "R000", "peer_key": "R000"}],
                    "automatic_box": [{**item, "prompt_key": "P000", "peer_key": "R999"}],
                },
            })
    try:
        ds1_mask_route_package._validate(rows)
    except ValueError as error:
        assert "unknown peer" in str(error)
    else:
        raise AssertionError("Unknown automatic peer identity was accepted")


def test_upstream_identity_rejects_another_rgb_manifest(tmp_path, monkeypatch):
    contract = _small_contract()
    monkeypatch.setattr(mask_evaluation_ds1, "DS1_SCREEN_CONTRACT", contract)
    cohort = {
        key: getattr(contract, key)
        for key in (
            "cohort_id",
            "release_id",
            "role",
            "pilot_stage",
            "membership_sha256",
            "scene_count",
            "views_per_scene",
            "image_count",
            "asset_group_count",
        )
    }
    proposal_root = tmp_path / "proposal"
    sam_root = tmp_path / "sam"
    evaluator_root = tmp_path / "evaluator"
    for root in (proposal_root, sam_root):
        (root / "safe").mkdir(parents=True)
        (root / "private").mkdir()
    evaluator_root.mkdir()
    proposal_rows = []
    sam_rows = []
    for scene in ("S000001", "S000002"):
        for view in ("V00", "V01"):
            proposal_rows.append({
                "opaque_scene_id": scene,
                "view_id": view,
                "status": "ok",
                "producer_kind": mask_evaluation_ds1.LOCATEANYTHING_MODEL_PRODUCER_KIND,
                "configuration_sha256": "d" * 64,
                "input_manifest_sha256": "a" * 64,
            })
            sam_rows.append({
                "opaque_scene_id": scene,
                "view_id": view,
                "status": "ok",
                "producer_kind": mask_evaluation_ds1.SAM_MODEL_PRODUCER_KIND,
                "cohort_id": contract.cohort_id,
                "configuration_sha256": "e" * 64,
                "prompt_package_sha256": "b" * 64,
                "routes": {"reference_box": [], "automatic_box": []},
            })
    proposal_path = proposal_root / "private" / "proposal_results.jsonl"
    sam_path = sam_root / "private" / "mask_results.jsonl"
    _write_jsonl(proposal_path, proposal_rows)
    _write_jsonl(sam_path, sam_rows)
    proposal_run = {
        "stage": "ds1_locateanything_screen",
        "status": "completed",
        "run_mode": "model_inference",
        "inference_stage": "full",
        "producer_kind": mask_evaluation_ds1.LOCATEANYTHING_MODEL_PRODUCER_KIND,
        "completion_gate_passed": True,
        "cohort": cohort,
        "configuration": {
            "model_id": "test-model",
            "model_revision": "test-revision",
        },
        "configuration_sha256": "d" * 64,
        "output_identity": {
            "result_filename": "proposal_results.jsonl",
            "result_sha256": file_sha256(proposal_path),
            "proposal_results_sha256": file_sha256(proposal_path),
            "result_count": len(proposal_rows),
        },
    }
    sam_run = {
        "stage": "ds1_sam21_prompted_masks",
        "status": "completed",
        "run_mode": "model_inference",
        "inference_stage": "full",
        "producer_kind": mask_evaluation_ds1.SAM_MODEL_PRODUCER_KIND,
        "completion_gate_passed": True,
        "cohort": cohort,
        "configuration_sha256": "e" * 64,
        "prompt_package_sha256": "b" * 64,
        "input_identity": {
            "prompt_package_record_sha256": "b" * 64,
            "prompt_manifest_sha256": "c" * 64,
        },
        "output_identity": {
            "mask_results_sha256": file_sha256(sam_path),
            "mask_result_count": len(sam_rows),
        },
    }
    (proposal_root / "safe" / "run_record.json").write_text(
        json.dumps(proposal_run), encoding="utf-8"
    )
    (sam_root / "safe" / "run_record.json").write_text(
        json.dumps(sam_run), encoding="utf-8"
    )
    for root in (proposal_root, sam_root):
        _write_artifact_manifest(root / "safe")
        _write_artifact_manifest(root / "private")
    (evaluator_root / "package_record.json").write_text(
        json.dumps({"proposal_results_sha256": file_sha256(proposal_path)}),
        encoding="utf-8",
    )
    crosswalk_record = {
        "proposal_results_sha256": file_sha256(proposal_path),
        "prompt_package_record_sha256": "b" * 64,
        "prompt_manifest_sha256": "c" * 64,
        "model_input_manifest_sha256": "a" * 64,
    }
    crosswalk_record_path = evaluator_root / "crosswalk_package_record.json"
    crosswalk_record_path.write_text(json.dumps(crosswalk_record), encoding="utf-8")
    class _Resolved:
        latest_rows = tuple(proposal_rows)

    def resolve(*args, **kwargs):
        if kwargs["expected_input_manifest_sha256"] != "a" * 64:
            raise ValueError("Proposal history identifies another model-input manifest")
        return _Resolved()

    monkeypatch.setattr(
        mask_evaluation_ds1, "resolve_committed_proposal_history", resolve
    )
    verified, latest = mask_evaluation_ds1._verify_upstream_identity(
        evaluator_root,
        proposal_root,
        sam_root,
        proposal_path,
        sam_path,
        sam_rows,
    )
    assert verified["model_input_manifest_sha256"] == "a" * 64
    assert latest == proposal_rows
    crosswalk_record["model_input_manifest_sha256"] = "f" * 64
    crosswalk_record_path.write_text(json.dumps(crosswalk_record), encoding="utf-8")
    try:
        mask_evaluation_ds1._verify_upstream_identity(
            evaluator_root,
            proposal_root,
            sam_root,
            proposal_path,
            sam_path,
            sam_rows,
        )
    except ValueError as error:
        assert "model-input manifest" in str(error)
    else:
        raise AssertionError("Another RGB manifest was accepted")
