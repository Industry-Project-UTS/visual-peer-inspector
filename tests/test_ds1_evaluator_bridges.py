from __future__ import annotations

from binascii import crc32
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
from PIL import Image

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import canonical_json_bytes, file_sha256, membership_sha256
from visual_peer_inspector.evaluation import proposal_reference_ds1, sam_crosswalk_ds1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _contract() -> CohortContract:
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


def _archive_and_index(tmp_path: Path):
    archive = tmp_path / "source.zip"
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 40:120] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)
    data = mask_path.read_bytes()
    members = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for scene in ("S000001", "S000002"):
            for view in ("V00", "V01"):
                name = f"dataset/{scene}/segmentations/O01/{view}.png"
                handle.writestr(name, data)
                members.append((scene, view, name))
    index = tmp_path / "archive.sqlite"
    with zipfile.ZipFile(archive) as handle, sqlite3.connect(index) as connection:
        connection.execute(
            "CREATE TABLE members (name TEXT, compressed_size INTEGER, uncompressed_size INTEGER, crc32 INTEGER, compression_method INTEGER, flags INTEGER, local_header_offset INTEGER, is_directory INTEGER, is_symlink INTEGER, unsafe_reason TEXT)"
        )
        for info in handle.infolist():
            connection.execute(
                "INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (info.filename, info.compress_size, info.file_size, info.CRC, info.compress_type, info.flag_bits, info.header_offset, 0, 0, None),
            )
    return archive, index, members, data


def test_ds1_reference_builder_extracts_and_verifies_private_inputs(tmp_path, monkeypatch):
    contract = _contract()
    archive, index, members, mask_data = _archive_and_index(tmp_path)
    monkeypatch.setattr(proposal_reference_ds1, "DS1_SCREEN_CONTRACT", contract)
    monkeypatch.setattr(proposal_reference_ds1, "EXPECTED_ARCHIVE_SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(proposal_reference_ds1, "EXPECTED_ARCHIVE_SHA256", file_sha256(archive))
    model_rows = []
    sources = []
    labels = []
    for number, scene in enumerate(("S000001", "S000002"), 1):
        model_rows.append({"opaque_scene_id": scene, "view_ids": ["V00", "V01"], "object_ids": ["O01"], "object_count": 1})
        sources.append({"opaque_scene_id": scene, "opaque_asset_group_id": f"A{number:03d}", "category": f"category_{number}", "object_count": 1, "anomaly_count": 1, "raw_anomaly_prefixes": ["size"]})
        labels.append({"opaque_scene_id": scene, "labels_by_object_id": {"O01": 1}})
    required = [{
        "opaque_scene_id": scene,
        "view_id": view,
        "opaque_object_id": "O01",
        "component": "object_mask",
        "include_in_media_package": True,
        "source_member": name,
        "crc32": f"{crc32(mask_data) & 0xFFFFFFFF:08x}",
        "uncompressed_size": len(mask_data),
    } for scene, view, name in members]
    paths = [tmp_path / name for name in ("model.jsonl", "source.jsonl", "labels.jsonl", "required.jsonl")]
    for path, rows in zip(paths, (model_rows, sources, labels, required)):
        _write_jsonl(path, rows)
    geometry, attributes = tmp_path / "geometry", tmp_path / "attributes"
    record = proposal_reference_ds1.build_ds1_reference_packages(
        model_manifest_path=paths[0],
        development_source_map_path=paths[1],
        development_labels_path=paths[2],
        required_source_members_path=paths[3],
        source_archive_path=archive,
        archive_index_path=index,
        geometry_output_root=geometry,
        attributes_output_root=attributes,
    )
    assert record["geometry"]["object_view_count"] == 4
    assert proposal_reference_ds1.verify_ds1_geometry_package(geometry)["all_checks_passed"]
    assert proposal_reference_ds1.verify_ds1_attributes_package(attributes)["all_checks_passed"]
    assert "labels_by_object_id" in (attributes / "protected_scene_attributes.jsonl").read_text()


def test_ds1_crosswalk_uses_exact_prompt_inputs_and_deterministic_matches(tmp_path, monkeypatch):
    contract = _contract()
    monkeypatch.setattr(sam_crosswalk_ds1, "DS1_SCREEN_CONTRACT", contract)
    prompt_root = tmp_path / "prompts"
    geometry_root = tmp_path / "geometry"
    prompt_root.mkdir()
    geometry_root.mkdir()
    prompt_rows = []
    geometry_rows = []
    proposal_rows = []
    box = [40, 50, 120, 150]
    for scene in ("S000001", "S000002"):
        for view in ("V00", "V01"):
            prompt_rows.append({"opaque_scene_id": scene, "view_id": view, "image_sha256": sha256(f"{scene}/{view}".encode()).hexdigest(), "routes": {"reference_box": [{"prompt_key": "R000", "box_xyxy": box}], "automatic_box": [{"prompt_key": "P000", "box_xyxy": box}]}})
            geometry_rows.append({"opaque_scene_id": scene, "view_id": view, "opaque_object_id": "O01", "visible": True, "box_xyxy": box})
            proposal_rows.append({"opaque_scene_id": scene, "view_id": view, "status": "ok", "proposal_count": 1, "boxes_xyxy": [box]})
    _write_jsonl(prompt_root / "sam_prompt_manifest.jsonl", prompt_rows)
    _write_jsonl(geometry_root / "reference_manifest.jsonl", geometry_rows)
    _write_jsonl(tmp_path / "proposals.jsonl", proposal_rows)
    (prompt_root / "package_record.json").write_text("{}\n")
    (geometry_root / "package_record.json").write_text("{}\n")
    upstream = {"geometry_manifest_sha256": file_sha256(geometry_root / "reference_manifest.jsonl"), "proposal_results_sha256": file_sha256(tmp_path / "proposals.jsonl")}
    (prompt_root / "upstream_decision.json").write_text(json.dumps(upstream))
    monkeypatch.setattr(
        sam_crosswalk_ds1,
        "verify_ds1_sam_prompt_package",
        lambda root: {
            "reference_prompt_count": 4,
            "automatic_prompt_count": 4,
            "prompt_manifest_sha256": file_sha256(
                prompt_root / "sam_prompt_manifest.jsonl"
            ),
            "model_manifest_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(sam_crosswalk_ds1, "verify_ds1_geometry_package", lambda root: {"reference_manifest_sha256": upstream["geometry_manifest_sha256"]})
    output = tmp_path / "crosswalk"
    record = sam_crosswalk_ds1.build_ds1_sam_evaluator_crosswalk(
        prompt_root=prompt_root,
        geometry_root=geometry_root,
        proposal_results_path=tmp_path / "proposals.jsonl",
        output_root=output,
    )
    assert record["matched_automatic_prompt_count"] == 4
    rows = [json.loads(line) for line in (output / "evaluator_crosswalk.jsonl").read_text().splitlines()]
    assert all(row["opaque_object_id"] == "O01" for row in rows)
    assert sam_crosswalk_ds1.verify_ds1_sam_evaluator_crosswalk(output)["all_checks_passed"]
