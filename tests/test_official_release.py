from __future__ import annotations

import json
import sqlite3
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import membership_sha256
from visual_peer_inspector.frozen_resource import publish_frozen_resource
from visual_peer_inspector.workflows.official_release import (
    EVALUATOR_MANIFEST_SCHEMA,
    build_official_resource,
    official_opaque_scene_id,
    verify_internal_lock_decision,
    verify_official_membership,
)


def _write_lock_decision(tmp_path: Path, *, status: str = "protocol_complete") -> Path:
    source_root = tmp_path / "lock-source"
    source_root.mkdir()
    source = source_root / "internal_lock_result_receipt.json"
    digest = lambda value: sha256(value.encode()).hexdigest()
    source.write_text(
        json.dumps(
            {
                "schema_version": "visual-peer-09d-lock-result/v1",
                "status": status,
                "decision_scope": "evaluation_completed_once_without_method_or_policy_change",
                "scientific_performance_accepted": False,
                "method_record_sha256": digest("method"),
                "policy_record_sha256": digest("policy"),
                "internal_lock_prediction_lock_sha256": digest("prediction"),
                "internal_lock_evaluation_manifest_sha256": digest("evaluation"),
                "method_or_policy_changed": False,
            }
        ),
        encoding="utf-8",
    )
    resource = tmp_path / "lock-resource"
    publish_frozen_resource(
        resource_root=resource,
        stage="internal_lock_decision_release",
        files={"internal_lock_result_receipt.json": source},
        bindings={},
    )
    return resource / "internal_lock_result_receipt.json"


def test_official_release_requires_a_completed_internal_lock_protocol(tmp_path: Path) -> None:
    completed = verify_internal_lock_decision(_write_lock_decision(tmp_path))
    assert completed["status"] == "protocol_complete"

    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    with pytest.raises(ValueError, match="completed"):
        verify_internal_lock_decision(
            _write_lock_decision(rejected_root, status="rejected")
        )


def _fixture_source(tmp_path: Path, scene_count: int = 1) -> tuple[Path, dict]:
    root = tmp_path / "source"
    root.mkdir()
    scenes = []
    for number in range(1, scene_count + 1):
        scene_id = f"toy_{number:03d}-001"
        scenes.append({
            "scene_id": scene_id,
            "split": "seen-test",
            "category": "toy",
            "base_asset_id": f"toy_{number:03d}",
            "object_count": 3,
            "anomaly_count": 1,
            "anomaly_variants": ["colour-red"],
            "raw_anomaly_labels": [0, 1, 1],
            "normalized_anomaly_flags": [1, 0, 0],
            "archive_members": {
                "rgb_directory": f"toysad8k/{scene_id}/RGB",
                "scene2d_metadata": f"toysad8k/{scene_id}/scene2d.metadata.json",
                "scene3d_metadata": f"toysad8k/{scene_id}/scene3d.metadata.json",
            },
        })
    image_path = tmp_path / "image.png"
    Image.new("RGB", (256, 256), (40, 80, 120)).save(image_path)
    image_bytes = image_path.read_bytes()
    bbox = {
        f"{index:04d}.png": [
            [10, 20, 60, 80],
            [80, 20, 130, 80],
            [150, 20, 200, 80],
        ]
        for index in range(20)
    }
    scene2d = json.dumps({"bbox": bbox}).encode()
    scene3d = json.dumps({
        "objects": [
            {"bbox": [[0, 0, 0], [1, 1, 1]], "location": [index, 0, 0]}
            for index in range(3)
        ]
    }).encode()
    archive_path = root / "toysAD8K.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for scene in scenes:
            for index in range(20):
                archive.writestr(f"{scene['archive_members']['rgb_directory']}/{index:04d}.png", image_bytes)
            archive.writestr(scene["archive_members"]["scene2d_metadata"], scene2d)
            archive.writestr(scene["archive_members"]["scene3d_metadata"], scene3d)
    with zipfile.ZipFile(archive_path) as archive, sqlite3.connect(root / "archive_index.sqlite3") as database:
        database.execute(
            "CREATE TABLE members(name TEXT, compressed_size INTEGER, uncompressed_size INTEGER, "
            "crc32 INTEGER, compression_method INTEGER, flags INTEGER, local_header_offset INTEGER, "
            "is_directory INTEGER, is_symlink INTEGER, unsafe_reason TEXT)"
        )
        for info in archive.infolist():
            database.execute(
                "INSERT INTO members VALUES(?,?,?,?,?,?,?,?,?,?)",
                (info.filename, info.compress_size, info.file_size, info.CRC, info.compress_type, info.flag_bits, info.header_offset, 0, 0, None),
            )
    return root, {"scenes": scenes}


def _fixture_contract(scene_count: int = 1) -> CohortContract:
    source_hash = membership_sha256(
        [f"toy_{number:03d}-001" for number in range(1, scene_count + 1)]
    )
    model_hash = membership_sha256(
        [official_opaque_scene_id("seen-test", number) for number in range(1, scene_count + 1)]
    )
    return CohortContract(
        cohort_id="fixture-seen",
        release_id="toysad8k-official-evaluation-v1",
        role="seen-test",
        pilot_stage=None,
        membership_sha256=source_hash,
        scene_count=scene_count,
        views_per_scene=20,
        image_count=scene_count * 20,
        asset_group_count=scene_count,
        model_membership_sha256=model_hash,
    )


def test_official_release_is_opaque_separate_and_resumable(tmp_path: Path) -> None:
    source, master = _fixture_source(tmp_path)
    contract = _fixture_contract()
    output = tmp_path / "release"
    first = build_official_resource(
        source_root=source,
        master=master,
        contract=contract,
        output_root=output,
        checkpoint_every=1,
    )
    assert first["written_image_count"] == 20
    model_root = output / "seen-model-input"
    evaluator_root = output / "seen-evaluator-only"
    model_text = "\n".join(path.read_text(errors="ignore") for path in model_root.glob("*.json*"))
    assert "toy_001" not in model_text
    assert "anomaly" not in model_text.lower()
    manifest = json.loads((evaluator_root / "evaluator_manifest.json").read_text())
    assert manifest["schema_version"] == EVALUATOR_MANIFEST_SCHEMA
    assert {item["path"] for item in manifest["files"]} == {
        "crosswalk_private.jsonl",
        "labels_private.jsonl",
        "geometry_private.jsonl",
    }
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    label = json.loads(
        (evaluator_root / "labels_private.jsonl").read_text().splitlines()[0]
    )
    assert label["anomaly_type"] == "colour"
    second = build_official_resource(
        source_root=source,
        master=master,
        contract=contract,
        output_root=output,
        checkpoint_every=1,
    )
    assert second["resumed_scene_count"] == 1
    assert second["written_image_count"] == 0


def test_bounded_official_release_matches_uninterrupted_output(tmp_path: Path) -> None:
    source, master = _fixture_source(tmp_path, scene_count=2)
    contract = _fixture_contract(scene_count=2)
    bounded_root = tmp_path / "bounded"
    first = build_official_resource(
        source_root=source,
        master=master,
        contract=contract,
        output_root=bounded_root,
        checkpoint_every=1,
        maximum_new_scenes=1,
    )
    assert first["status"] == "checkpointed"
    assert first["completed_scene_count"] == 1
    assert not (bounded_root / "seen-model-input/package_record.json").exists()
    second = build_official_resource(
        source_root=source,
        master=master,
        contract=contract,
        output_root=bounded_root,
        checkpoint_every=1,
        maximum_new_scenes=1,
    )
    assert second["status"] == "complete"
    uninterrupted_root = tmp_path / "uninterrupted"
    complete = build_official_resource(
        source_root=source,
        master=master,
        contract=contract,
        output_root=uninterrupted_root,
        checkpoint_every=1,
    )
    assert complete["status"] == "complete"
    for relative in (
        "seen-model-input/package_record.json",
        "seen-model-input/content_checksums.sha256",
        "seen-evaluator-only/evaluator_manifest.json",
    ):
        assert (bounded_root / relative).read_bytes() == (
            uninterrupted_root / relative
        ).read_bytes()


def test_official_membership_rejects_changed_source() -> None:
    contract = _fixture_contract()
    master = {"scenes": [{"scene_id": "wrong", "split": "seen-test", "base_asset_id": "toy_001"}]}
    try:
        verify_official_membership(master, contract)
    except ValueError as error:
        assert "membership" in str(error).lower()
    else:
        raise AssertionError("Changed official membership was accepted")
