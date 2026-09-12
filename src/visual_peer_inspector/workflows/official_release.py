"""Create physically separated official evaluation resources."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Self

from PIL import Image

from visual_peer_inspector.data.archive_members import (
    IndexedMember,
    read_indexed_member,
)
from visual_peer_inspector.data.cohort_contract import (
    OFFICIAL_SEEN_TEST_CONTRACT,
    OFFICIAL_UNSEEN_TEST_CONTRACT,
    CohortContract,
)
from visual_peer_inspector.data.model_safe_package import (
    PackageAudit,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    verify_model_input_package,
)
from visual_peer_inspector.data.source_storage import (
    extracted_source_root,
    read_extracted_source_member,
    verify_archive_storage,
)
from visual_peer_inspector.data.toysad_selection import (
    MODEL_FACING_MEDIA_FIELDS,
    anomaly_prefixes,
    validate_model_facing_media_record,
)
from visual_peer_inspector.frozen_resource import verify_frozen_resource

SCHEMA_VERSION = "visual-peer-official-release/v1"
EVALUATOR_MANIFEST_SCHEMA = "visual-peer-official-evaluator-manifest/v1"
LOCK_RESULT_SCHEMA = "visual-peer-09d-lock-result/v1"
EXPECTED_ARCHIVE_SHA256 = (
    "e49c30857b96fdede3f374c773ce0758174f225958ba29c9ad5ca5b075dfc733"
)
EXPECTED_ARCHIVE_SIZE_BYTES = 52_993_751_003
OFFICIAL_CONTRACTS = {
    "seen": OFFICIAL_SEEN_TEST_CONTRACT,
    "unseen": OFFICIAL_UNSEEN_TEST_CONTRACT,
}


def verify_internal_lock_decision(path: Path) -> dict[str, Any]:
    """Require completed one-shot protocol evidence before official data is released."""

    source = Path(path).expanduser().resolve(strict=True)
    verify_frozen_resource(
        source.parent,
        expected_stage="internal_lock_decision_release",
        expected_files={"internal_lock_result_receipt.json"},
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != LOCK_RESULT_SCHEMA:
        raise ValueError("Official release requires the Notebook 09D decision schema")
    if value.get("status") != "protocol_complete":
        raise ValueError("Official release requires a completed Notebook 09D protocol")
    if value.get("decision_scope") != "evaluation_completed_once_without_method_or_policy_change":
        raise ValueError("Notebook 09D has an unexpected decision scope")
    if value.get("scientific_performance_accepted") is not False:
        raise ValueError("Notebook 09D must not claim scientific performance acceptance")
    if value.get("method_or_policy_changed") is not False:
        raise ValueError("Official release requires an unchanged locked method and policy")
    for field in (
        "method_record_sha256",
        "policy_record_sha256",
        "internal_lock_prediction_lock_sha256",
        "internal_lock_evaluation_manifest_sha256",
    ):
        if not _is_sha256(value.get(field)):
            raise ValueError(f"Notebook 09D decision is missing {field}")
    return {"path": source, "sha256": file_sha256(source), **value}


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _atomic_bytes(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing output differs: {path}")
        return False
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return True


def _atomic_json(path: Path, value: object) -> bool:
    return _atomic_bytes(path, canonical_json_bytes(value) + b"\n")


def _replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def official_opaque_scene_id(role: str, index: int) -> str:
    """Return a role-disjoint opaque scene identifier."""

    prefix = {"seen-test": "S1", "unseen-test": "S2"}[role]
    if index < 1:
        raise ValueError("Official scene indices begin at one")
    return f"{prefix}{index:06d}"


def verify_official_membership(
    master: Mapping[str, Any], contract: CohortContract
) -> list[dict[str, Any]]:
    """Verify one official split against its frozen source contract."""

    scenes = master.get("scenes")
    if not isinstance(scenes, list):
        raise TypeError("The master file has no scenes array")
    selected = sorted(
        (dict(row) for row in scenes if row.get("split") == contract.role),
        key=lambda row: str(row["scene_id"]),
    )
    scene_ids = [str(row["scene_id"]) for row in selected]
    if len(scene_ids) != contract.scene_count or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Official scene count or uniqueness does not match the contract")
    if membership_sha256(scene_ids) != contract.membership_sha256:
        raise ValueError("Official source membership does not match the frozen hash")
    assets = {str(row["base_asset_id"]) for row in selected}
    if len(assets) != contract.asset_group_count:
        raise ValueError("Official asset-group count does not match the contract")
    return selected


def estimate_official_model_bytes(
    *, master: Mapping[str, Any], archive_index: Path, contract: CohortContract
) -> int:
    """Return the indexed uncompressed RGB bytes needed by one split."""

    scenes = verify_official_membership(master, contract)
    names = [
        f"{scene['archive_members']['rgb_directory']}/{view_index:04d}.png"
        for scene in scenes
        for view_index in range(contract.views_per_scene)
    ]
    total = 0
    with sqlite3.connect(f"file:{Path(archive_index).resolve()}?mode=ro", uri=True) as database:
        for name in names:
            rows = database.execute(
                "SELECT uncompressed_size FROM members WHERE name = ?", (name,)
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(f"Expected one indexed official RGB member: {name}")
            total += int(rows[0][0])
    return total


class IndexedArchive(AbstractContextManager["IndexedArchive"]):
    """Read verified ZIP or extracted members with one index connection."""

    def __init__(self, archive_path: Path, index_path: Path) -> None:
        self.archive_path = Path(archive_path)
        self.index_path = Path(index_path)
        self.stream = None
        self.database = None
        self.extracted_root = None

    def __enter__(self) -> Self:
        if self.archive_path.is_file():
            self.stream = self.archive_path.open("rb")
        else:
            self.extracted_root = extracted_source_root(self.archive_path)
            if self.extracted_root is None:
                raise FileNotFoundError("No indexed source archive or extraction was found")
        self.database = sqlite3.connect(
            f"file:{self.index_path.resolve()}?mode=ro", uri=True
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self.database is not None:
            self.database.close()
        if self.stream is not None:
            self.stream.close()

    def read(self, name: str, maximum_bytes: int = 16 * 1024 * 1024) -> bytes:
        if self.database is None:
            raise RuntimeError("Archive reader is not open")
        rows = self.database.execute(
            "SELECT name, compressed_size, uncompressed_size, crc32, "
            "compression_method, flags, local_header_offset, is_directory, "
            "is_symlink, unsafe_reason FROM members WHERE name = ?",
            (name,),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(f"Expected one indexed archive member: {name}")
        row = rows[0]
        member = IndexedMember(
            name=str(row[0]),
            compressed_size=int(row[1]),
            uncompressed_size=int(row[2]),
            crc32=int(row[3]),
            compression_method=int(row[4]),
            flags=int(row[5]),
            local_header_offset=int(row[6]),
            is_directory=bool(row[7]),
            is_symlink=bool(row[8]),
            unsafe_reason=row[9],
        )
        if self.extracted_root is not None:
            return read_extracted_source_member(
                self.extracted_root, member, maximum_bytes=maximum_bytes,
            )
        return read_indexed_member(
            self.stream,
            member,
            maximum_uncompressed_bytes=maximum_bytes,
        )


def _verify_source_identity(
    source_root: Path, master_path: Path, *, verify_archive_hash: bool
) -> dict[str, Any]:
    checksum_path = source_root / "package-checksums.sha256"
    declared: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if line:
            digest, relative = line.split("  ", 1)
            declared[relative] = digest
    master_relative = master_path.relative_to(source_root).as_posix()
    if declared.get(master_relative) != file_sha256(master_path):
        raise ValueError("The official master file does not match its package checksum")
    index_digest = file_sha256(source_root / "archive_index.sqlite3")
    if declared.get("archive_index.sqlite3") != index_digest:
        raise ValueError("The source archive index checksum differs")
    storage = verify_archive_storage(
        source_root, declared, expected_sha256=EXPECTED_ARCHIVE_SHA256,
        expected_size=EXPECTED_ARCHIVE_SIZE_BYTES, verify_archive_hash=verify_archive_hash,
    )
    return {
        "master_sha256": file_sha256(master_path),
        **storage,
        "archive_index_sha256": index_digest,
    }


def _scene_records(
    scene: Mapping[str, Any], opaque_scene_id: str, contract: CohortContract,
    scene2d: Mapping[str, Any], scene3d: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    object_count = int(scene["object_count"])
    object_ids = [f"O{index:02d}" for index in range(object_count)]
    view_ids = [f"V{index:02d}" for index in range(contract.views_per_scene)]
    neutral = {
        "schema_version": 1,
        "release_id": contract.release_id,
        "opaque_scene_id": opaque_scene_id,
        "object_count": object_count,
        "object_ids": object_ids,
        "view_ids": view_ids,
        "media_root": f"media/{opaque_scene_id}",
    }
    validate_model_facing_media_record(
        neutral, expected_release_id=contract.release_id
    )
    if set(neutral) != set(MODEL_FACING_MEDIA_FIELDS):
        raise ValueError("Official model-facing row has an unexpected field")
    crosswalk = {
        "schema_version": "visual-peer-official-crosswalk/v1",
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "opaque_scene_id": opaque_scene_id,
        "source_scene_id": str(scene["scene_id"]),
        "source_split": contract.role,
        "category": str(scene["category"]),
        "base_asset_id": str(scene["base_asset_id"]),
        "object_id_map": {object_id: index for index, object_id in enumerate(object_ids)},
        "view_id_map": {view_id: f"{index:04d}.png" for index, view_id in enumerate(view_ids)},
    }
    flags = [bool(value) for value in scene["normalized_anomaly_flags"]]
    labels = {
        "schema_version": "visual-peer-official-labels/v1",
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "opaque_scene_id": opaque_scene_id,
        "object_labels": [
            {
                "object_id": object_id,
                "raw_label": int(scene["raw_anomaly_labels"][index]),
                "is_anomaly": flags[index],
            }
            for index, object_id in enumerate(object_ids)
        ],
        "anomaly_count": sum(flags),
        "anomaly_type": "+".join(anomaly_prefixes(scene)) or "not_recorded",
    }
    bbox = scene2d.get("bbox")
    objects3d = scene3d.get("objects")
    if not isinstance(bbox, dict) or not isinstance(objects3d, list):
        raise TypeError("Official evaluator metadata is incomplete")
    geometry = {
        "schema_version": "visual-peer-official-geometry/v1",
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "opaque_scene_id": opaque_scene_id,
        "views": [
            {
                "view_id": view_id,
                "source_filename": f"{index:04d}.png",
                "boxes_xyxy": bbox.get(f"{index:04d}.png", []),
            }
            for index, view_id in enumerate(view_ids)
        ],
        "objects_3d": [
            {
                "object_id": object_ids[index],
                "bbox_xyz": value.get("bbox"),
                "location_xyz": value.get("location"),
            }
            for index, value in enumerate(objects3d)
        ],
    }
    return neutral, crosswalk, labels, geometry


def _model_checksum_bytes(paths: Sequence[Path], root: Path) -> bytes:
    return "".join(
        f"{file_sha256(root / relative)}  {relative.as_posix()}\n"
        for relative in sorted(paths, key=lambda value: value.as_posix())
    ).encode("utf-8")


def _write_evaluator_manifest(
    evaluator_root: Path, contract: CohortContract
) -> dict[str, Any]:
    names = ("crosswalk_private.jsonl", "labels_private.jsonl", "geometry_private.jsonl")
    files = [
        {
            "path": name,
            "sha256": file_sha256(evaluator_root / name),
            "size_bytes": (evaluator_root / name).stat().st_size,
        }
        for name in names
    ]
    manifest = {
        "schema_version": EVALUATOR_MANIFEST_SCHEMA,
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "source_membership_sha256": contract.membership_sha256,
        "scene_count": contract.scene_count,
        "image_count": contract.image_count,
        "files": files,
    }
    _atomic_json(evaluator_root / "evaluator_manifest.json", manifest)
    return manifest


def build_official_resource(
    *, source_root: Path, master: Mapping[str, Any], contract: CohortContract,
    output_root: Path, checkpoint_every: int = 10,
    maximum_new_scenes: int | None = None,
) -> dict[str, Any]:
    """Build and verify one model resource and its separate evaluator resource."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    short = "seen" if contract.role == "seen-test" else "unseen"
    model_root = output_root / f"{short}-model-input"
    evaluator_root = output_root / f"{short}-evaluator-only"
    state_root = output_root / "trusted-release-state" / short
    scenes = verify_official_membership(master, contract)
    opaque_ids = [official_opaque_scene_id(contract.role, index) for index in range(1, len(scenes) + 1)]
    if membership_sha256(opaque_ids) != contract.model_membership_sha256:
        raise ValueError("Official opaque membership does not match the model contract")
    input_signature = sha256(canonical_json_bytes({
        "schema_version": SCHEMA_VERSION,
        "cohort_id": contract.cohort_id,
        "source_membership_sha256": contract.membership_sha256,
        "model_membership_sha256": contract.model_membership_sha256,
    })).hexdigest()
    state_root.mkdir(parents=True, exist_ok=True)
    if maximum_new_scenes is not None and maximum_new_scenes < 1:
        raise ValueError("maximum_new_scenes must be positive")
    processed = 0
    resumed = 0
    written = 0
    archive_path = source_root / "toysAD8K.zip"
    index_path = source_root / "archive_index.sqlite3"
    with IndexedArchive(archive_path, index_path) as archive:
        for index, (scene, opaque_id) in enumerate(zip(scenes, opaque_ids, strict=True), 1):
            state_path = state_root / f"{opaque_id}.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("input_signature") != input_signature:
                    raise ValueError("Official resume state identifies another input")
                for item in state["media"]:
                    target = model_root / item["path"]
                    if not target.is_file() or file_sha256(target) != item["sha256"]:
                        raise ValueError("Official resume media failed its checksum")
                resumed += 1
                continue
            if maximum_new_scenes is not None and processed >= maximum_new_scenes:
                break
            member_root = str(scene["archive_members"]["rgb_directory"])
            media: list[dict[str, Any]] = []
            for view_index in range(contract.views_per_scene):
                data = archive.read(f"{member_root}/{view_index:04d}.png")
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    if image.format != "PNG" or image.size != (256, 256) or image.mode not in {"RGB", "RGBA"}:
                        raise ValueError("Official RGB media has an unexpected format")
                    if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
                        raise ValueError("Official RGB media contains transparency")
                    media_mode = image.mode
                relative = Path(f"media/{opaque_id}/rgb/V{view_index:02d}.png")
                if _atomic_bytes(model_root / relative, data):
                    written += 1
                media.append({"path": relative.as_posix(), "sha256": sha256(data).hexdigest(), "size_bytes": len(data), "media_mode": media_mode})
            scene2d = json.loads(archive.read(str(scene["archive_members"]["scene2d_metadata"])))
            scene3d = json.loads(archive.read(str(scene["archive_members"]["scene3d_metadata"])))
            neutral, crosswalk, labels, geometry = _scene_records(
                scene, opaque_id, contract, scene2d, scene3d
            )
            _atomic_json(state_path, {
                "schema_version": SCHEMA_VERSION,
                "input_signature": input_signature,
                "neutral": neutral,
                "crosswalk": crosswalk,
                "labels": labels,
                "geometry": geometry,
                "media": media,
            })
            processed += 1
            if index % checkpoint_every == 0:
                _replace_json(state_root / "checkpoint.json", {
                    "schema_version": SCHEMA_VERSION,
                    "input_signature": input_signature,
                    "completed_scene_count": index,
                    "exact_resume": True,
                })
    completed_ids = [
        opaque_id
        for opaque_id in opaque_ids
        if (state_root / f"{opaque_id}.json").is_file()
    ]
    if len(completed_ids) != len(opaque_ids):
        _replace_json(
            state_root / "checkpoint.json",
            {
                "schema_version": SCHEMA_VERSION,
                "input_signature": input_signature,
                "completed_scene_count": len(completed_ids),
                "remaining_scene_count": len(opaque_ids) - len(completed_ids),
                "exact_resume": True,
            },
        )
        return {
            "cohort_id": contract.cohort_id,
            "source_membership_sha256": contract.membership_sha256,
            "model_membership_sha256": contract.model_membership_sha256,
            "processed_scene_count": processed,
            "resumed_scene_count": resumed,
            "completed_scene_count": len(completed_ids),
            "remaining_scene_count": len(opaque_ids) - len(completed_ids),
            "written_image_count": written,
            "all_checks_passed": False,
            "status": "checkpointed",
        }
    states = [json.loads((state_root / f"{opaque_id}.json").read_text(encoding="utf-8")) for opaque_id in opaque_ids]
    _atomic_bytes(model_root / "neutral_corpus_manifest.jsonl", _jsonl_bytes(item["neutral"] for item in states))
    _atomic_bytes(evaluator_root / "crosswalk_private.jsonl", _jsonl_bytes(item["crosswalk"] for item in states))
    _atomic_bytes(evaluator_root / "labels_private.jsonl", _jsonl_bytes(item["labels"] for item in states))
    _atomic_bytes(evaluator_root / "geometry_private.jsonl", _jsonl_bytes(item["geometry"] for item in states))
    media_paths = [Path(item["path"]) for state in states for item in state["media"]]
    content_paths = [Path("neutral_corpus_manifest.jsonl"), *media_paths]
    checksum_bytes = _model_checksum_bytes(content_paths, model_root)
    _atomic_bytes(model_root / "content_checksums.sha256", checksum_bytes)
    modes = Counter(str(item["media_mode"]) for state in states for item in state["media"])
    audit = PackageAudit(
        cohort_id=contract.cohort_id,
        release_id=contract.release_id,
        scene_count=contract.scene_count,
        view_count=contract.views_per_scene,
        image_count=len(media_paths),
        width=256,
        height=256,
        media_mode=next(iter(modes)) if len(modes) == 1 else "mixed",
        total_media_bytes=sum(int(item["size_bytes"]) for state in states for item in state["media"]),
        manifest_sha256=file_sha256(model_root / "neutral_corpus_manifest.jsonl"),
        membership_sha256=membership_sha256(opaque_ids),
        content_checksums_sha256=sha256(checksum_bytes).hexdigest(),
        verified_image_count=len(media_paths),
        duplicate_archive_member_count=0,
        duplicate_package_path_count=0,
        decoded_image_failure_count=0,
        all_checks_passed=True,
    )
    _atomic_json(model_root / "package_record.json", asdict(audit))
    verify_model_input_package(model_root, contract=contract)
    evaluator_manifest = _write_evaluator_manifest(evaluator_root, contract)
    return {
        "cohort_id": contract.cohort_id,
        "source_membership_sha256": contract.membership_sha256,
        "model_membership_sha256": contract.model_membership_sha256,
        "model_package_record_sha256": file_sha256(model_root / "package_record.json"),
        "model_content_checksums_sha256": file_sha256(model_root / "content_checksums.sha256"),
        "evaluator_manifest_sha256": file_sha256(evaluator_root / "evaluator_manifest.json"),
        "evaluator_manifest": evaluator_manifest,
        "processed_scene_count": processed,
        "resumed_scene_count": resumed,
        "written_image_count": written,
        "all_checks_passed": True,
        "status": "complete",
    }


def build_official_release(
    *, source_root: Path, master_path: Path, output_root: Path,
    lock_result_receipt: Path,
    roles: Sequence[str] = ("seen", "unseen"), verify_archive_hash: bool = True,
    checkpoint_every: int = 10,
    maximum_new_scenes: int | None = None,
) -> dict[str, Any]:
    """Build selected official resources and publish a safe aggregate receipt."""

    release_authorization = verify_internal_lock_decision(lock_result_receipt)
    source_root = Path(source_root).resolve()
    master_path = Path(master_path).resolve()
    output_root = Path(output_root).resolve()
    source_identity = _verify_source_identity(
        source_root, master_path, verify_archive_hash=verify_archive_hash
    )
    master = json.loads(master_path.read_text(encoding="utf-8"))
    results = [
        build_official_resource(
            source_root=source_root,
            master=master,
            contract=OFFICIAL_CONTRACTS[role],
            output_root=output_root,
            checkpoint_every=checkpoint_every,
            maximum_new_scenes=maximum_new_scenes,
        )
        for role in roles
    ]
    safe_root = output_root / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete"
            if all(result.get("status") == "complete" for result in results)
            else "checkpointed"
        ),
        "source_identity": source_identity,
        "release_authorization": {
            "status": release_authorization["status"],
            "lock_result_receipt_sha256": release_authorization["sha256"],
            "method_record_sha256": release_authorization["method_record_sha256"],
            "policy_record_sha256": release_authorization["policy_record_sha256"],
        },
        "resources": [
            {key: value for key, value in result.items() if key != "evaluator_manifest"}
            for result in results
        ],
        "physical_resources": [
            name
            for role in roles
            for name in (f"{role}-model-input", f"{role}-evaluator-only")
        ],
        "evaluator_is_model_input": False,
        "aggregate_only": True,
    }
    _atomic_json(safe_root / "official_release_receipt.json", receipt)
    return receipt


__all__ = [
    "EVALUATOR_MANIFEST_SCHEMA",
    "OFFICIAL_CONTRACTS",
    "SCHEMA_VERSION",
    "build_official_release",
    "build_official_resource",
    "estimate_official_model_bytes",
    "official_opaque_scene_id",
    "verify_internal_lock_decision",
    "verify_official_membership",
]
