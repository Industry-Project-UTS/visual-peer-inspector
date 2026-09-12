"""Build restartable primary-cohort resources at their protocol gates."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from visual_peer_inspector.data.cohort_contract import (
    CALIBRATION_CONTRACT,
    DEVELOPMENT_FULL_CONTRACT,
    INTERNAL_LOCK_CONTRACT,
    CohortContract,
)
from visual_peer_inspector.data.model_safe_package import (
    PackageAudit,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.data.source_storage import verify_archive_storage
from visual_peer_inspector.data.toysad_selection import (
    MODEL_FACING_MEDIA_FIELDS,
    anomaly_prefixes,
    validate_model_facing_media_record,
)
from visual_peer_inspector.evaluation.proposal_reference import (
    DEFAULT_MASK_THRESHOLD,
    REFERENCE_GEOMETRY_POLICY,
    _mask_geometry,
)
from visual_peer_inspector.evaluation.proposals import match_boxes
from visual_peer_inspector.workflows.cross_view_identity_freeze import (
    EVALUATOR_SCHEMA_VERSION,
    load_08e_predictions,
)
from visual_peer_inspector.workflows.official_release import IndexedArchive
from visual_peer_inspector.workflows.protected_cohort_inference import (
    load_method_freeze,
    load_policy_freeze,
)

SCHEMA_VERSION = "visual-peer-primary-resource-release/v1"
RAW_EVALUATOR_MANIFEST_SCHEMA = "visual-peer-official-evaluator-manifest/v1"
ROLE_CONTRACTS = {
    "calibration": CALIBRATION_CONTRACT,
    "internal-lock": INTERNAL_LOCK_CONTRACT,
}
ROLE_LABEL_FILES = {
    "calibration": "calibration_labels.jsonl",
    "internal-lock": "internal_lock_labels.jsonl",
}


def _digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def verify_release_source(source_root: Path) -> dict[str, Any]:
    """Bind the indexed source files to their complete package checksum record."""

    root = Path(source_root).expanduser().resolve(strict=True)
    checksum_path = root / "package-checksums.sha256"
    declared: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relative = line.split("  ", 1)
        declared[relative] = digest
    master = root / "toysad8k_scene_master.json"
    index = root / "archive_index.sqlite3"
    actual = {
        "toysad8k_scene_master.json": file_sha256(master),
        "archive_index.sqlite3": file_sha256(index),
    }
    for relative, digest in actual.items():
        if declared.get(relative) != digest:
            raise ValueError(f"Source package checksum differs: {relative}")
    storage = verify_archive_storage(root, declared)
    return {
        "package_checksums_sha256": file_sha256(checksum_path),
        "source_master_sha256": actual["toysad8k_scene_master.json"],
        "archive_index_sha256": actual["archive_index.sqlite3"],
        **storage,
    }


def _atomic_bytes(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError(f"Existing output differs: {path}")
        return False
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return True


def _replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _mask_box(content: bytes) -> list[int]:
    """Use the same foreground definition as the earlier mask evaluations."""
    box, _area = _mask_geometry(content, DEFAULT_MASK_THRESHOLD)
    return [0, 0, 0, 0] if box is None else box


def _role_rows(
    *, selection_root: Path, administrative_root: Path, contract: CohortContract
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    roles = read_jsonl(selection_root / "experiment_roles.jsonl")
    neutral = read_jsonl(selection_root / "neutral_corpus_manifest.jsonl")
    trusted = read_jsonl(administrative_root / "restricted_source_map.jsonl")
    selected_ids = {
        str(row["opaque_scene_id"])
        for row in roles
        if str(row.get("role")) == contract.role
    }
    if len(selected_ids) != contract.scene_count:
        raise ValueError("Protected role scene count differs from its frozen contract")
    if membership_sha256(selected_ids) != contract.membership_sha256:
        raise ValueError("Protected role membership differs from its frozen hash")
    neutral_rows = sorted(
        (dict(row) for row in neutral if str(row["opaque_scene_id"]) in selected_ids),
        key=lambda row: str(row["opaque_scene_id"]),
    )
    trusted_by_scene = {
        str(row["opaque_scene_id"]): dict(row)
        for row in trusted
        if str(row["opaque_scene_id"]) in selected_ids
    }
    if len(neutral_rows) != contract.scene_count or set(trusted_by_scene) != selected_ids:
        raise ValueError("Protected manifests do not cover the same frozen scenes")
    assets = {str(row["opaque_asset_group_id"]) for row in trusted_by_scene.values()}
    if len(assets) != contract.asset_group_count:
        raise ValueError("Protected asset groups differ from the frozen contract")
    return neutral_rows, trusted_by_scene, roles


def _required_by_scene(
    administrative_root: Path, selected_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    grouped = {scene_id: [] for scene_id in selected_ids}
    for row in read_jsonl(administrative_root / "required_source_members.jsonl"):
        scene_id = str(row.get("opaque_scene_id"))
        if scene_id in grouped and row.get("component") in {"rgb", "object_mask"}:
            grouped[scene_id].append(dict(row))
    if any(not rows for rows in grouped.values()):
        raise ValueError("Protected source-member manifest is incomplete")
    return grouped


def _safe_neutral(row: Mapping[str, Any], contract: CohortContract) -> dict[str, Any]:
    value = {key: row[key] for key in MODEL_FACING_MEDIA_FIELDS}
    validate_model_facing_media_record(value, expected_release_id=contract.release_id)
    return value


def _evaluator_records(
    *,
    neutral: Mapping[str, Any],
    trusted: Mapping[str, Any],
    label: Mapping[str, Any],
    member_rows: Sequence[Mapping[str, Any]],
    archive: IndexedArchive,
    contract: CohortContract,
    scene_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scene_id = str(neutral["opaque_scene_id"])
    object_ids = [str(value) for value in neutral["object_ids"]]
    view_ids = [str(value) for value in neutral["view_ids"]]
    label_map = {str(key): int(value) for key, value in label["labels_by_object_id"].items()}
    if set(label_map) != set(object_ids):
        raise ValueError("Protected labels and opaque object identifiers differ")
    mask_rows = [row for row in member_rows if row["component"] == "object_mask"]
    boxes: dict[str, dict[str, list[int]]] = {
        view_id: {} for view_id in view_ids
    }
    for row in mask_rows:
        view_id = str(row["view_id"])
        object_id = str(row["opaque_object_id"])
        if view_id not in boxes or object_id not in object_ids:
            raise ValueError("Protected mask row has an unexpected opaque key")
        content = archive.read(str(row["source_member"]))
        boxes[view_id][object_id] = _mask_box(content)
    if any(set(values) != set(object_ids) for values in boxes.values()):
        raise ValueError("Protected evaluator geometry is incomplete")
    crosswalk = {
        "schema_version": "visual-peer-official-crosswalk/v1",
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "opaque_scene_id": scene_id,
        "source_split": contract.role,
        "category": str(scene_metadata["category"]),
        "base_asset_id": str(trusted["opaque_asset_group_id"]),
        "object_id_map": {object_id: index for index, object_id in enumerate(object_ids)},
        "view_id_map": {view_id: f"{view_id}.png" for view_id in view_ids},
    }
    labels = {
        "schema_version": "visual-peer-official-labels/v1",
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "opaque_scene_id": scene_id,
        "object_labels": [
            {
                "object_id": object_id,
                "raw_label": label_map[object_id],
                "is_anomaly": bool(label_map[object_id]),
            }
            for object_id in object_ids
        ],
        "anomaly_count": sum(bool(value) for value in label_map.values()),
        "anomaly_type": "+".join(anomaly_prefixes(scene_metadata)),
    }
    geometry = {
        "schema_version": "visual-peer-official-geometry/v1",
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "opaque_scene_id": scene_id,
        "views": [
            {
                "view_id": view_id,
                "source_filename": f"{view_id}.png",
                "boxes_xyxy": [boxes[view_id][object_id] for object_id in object_ids],
            }
            for view_id in view_ids
        ],
        "objects_3d": [],
    }
    return crosswalk, labels, geometry


def _model_checksums(root: Path, paths: Sequence[Path]) -> bytes:
    return "".join(
        f"{file_sha256(root / relative)}  {relative.as_posix()}\n"
        for relative in sorted(paths, key=lambda item: item.as_posix())
    ).encode()


def _evaluator_manifest(
    root: Path, contract: CohortContract, *, release_binding_sha256: str
) -> dict[str, Any]:
    names = ("crosswalk_private.jsonl", "labels_private.jsonl", "geometry_private.jsonl")
    files = [
        {
            "path": name,
            "sha256": file_sha256(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in names
    ]
    value = {
        "schema_version": RAW_EVALUATOR_MANIFEST_SCHEMA,
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "source_membership_sha256": contract.membership_sha256,
        "scene_count": contract.scene_count,
        "image_count": contract.image_count,
        "asset_group_count": contract.asset_group_count,
        "release_binding_sha256": release_binding_sha256,
        "files": files,
    }
    _atomic_bytes(root / "evaluator_manifest.json", canonical_json_bytes(value) + b"\n")
    return value


def build_protected_role_resources(
    *,
    role: str,
    selection_root: Path,
    administrative_root: Path,
    label_root: Path,
    source_root: Path,
    prerequisite_lock: Path,
    output_root: Path,
    method_lock: Path | None = None,
    maximum_new_scenes: int | None = None,
) -> dict[str, Any]:
    """Build model and raw evaluator resources for one protected role."""

    contract = ROLE_CONTRACTS[role]
    if maximum_new_scenes is not None and maximum_new_scenes < 1:
        raise ValueError("maximum_new_scenes must be positive")
    prerequisite_path = Path(prerequisite_lock).expanduser().resolve(strict=True)
    if role == "calibration":
        method_path = prerequisite_path
        method = load_method_freeze(
            method_path,
            application_contract_path=method_path.parent / "application_output_contract.json",
            require_resource=True,
        )
        method_sha256 = str(method["sha256"])
        policy_sha256 = None
    else:
        if method_lock is None:
            raise ValueError("Internal-lock release requires the frozen method resource")
        method_path = Path(method_lock).expanduser().resolve(strict=True)
        method = load_method_freeze(
            method_path,
            application_contract_path=method_path.parent
            / "application_output_contract.json",
            require_resource=True,
        )
        policy = load_policy_freeze(
            prerequisite_path,
            method_sha256=str(method["sha256"]),
            require_resource=True,
        )
        method_sha256 = str(method["sha256"])
        policy_sha256 = str(policy["sha256"])
    source_identity = verify_release_source(Path(source_root))
    neutral_rows, trusted_by_scene, _roles = _role_rows(
        selection_root=Path(selection_root),
        administrative_root=Path(administrative_root),
        contract=contract,
    )
    selected_ids = {str(row["opaque_scene_id"]) for row in neutral_rows}
    source_rows = _required_by_scene(Path(administrative_root), selected_ids)
    labels = {
        str(row["opaque_scene_id"]): row
        for row in read_jsonl(Path(label_root) / ROLE_LABEL_FILES[role])
    }
    if set(labels) != selected_ids:
        raise ValueError("Protected evaluator labels do not match the frozen role")
    master_path = Path(source_root) / "toysad8k_scene_master.json"
    master = json.loads(master_path.read_text(encoding="utf-8"))
    source_scenes = {
        str(row["scene_id"]): row for row in master.get("scenes", [])
    }
    metadata_by_opaque = {
        scene_id: source_scenes[str(trusted_by_scene[scene_id]["source_scene_id"])]
        for scene_id in selected_ids
    }
    output_root = Path(output_root).resolve()
    short = "calibration" if role == "calibration" else "internal-lock"
    model_root = output_root / f"{short}-model-input"
    evaluator_root = output_root / f"{short}-evaluator-only"
    state_root = output_root / "trusted-release-state" / short
    state_root.mkdir(parents=True, exist_ok=True)
    signature = _digest(
        {
            "schema_version": SCHEMA_VERSION,
            "cohort_id": contract.cohort_id,
            "membership_sha256": contract.membership_sha256,
            "selection_sha256": file_sha256(Path(selection_root) / "checksums.sha256"),
            "administrative_sha256": file_sha256(Path(administrative_root) / "checksums.sha256"),
            "label_vault_sha256": file_sha256(Path(label_root) / "checksums.sha256"),
            "source_identity": source_identity,
            "reference_geometry_policy": REFERENCE_GEOMETRY_POLICY,
            "method_record_sha256": method_sha256,
            "method_application_contract_sha256": file_sha256(
                method_path.parent / "application_output_contract.json"
            ),
            "method_resource_manifest_sha256": file_sha256(
                method_path.parent / "artifact_manifest.json"
            ),
            "method_identity_configuration_sha256": str(
                method["identity_configuration_sha256"]
            ),
            "method_frozen_model_identity_sha256": str(
                method["frozen_model_identity_sha256"]
            ),
            "policy_record_sha256": policy_sha256,
            "policy_resource_manifest_sha256": (
                None
                if role == "calibration"
                else file_sha256(prerequisite_path.parent / "artifact_manifest.json")
            ),
            "policy_configuration_sha256": (
                None if role == "calibration" else str(policy["policy_configuration_sha256"])
            ),
        }
    )
    processed = resumed = written = 0
    with IndexedArchive(
        Path(source_root) / "toysAD8K.zip",
        Path(source_root) / "archive_index.sqlite3",
    ) as archive:
        for index, neutral in enumerate(neutral_rows):
            if index % 25 == 0:
                print(json.dumps({"event": "resource_progress", "checked_scenes": index,
                                  "total_scenes": len(neutral_rows), "resumed_scenes": resumed,
                                  "new_scenes": processed}), flush=True)
            scene_id = str(neutral["opaque_scene_id"])
            state_path = state_root / f"{scene_id}.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("input_signature") != signature:
                    raise ValueError("Protected resume state identifies another release")
                for item in state["media"]:
                    target = model_root / item["path"]
                    if not target.is_file() or file_sha256(target) != item["sha256"]:
                        raise ValueError("Protected resume media failed checksum verification")
                resumed += 1
                continue
            if maximum_new_scenes is not None and processed >= maximum_new_scenes:
                break
            media = []
            rgb_rows = sorted(
                (row for row in source_rows[scene_id] if row["component"] == "rgb"),
                key=lambda row: str(row["view_id"]),
            )
            if len(rgb_rows) != contract.views_per_scene:
                raise ValueError("Protected RGB source coverage is incomplete")
            for row in rgb_rows:
                content = archive.read(str(row["source_member"]))
                with Image.open(BytesIO(content)) as opened:
                    opened.load()
                    if opened.size != (256, 256) or opened.mode not in {"RGB", "RGBA"}:
                        raise ValueError("Protected RGB image has an unexpected format")
                    mode = opened.mode
                relative = Path("media") / scene_id / "rgb" / f"{row['view_id']}.png"
                written += int(_atomic_bytes(model_root / relative, content))
                media.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": sha256(content).hexdigest(),
                        "size_bytes": len(content),
                        "media_mode": mode,
                    }
                )
            crosswalk, label_record, geometry = _evaluator_records(
                neutral=neutral,
                trusted=trusted_by_scene[scene_id],
                label=labels[scene_id],
                member_rows=source_rows[scene_id],
                archive=archive,
                contract=contract,
                scene_metadata=metadata_by_opaque[scene_id],
            )
            _replace_json(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "input_signature": signature,
                    "neutral": _safe_neutral(neutral, contract),
                    "crosswalk": crosswalk,
                    "labels": label_record,
                    "geometry": geometry,
                    "media": media,
                },
            )
            processed += 1
    state_paths = [state_root / f"{row['opaque_scene_id']}.json" for row in neutral_rows]
    completed = sum(path.is_file() for path in state_paths)
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "input_signature": signature,
        "cohort_id": contract.cohort_id,
        "completed_scene_count": completed,
        "remaining_scene_count": contract.scene_count - completed,
        "exact_resume": True,
    }
    _replace_json(state_root / "checkpoint.json", checkpoint)
    if completed != contract.scene_count:
        return {
            **checkpoint,
            "status": "checkpointed",
            "processed_scene_count": processed,
            "resumed_scene_count": resumed,
            "written_image_count": written,
            "model_resource_published": False,
            "evaluator_resource_published": False,
        }
    states = [json.loads(path.read_text(encoding="utf-8")) for path in state_paths]
    _atomic_bytes(
        model_root / "neutral_corpus_manifest.jsonl",
        _jsonl_bytes(state["neutral"] for state in states),
    )
    _atomic_bytes(
        evaluator_root / "crosswalk_private.jsonl",
        _jsonl_bytes(state["crosswalk"] for state in states),
    )
    _atomic_bytes(
        evaluator_root / "labels_private.jsonl",
        _jsonl_bytes(state["labels"] for state in states),
    )
    _atomic_bytes(
        evaluator_root / "geometry_private.jsonl",
        _jsonl_bytes(state["geometry"] for state in states),
    )
    media_paths = [Path(item["path"]) for state in states for item in state["media"]]
    checksum_bytes = _model_checksums(
        model_root, [Path("neutral_corpus_manifest.jsonl"), *media_paths]
    )
    _atomic_bytes(model_root / "content_checksums.sha256", checksum_bytes)
    modes = Counter(item["media_mode"] for state in states for item in state["media"])
    package = PackageAudit(
        cohort_id=contract.cohort_id,
        release_id=contract.release_id,
        scene_count=contract.scene_count,
        view_count=contract.views_per_scene,
        image_count=len(media_paths),
        width=256,
        height=256,
        media_mode=next(iter(modes)) if len(modes) == 1 else "mixed",
        total_media_bytes=sum(item["size_bytes"] for state in states for item in state["media"]),
        manifest_sha256=file_sha256(model_root / "neutral_corpus_manifest.jsonl"),
        membership_sha256=contract.membership_sha256,
        content_checksums_sha256=sha256(checksum_bytes).hexdigest(),
        verified_image_count=len(media_paths),
        duplicate_archive_member_count=0,
        duplicate_package_path_count=0,
        decoded_image_failure_count=0,
        all_checks_passed=True,
    )
    _atomic_bytes(model_root / "package_record.json", canonical_json_bytes(asdict(package)) + b"\n")
    verify_model_input_package(model_root, contract=contract)
    manifest = _evaluator_manifest(
        evaluator_root, contract, release_binding_sha256=signature
    )
    return {
        **checkpoint,
        "status": "complete",
        "processed_scene_count": processed,
        "resumed_scene_count": resumed,
        "written_image_count": written,
        "model_package_record_sha256": file_sha256(model_root / "package_record.json"),
        "evaluator_manifest_sha256": file_sha256(evaluator_root / "evaluator_manifest.json"),
        "evaluator_file_count": len(manifest["files"]),
        "model_resource_published": True,
        "evaluator_resource_published": True,
        "physical_resources_separate": model_root.resolve() != evaluator_root.resolve(),
    }


def build_development_identity_evaluator(
    *,
    prediction_roots: Sequence[Path],
    administrative_root: Path,
    source_root: Path,
    output_root: Path,
    maximum_new_scenes: int | None = None,
    minimum_iou: float = 0.50,
) -> dict[str, Any]:
    """Match locked development predictions to raw reference identities."""

    if maximum_new_scenes is not None and maximum_new_scenes < 1:
        raise ValueError("maximum_new_scenes must be positive")
    verified = load_08e_predictions(prediction_roots)
    if len(verified.views_by_scene) != DEVELOPMENT_FULL_CONTRACT.scene_count:
        raise ValueError("Development evaluator requires all 1,000 frozen scenes")
    scene_ids = sorted(verified.views_by_scene)
    if membership_sha256(scene_ids) != DEVELOPMENT_FULL_CONTRACT.membership_sha256:
        raise ValueError("Development predictions have the wrong cohort membership")
    trusted = {
        str(row["opaque_scene_id"]): row
        for row in read_jsonl(Path(administrative_root) / "restricted_source_map.jsonl")
        if str(row.get("role")) == "development"
    }
    if set(trusted) != set(scene_ids):
        raise ValueError("Development source map does not cover the frozen cohort")
    source_identity = verify_release_source(Path(source_root))
    sources = _required_by_scene(Path(administrative_root), set(scene_ids))
    binding = {
        "schema_version": SCHEMA_VERSION,
        "cohort_id": DEVELOPMENT_FULL_CONTRACT.cohort_id,
        "membership_sha256": DEVELOPMENT_FULL_CONTRACT.membership_sha256,
        "prediction_lock_sha256s": list(verified.input_lock_sha256s),
        "pipeline_identity_sha256": verified.frozen_identity_sha256,
        "administrative_sha256": file_sha256(Path(administrative_root) / "checksums.sha256"),
        "source_identity": source_identity,
        "minimum_iou": minimum_iou,
        "reference_geometry_policy": REFERENCE_GEOMETRY_POLICY,
    }
    signature = _digest(binding)
    output_root = Path(output_root).resolve()
    state_root = output_root / "trusted-release-state" / "development-identity"
    evaluator_root = output_root / "development-identity-evaluator"
    state_root.mkdir(parents=True, exist_ok=True)
    processed = resumed = 0
    with IndexedArchive(
        Path(source_root) / "toysAD8K.zip",
        Path(source_root) / "archive_index.sqlite3",
    ) as archive:
        for index, scene_id in enumerate(scene_ids):
            if index % 25 == 0:
                print(json.dumps({"event": "identity_release_progress", "checked_scenes": index,
                                  "total_scenes": len(scene_ids), "resumed_scenes": resumed,
                                  "new_scenes": processed}), flush=True)
            state_path = state_root / f"{scene_id}.json"
            if state_path.is_file():
                value = json.loads(state_path.read_text(encoding="utf-8"))
                if value.get("input_signature") != signature:
                    raise ValueError("Identity evaluator resume state identifies another input")
                resumed += 1
                continue
            if maximum_new_scenes is not None and processed >= maximum_new_scenes:
                break
            mask_rows = [row for row in sources[scene_id] if row["component"] == "object_mask"]
            by_view: dict[str, dict[str, list[int]]] = {}
            for row in mask_rows:
                by_view.setdefault(str(row["view_id"]), {})[
                    str(row["opaque_object_id"])
                ] = _mask_box(archive.read(str(row["source_member"])))
            correspondence = []
            for view in verified.views_by_scene[scene_id]:
                reference = by_view.get(view.view_id, {})
                # Invisible reference objects have no valid box to match.
                reference_ids = sorted(
                    key for key, box in reference.items()
                    if box[0] < box[2] and box[1] < box[3]
                )
                matches = match_boxes(
                    [reference[key] for key in reference_ids],
                    [list(item.box_xyxy) for item in view.objects],
                    minimum_iou=minimum_iou,
                )
                mapped = {
                    match.proposal_index: reference_ids[match.reference_index]
                    for match in matches
                }
                for index, item in enumerate(view.objects):
                    correspondence.append(
                        {
                            "schema_version": EVALUATOR_SCHEMA_VERSION,
                            "opaque_scene_id": scene_id,
                            "opaque_asset_group_id": str(
                                trusted[scene_id]["opaque_asset_group_id"]
                            ),
                            "view_id": view.view_id,
                            "object_id": item.object_id,
                            "reference_identity_id": mapped.get(index),
                        }
                    )
            _replace_json(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "input_signature": signature,
                    "correspondence": correspondence,
                },
            )
            processed += 1
    state_paths = [state_root / f"{scene_id}.json" for scene_id in scene_ids]
    completed = sum(path.is_file() for path in state_paths)
    checkpoint = {
        **binding,
        "input_signature": signature,
        "completed_scene_count": completed,
        "remaining_scene_count": DEVELOPMENT_FULL_CONTRACT.scene_count - completed,
        "exact_resume": True,
    }
    _replace_json(state_root / "checkpoint.json", checkpoint)
    if completed != DEVELOPMENT_FULL_CONTRACT.scene_count:
        return {
            "status": "checkpointed",
            "processed_scene_count": processed,
            "resumed_scene_count": resumed,
            **checkpoint,
            "evaluator_resource_published": False,
        }
    states = [json.loads(path.read_text(encoding="utf-8")) for path in state_paths]
    rows = [row for state in states for row in state["correspondence"]]
    mapped_count = sum(row["reference_identity_id"] is not None for row in rows)
    coverage = {
        "reference_mapped_observation_count": mapped_count,
        "reference_unmapped_observation_count": len(rows) - mapped_count,
        "reference_mapping_fraction": mapped_count / len(rows) if rows else 0.0,
    }
    _atomic_bytes(
        evaluator_root / "evaluator_correspondence.jsonl", _jsonl_bytes(rows)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked",
        "cohort_id": DEVELOPMENT_FULL_CONTRACT.cohort_id,
        "scene_count": DEVELOPMENT_FULL_CONTRACT.scene_count,
        "image_count": DEVELOPMENT_FULL_CONTRACT.image_count,
        "membership_sha256": DEVELOPMENT_FULL_CONTRACT.membership_sha256,
        "input_binding_sha256": signature,
        "reference_geometry_policy": REFERENCE_GEOMETRY_POLICY,
        **coverage,
        "correspondence_row_count": len(rows),
        "correspondence_sha256": file_sha256(
            evaluator_root / "evaluator_correspondence.jsonl"
        ),
    }
    _atomic_bytes(
        evaluator_root / "evaluator_manifest.json", canonical_json_bytes(manifest) + b"\n"
    )
    return {
        "status": "complete",
        "processed_scene_count": processed,
        "resumed_scene_count": resumed,
        **checkpoint,
        "correspondence_row_count": len(rows),
        "evaluator_manifest_sha256": file_sha256(evaluator_root / "evaluator_manifest.json"),
        "evaluator_resource_published": True,
        **coverage,
    }


__all__ = [
    "RAW_EVALUATOR_MANIFEST_SCHEMA",
    "ROLE_CONTRACTS",
    "SCHEMA_VERSION",
    "build_development_identity_evaluator",
    "build_protected_role_resources",
    "verify_release_source",
]
