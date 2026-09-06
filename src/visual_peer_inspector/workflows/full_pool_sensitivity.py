"""Post-freeze full-pool sensitivity release and evaluation contracts.

This stage is descriptive. It reuses the already frozen method and policy after
the official evaluation is complete. Nothing produced here can change the
primary or official result.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import (
    PackageAudit,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.data.toysad_selection import (
    SourcePaths,
    assemble_eligible_scene_pool,
    verify_pinned_sources,
)
from visual_peer_inspector.workflows.official_release import (
    IndexedArchive,
    _atomic_bytes,
    _atomic_json,
    _jsonl_bytes,
    _model_checksum_bytes,
    _scene_records,
    _write_evaluator_manifest,
)

STAGE_RELEASE = "post_freeze_full_pool_release"
STAGE_INFERENCE = "post_freeze_full_pool_inference"
STAGE_EVALUATION = "post_freeze_full_pool_sensitivity_evaluation"
RELEASE_ID = "toysad8k-post-freeze-sensitivity-v1"
COHORT_ID = "toysad8k-eligible-train-full-pool-4107-v1"
SCENE_COUNT = 4_107
VIEW_COUNT = 20
IMAGE_COUNT = 82_140
ASSET_GROUP_COUNT = 545
PRIMARY_ROLE_COUNTS = {
    "development": 1_000,
    "calibration": 600,
    "internal_lock": 400,
    "eligible_remainder": 2_107,
}
MODEL_MEMBERSHIP_SHA256 = membership_sha256(
    f"F{index:06d}" for index in range(1, SCENE_COUNT + 1)
)


def _digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def full_pool_opaque_scene_id(index: int) -> str:
    """Return the stable full-pool model identifier for one ordered scene."""

    if not 1 <= index <= SCENE_COUNT:
        raise ValueError("Full-pool scene index is outside the release")
    return f"F{index:06d}"


def full_pool_contract(source_membership_sha256: str) -> CohortContract:
    """Build the runtime contract bound to the verified eligible source pool."""

    return CohortContract(
        cohort_id=COHORT_ID,
        release_id=RELEASE_ID,
        role="post-freeze-sensitivity",
        pilot_stage=None,
        membership_sha256=source_membership_sha256,
        model_membership_sha256=MODEL_MEMBERSHIP_SHA256,
        scene_count=SCENE_COUNT,
        views_per_scene=VIEW_COUNT,
        image_count=IMAGE_COUNT,
        asset_group_count=ASSET_GROUP_COUNT,
    )


def verify_10b_completion(result_root: Path) -> dict[str, Any]:
    """Verify completed official evidence before authorising sensitivity work."""

    root = Path(result_root).expanduser().resolve(strict=True)
    completion_path = root / "evaluation_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    safe_root = root / "safe"
    run_path = safe_root / "run_record.json"
    manifest_path = safe_root / "artifact_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        completion.get("schema_version") != "visual-peer-one-shot-evaluation/v1"
        or completion.get("status") != "complete"
        or completion.get("stage") != "10b"
        or completion.get("safe_manifest_sha256") != file_sha256(manifest_path)
        or run.get("status") != "completed"
        or run.get("stage") != "10b"
        or run.get("no_retuning") is not True
        or run.get("method_changed") is not False
        or run.get("policy_changed") is not False
    ):
        raise ValueError("Notebook 10B is not a completed frozen official evaluation")
    declared = manifest.get("artifacts")
    if not isinstance(declared, list) or int(manifest.get("artifact_count", -1)) != len(declared):
        raise ValueError("Notebook 10B safe artifact manifest is invalid")
    for item in declared:
        path = safe_root / str(item["path"])
        if file_sha256(path) != item["sha256"] or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError("Notebook 10B safe evidence differs from its manifest")
    return {
        "schema_version": "visual-peer-11-post-freeze-authorization/v1",
        "status": "authorised",
        "scope": "descriptive_sensitivity_only",
        "official_evaluation_completion_sha256": file_sha256(completion_path),
        "official_safe_manifest_sha256": file_sha256(manifest_path),
        "method_record_sha256": str(run["method_record_sha256"]),
        "policy_record_sha256": str(run["policy_record_sha256"]),
        "no_retuning": True,
        "primary_results_unchanged": True,
        "official_results_unchanged": True,
    }


def load_primary_role_map(administrative_root: Path) -> dict[str, dict[str, Any]]:
    """Load the original primary roles without changing their meaning."""

    path = Path(administrative_root).expanduser().resolve(strict=True) / "restricted_source_map.jsonl"
    rows = read_jsonl(path)
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_scene_id", ""))
        role = str(row.get("role", ""))
        if not source_id or source_id in mapping or role not in {"development", "calibration", "internal_lock"}:
            raise ValueError("Primary source map contains an invalid role record")
        mapping[source_id] = {
            "primary_role": role,
            "primary_pilot_stage": row.get("pilot_stage"),
        }
    counts = Counter(item["primary_role"] for item in mapping.values())
    if counts != Counter({key: value for key, value in PRIMARY_ROLE_COUNTS.items() if key != "eligible_remainder"}):
        raise ValueError("Primary source map does not preserve the 1,000/600/400 roles")
    return mapping


def prepare_full_pool_release(
    *,
    source_paths: SourcePaths,
    official_result_root: Path,
    administrative_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], CohortContract, dict[str, dict[str, Any]]]:
    """Verify every frozen input and construct the deterministic release plan."""

    authorization = verify_10b_completion(official_result_root)
    master, source_hashes = verify_pinned_sources(source_paths)
    scenes = assemble_eligible_scene_pool(master, source_paths.official_split)
    source_membership = membership_sha256(str(scene["scene_id"]) for scene in scenes)
    contract = full_pool_contract(source_membership)
    role_map = load_primary_role_map(administrative_root)
    source_ids = {str(scene["scene_id"]) for scene in scenes}
    if not set(role_map).issubset(source_ids):
        raise ValueError("A primary scene is absent from the eligible full pool")
    role_counts = Counter(
        role_map.get(str(scene["scene_id"]), {"primary_role": "eligible_remainder"})["primary_role"]
        for scene in scenes
    )
    if role_counts != Counter(PRIMARY_ROLE_COUNTS):
        raise ValueError("Full-pool primary role labels do not match the protocol")
    plan = {
        "schema_version": "visual-peer-11r-release-plan/v1",
        "cohort_id": COHORT_ID,
        "release_id": RELEASE_ID,
        "source_membership_sha256": source_membership,
        "model_membership_sha256": MODEL_MEMBERSHIP_SHA256,
        "scene_count": len(scenes),
        "image_count": IMAGE_COUNT,
        "asset_group_count": len({str(scene["base_asset_id"]) for scene in scenes}),
        "category_count": len({str(scene["category"]) for scene in scenes}),
        "primary_role_counts": dict(sorted(role_counts.items())),
        "source_hashes": source_hashes,
        "official_authorization": authorization,
        "descriptive_sensitivity_only": True,
        "method_or_policy_may_change": False,
    }
    plan["plan_sha256"] = _digest(plan)
    return plan, scenes, contract, role_map


def estimate_full_pool_rgb_bytes(
    archive_index: Path, scenes: Sequence[Mapping[str, Any]]
) -> int:
    """Return the indexed uncompressed RGB size for the eligible pool."""

    total = 0
    database_path = Path(archive_index).expanduser().resolve(strict=True)
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
        for scene in scenes:
            member_root = str(scene["archive_members"]["rgb_directory"])
            for view_index in range(VIEW_COUNT):
                rows = database.execute(
                    "SELECT uncompressed_size FROM members WHERE name = ?",
                    (f"{member_root}/{view_index:04d}.png",),
                ).fetchall()
                if len(rows) != 1:
                    raise ValueError("An eligible RGB member is absent from the archive index")
                total += int(rows[0][0])
    return total


def build_full_pool_resource(
    *,
    archive_zip: Path,
    archive_index: Path,
    scenes: Sequence[Mapping[str, Any]],
    contract: CohortContract,
    role_map: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    plan_sha256: str,
    maximum_new_scenes: int | None = None,
) -> dict[str, Any]:
    """Build or resume the separated full-pool resources."""

    if maximum_new_scenes is not None and maximum_new_scenes < 1:
        raise ValueError("maximum_new_scenes must be positive")
    root = Path(output_root).expanduser().resolve()
    archive_path = Path(archive_zip).expanduser().resolve(strict=True)
    index_path = Path(archive_index).expanduser().resolve(strict=True)
    model_root = root / "full-pool-model-input"
    evaluator_root = root / "full-pool-evaluator-only"
    state_root = root / "trusted-release-state"
    state_root.mkdir(parents=True, exist_ok=True)
    opaque_ids = [full_pool_opaque_scene_id(index) for index in range(1, len(scenes) + 1)]
    if membership_sha256(opaque_ids) != MODEL_MEMBERSHIP_SHA256:
        raise ValueError("Full-pool opaque membership differs")
    signature = _digest({
        "plan_sha256": plan_sha256,
        "source_membership_sha256": contract.membership_sha256,
        "model_membership_sha256": contract.model_membership_sha256,
    })
    processed = resumed = written = 0
    with IndexedArchive(archive_path, index_path) as archive:
        for scene, opaque_id in zip(scenes, opaque_ids, strict=True):
            state_path = state_root / f"{opaque_id}.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("input_signature") != signature:
                    raise ValueError("Full-pool recovery state belongs to another release")
                for item in state["media"]:
                    media_path = model_root / str(item["path"])
                    if not media_path.is_file() or file_sha256(media_path) != item["sha256"]:
                        raise ValueError("Recovered full-pool RGB differs from its state")
                resumed += 1
                continue
            if maximum_new_scenes is not None and processed >= maximum_new_scenes:
                break
            member_root = str(scene["archive_members"]["rgb_directory"])
            media = []
            for view_index in range(VIEW_COUNT):
                content = archive.read(f"{member_root}/{view_index:04d}.png")
                with Image.open(BytesIO(content)) as image:
                    image.load()
                    if (
                        image.format != "PNG"
                        or image.size != (256, 256)
                        or image.mode not in {"RGB", "RGBA"}
                    ):
                        raise ValueError("Full-pool RGB media has an unexpected format")
                    if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
                        raise ValueError("Full-pool RGB media contains transparency")
                    media_mode = image.mode
                relative = Path(f"media/{opaque_id}/rgb/V{view_index:02d}.png")
                if _atomic_bytes(model_root / relative, content):
                    written += 1
                media.append({
                    "path": relative.as_posix(),
                    "sha256": sha256(content).hexdigest(),
                    "size_bytes": len(content),
                    "media_mode": media_mode,
                })
            scene2d = json.loads(archive.read(str(scene["archive_members"]["scene2d_metadata"])))
            scene3d = json.loads(archive.read(str(scene["archive_members"]["scene3d_metadata"])))
            neutral, crosswalk, labels, geometry = _scene_records(
                scene, opaque_id, contract, scene2d, scene3d
            )
            preserved = role_map.get(str(scene["scene_id"]), {
                "primary_role": "eligible_remainder",
                "primary_pilot_stage": None,
            })
            crosswalk.update(preserved)
            crosswalk["source_split"] = "train"
            _atomic_json(state_path, {
                "schema_version": "visual-peer-11r-scene-state/v1",
                "input_signature": signature,
                "neutral": neutral,
                "crosswalk": crosswalk,
                "labels": labels,
                "geometry": geometry,
                "media": media,
            })
            processed += 1
    completed_ids = [opaque_id for opaque_id in opaque_ids if (state_root / f"{opaque_id}.json").is_file()]
    checkpoint = {
        "schema_version": "visual-peer-11r-checkpoint/v1",
        "status": "complete" if len(completed_ids) == SCENE_COUNT else "checkpointed",
        "input_signature": signature,
        "completed_scene_count": len(completed_ids),
        "remaining_scene_count": SCENE_COUNT - len(completed_ids),
        "exact_resume": True,
    }
    _atomic_json(root / "release_checkpoint.json", checkpoint)
    if checkpoint["status"] != "complete":
        return {**checkpoint, "processed_this_run": processed, "resumed_this_run": resumed, "written_images_this_run": written}

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
        scene_count=SCENE_COUNT,
        view_count=VIEW_COUNT,
        image_count=IMAGE_COUNT,
        width=256,
        height=256,
        media_mode=next(iter(modes)) if len(modes) == 1 else "mixed",
        total_media_bytes=sum(int(item["size_bytes"]) for state in states for item in state["media"]),
        manifest_sha256=file_sha256(model_root / "neutral_corpus_manifest.jsonl"),
        membership_sha256=MODEL_MEMBERSHIP_SHA256,
        content_checksums_sha256=sha256(checksum_bytes).hexdigest(),
        verified_image_count=IMAGE_COUNT,
        duplicate_archive_member_count=0,
        duplicate_package_path_count=0,
        decoded_image_failure_count=0,
        all_checks_passed=True,
    )
    _atomic_json(model_root / "package_record.json", asdict(audit))
    verify_model_input_package(model_root, contract=contract)
    evaluator_manifest = _write_evaluator_manifest(evaluator_root, contract)
    return {
        **checkpoint,
        "processed_this_run": processed,
        "resumed_this_run": resumed,
        "written_images_this_run": written,
        "model_package_record_sha256": file_sha256(model_root / "package_record.json"),
        "evaluator_manifest_sha256": file_sha256(evaluator_root / "evaluator_manifest.json"),
        "evaluator_file_count": len(evaluator_manifest["files"]),
        "model_and_evaluator_physically_separate": True,
    }


def load_full_pool_release_receipt(path: Path) -> tuple[dict[str, Any], CohortContract]:
    """Verify the immutable 11R release receipt and rebuild its cohort contract."""

    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version": "visual-peer-11r-release-receipt/v1",
        "status": "complete",
        "stage": STAGE_RELEASE,
        "cohort_id": COHORT_ID,
        "scene_count": SCENE_COUNT,
        "image_count": IMAGE_COUNT,
        "asset_group_count": ASSET_GROUP_COUNT,
        "model_membership_sha256": MODEL_MEMBERSHIP_SHA256,
        "descriptive_sensitivity_only": True,
        "no_retuning": True,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"Full-pool release receipt differs: {field}")
    recorded = value.get("receipt_sha256")
    if recorded != _digest({key: item for key, item in value.items() if key != "receipt_sha256"}):
        raise ValueError("Full-pool release receipt self-hash differs")
    return {"path": source, "sha256": file_sha256(source), **value}, full_pool_contract(
        str(value["source_membership_sha256"])
    )


__all__ = [
    "ASSET_GROUP_COUNT",
    "COHORT_ID",
    "IMAGE_COUNT",
    "MODEL_MEMBERSHIP_SHA256",
    "PRIMARY_ROLE_COUNTS",
    "RELEASE_ID",
    "SCENE_COUNT",
    "STAGE_EVALUATION",
    "STAGE_INFERENCE",
    "STAGE_RELEASE",
    "build_full_pool_resource",
    "estimate_full_pool_rgb_bytes",
    "full_pool_contract",
    "full_pool_opaque_scene_id",
    "load_full_pool_release_receipt",
    "load_primary_role_map",
    "prepare_full_pool_release",
    "verify_10b_completion",
]
