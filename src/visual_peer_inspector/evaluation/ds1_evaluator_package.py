"""Build and verify the private evaluator package for the DS1 screen.

This module is deliberately separate from model-facing loaders. The package
contains the supplied masks, protected scene attributes and prompt crosswalks
needed only after LocateAnything and SAM inference have finished.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
)
from visual_peer_inspector.evaluation.proposal_reference_ds1 import (
    verify_ds1_attributes_package,
    verify_ds1_geometry_package,
)


SCHEMA_VERSION = 1
PACKAGE_KIND = "toysad8k-ds1-mask-evaluator"
INTENDED_USE = "evaluator_only"
_FILES = {
    "reference_manifest.jsonl",
    "protected_scene_attributes.jsonl",
    "evaluator_crosswalk.jsonl",
    "crosswalk_package_record.json",
    "usage_policy.json",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Evaluator package path already exists: {path}")
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _safe_relative(value: Any) -> Path:
    text = str(value)
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValueError("Evaluator package contains an unsafe relative path")
    return Path(*path.parts)


def _check_inputs(
    references: list[dict[str, Any]],
    attributes: list[dict[str, Any]],
    crosswalk: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = DS1_SCREEN_CONTRACT
    scene_ids = {str(row["opaque_scene_id"]) for row in attributes}
    if len(attributes) != contract.scene_count or len(scene_ids) != contract.scene_count:
        raise ValueError("DS1 protected attributes must contain 100 unique scenes")
    if membership_sha256(scene_ids) != contract.membership_sha256:
        raise ValueError("DS1 evaluator membership differs from the frozen contract")
    asset_ids = {str(row["opaque_asset_group_id"]) for row in attributes}
    if len(asset_ids) != contract.asset_group_count:
        raise ValueError("DS1 evaluator asset coverage differs from the contract")

    reference_keys: set[tuple[str, str, str]] = set()
    image_keys: set[tuple[str, str]] = set()
    visible_count = 0
    for row in references:
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        )
        if key in reference_keys or key[0] not in scene_ids:
            raise ValueError("DS1 reference manifest has an invalid object-view key")
        reference_keys.add(key)
        image_keys.add(key[:2])
        visible = bool(row["visible"])
        visible_count += int(visible)
        if visible:
            box = [int(value) for value in row["box_xyxy"]]
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 256 and 0 <= box[1] < box[3] <= 256):
                raise ValueError("Visible DS1 reference box is invalid")
            _safe_relative(row["mask_path"])
            digest = str(row["mask_sha256"])
            if len(digest) != 64:
                raise ValueError("DS1 reference mask SHA-256 is invalid")
    if len(image_keys) != contract.image_count:
        raise ValueError("DS1 evaluator must cover all 2,000 images")
    expected_views = {f"V{index:02d}" for index in range(contract.views_per_scene)}
    for scene_id in scene_ids:
        observed = {view for scene, view in image_keys if scene == scene_id}
        if observed != expected_views:
            raise ValueError(f"DS1 evaluator view coverage is incomplete for {scene_id}")

    crosswalk_keys: set[tuple[str, str, str, str]] = set()
    crosswalk_image_digests: dict[tuple[str, str], str] = {}
    for row in crosswalk:
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["route"]),
            str(row["prompt_key"]),
        )
        if key in crosswalk_keys or key[:2] not in image_keys:
            raise ValueError("DS1 evaluator crosswalk has an invalid prompt key")
        if key[2] not in {"reference_box", "automatic_box"}:
            raise ValueError("DS1 evaluator crosswalk has an unexpected route")
        image_key = key[:2]
        image_digest = str(row.get("image_sha256", ""))
        if len(image_digest) != 64 or (
            image_key in crosswalk_image_digests
            and crosswalk_image_digests[image_key] != image_digest
        ):
            raise ValueError("DS1 evaluator crosswalk RGB identity is invalid")
        crosswalk_image_digests[image_key] = image_digest
        object_id = row.get("opaque_object_id")
        if object_id is not None and (key[0], key[1], str(object_id)) not in reference_keys:
            raise ValueError("DS1 evaluator crosswalk refers to an unknown object")
        crosswalk_keys.add(key)
    if not crosswalk_keys:
        raise ValueError("DS1 evaluator crosswalk cannot be empty")
    if set(crosswalk_image_digests) != image_keys:
        raise ValueError("DS1 evaluator crosswalk RGB coverage is incomplete")
    return {
        "scene_count": len(scene_ids),
        "image_count": len(image_keys),
        "asset_group_count": len(asset_ids),
        "reference_row_count": len(reference_keys),
        "visible_mask_count": visible_count,
        "crosswalk_row_count": len(crosswalk_keys),
    }


def build_ds1_evaluator_package(
    *,
    reference_manifest_path: Path,
    geometry_package_record_path: Path,
    protected_attributes_path: Path,
    attributes_package_record_path: Path,
    evaluator_crosswalk_path: Path,
    crosswalk_package_record_path: Path,
    source_mask_root: Path,
    output_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an immutable evaluator-only DS1 package from trusted records."""

    expanded_geometry_record = geometry_package_record_path.expanduser()
    expanded_attributes_record = attributes_package_record_path.expanduser()
    if expanded_geometry_record.is_symlink() or expanded_attributes_record.is_symlink():
        raise ValueError("DS1 canonical package records cannot be symbolic links")
    reference_manifest_path = reference_manifest_path.expanduser().resolve(
        strict=True
    )
    geometry_package_record_path = expanded_geometry_record.resolve(strict=True)
    protected_attributes_path = protected_attributes_path.expanduser().resolve(
        strict=True
    )
    attributes_package_record_path = expanded_attributes_record.resolve(strict=True)
    evaluator_crosswalk_path = evaluator_crosswalk_path.expanduser().resolve(
        strict=True
    )
    crosswalk_package_record_path = (
        crosswalk_package_record_path.expanduser().resolve(strict=True)
    )
    geometry_root = geometry_package_record_path.parent
    attributes_root = attributes_package_record_path.parent
    expected_reference_path = geometry_root / "reference_manifest.jsonl"
    expected_attributes_path = attributes_root / "protected_scene_attributes.jsonl"
    expected_mask_root = geometry_root / "masks"
    if geometry_package_record_path.name != "package_record.json" or (
        reference_manifest_path != expected_reference_path
    ):
        raise ValueError("DS1 reference manifest is outside its canonical geometry package")
    if attributes_package_record_path.name != "package_record.json" or (
        protected_attributes_path != expected_attributes_path
    ):
        raise ValueError(
            "DS1 protected attributes are outside their canonical attribute package"
        )
    expanded_mask_root = source_mask_root.expanduser()
    if expanded_mask_root.is_symlink():
        raise ValueError("DS1 source-mask root cannot be a symbolic link")
    source_mask_root = expanded_mask_root.resolve(strict=True)
    if source_mask_root != expected_mask_root:
        raise ValueError("DS1 source masks are outside the canonical geometry package")

    geometry_record = verify_ds1_geometry_package(geometry_root)
    attributes_record = verify_ds1_attributes_package(attributes_root)
    references = read_jsonl(reference_manifest_path)
    attributes = read_jsonl(protected_attributes_path)
    crosswalk = read_jsonl(evaluator_crosswalk_path)
    counts = _check_inputs(references, attributes, crosswalk)
    crosswalk_record = json.loads(
        crosswalk_package_record_path.read_text(encoding="utf-8")
    )
    expected_crosswalk = {
        "package_kind": "ds1-sam-evaluator-crosswalk",
        "intended_use": "evaluator_only",
        "model_facing_allowed": False,
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "release_id": DS1_SCREEN_CONTRACT.release_id,
        "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
        "scene_count": DS1_SCREEN_CONTRACT.scene_count,
        "image_count": DS1_SCREEN_CONTRACT.image_count,
        "row_count": len(crosswalk),
        "all_checks_passed": True,
    }
    for key, expected in expected_crosswalk.items():
        if crosswalk_record.get(key) != expected:
            raise ValueError(f"DS1 crosswalk package record mismatch: {key}")
    crosswalk_bytes = evaluator_crosswalk_path.read_bytes()
    if crosswalk_record.get("crosswalk_sha256") != sha256(crosswalk_bytes).hexdigest():
        raise ValueError("DS1 crosswalk package record identifies another crosswalk")
    geometry_package_record_sha256 = file_sha256(geometry_package_record_path)
    geometry_manifest_sha256 = file_sha256(reference_manifest_path)
    attributes_package_record_sha256 = file_sha256(attributes_package_record_path)
    protected_attributes_sha256 = file_sha256(protected_attributes_path)
    if (
        crosswalk_record.get("geometry_package_record_sha256")
        != geometry_package_record_sha256
        or crosswalk_record.get("geometry_manifest_sha256")
        != geometry_manifest_sha256
        or geometry_record.get("reference_manifest_sha256")
        != geometry_manifest_sha256
        or attributes_record.get("protected_attributes_sha256")
        != protected_attributes_sha256
    ):
        raise ValueError("DS1 evaluator sources differ from their canonical package chain")
    required_crosswalk_hashes = (
        "prompt_package_record_sha256",
        "prompt_manifest_sha256",
        "model_input_manifest_sha256",
        "geometry_package_record_sha256",
        "geometry_manifest_sha256",
        "proposal_results_sha256",
        "crosswalk_sha256",
    )
    if any(
        not _SHA256_PATTERN.fullmatch(str(crosswalk_record.get(key, "")))
        for key in required_crosswalk_hashes
    ):
        raise ValueError("DS1 crosswalk package record has incomplete upstream hashes")
    source_mask_root = source_mask_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"DS1 evaluator output already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".incomplete")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Incomplete DS1 evaluator output exists: {staging}")
    staging.mkdir(parents=True)
    try:
        _write(staging / "reference_manifest.jsonl", _jsonl_bytes(references))
        _write(staging / "protected_scene_attributes.jsonl", _jsonl_bytes(attributes))
        _write(staging / "evaluator_crosswalk.jsonl", _jsonl_bytes(crosswalk))
        _write(
            staging / "crosswalk_package_record.json",
            canonical_json_bytes(crosswalk_record) + b"\n",
        )
        policy = {
            "schema_version": SCHEMA_VERSION,
            "package_kind": PACKAGE_KIND,
            "intended_use": INTENDED_USE,
            "model_facing_allowed": False,
            "contains_labels": True,
            "contains_source_crosswalks": True,
            "clearml_full_package_upload_allowed": False,
        }
        _write(staging / "usage_policy.json", canonical_json_bytes(policy) + b"\n")
        copied_masks = 0
        for row in references:
            if not bool(row["visible"]):
                continue
            relative = _safe_relative(row["mask_path"])
            source = source_mask_root / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"DS1 supplied mask is missing: {relative}")
            if file_sha256(source) != str(row["mask_sha256"]):
                raise ValueError("DS1 supplied mask differs from its frozen SHA-256")
            target = staging / "masks" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied_masks += 1
        if copied_masks != counts["visible_mask_count"]:
            raise ValueError("DS1 supplied-mask copy count is incomplete")
        content = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            content.append({
                "path": path.relative_to(staging).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            })
        checksums = "".join(f"{row['sha256']}  {row['path']}\n" for row in content).encode()
        _write(staging / "content_checksums.sha256", checksums)
        record = {
            "schema_version": SCHEMA_VERSION,
            "package_kind": PACKAGE_KIND,
            "intended_use": INTENDED_USE,
            "model_facing_allowed": False,
            "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
            "release_id": DS1_SCREEN_CONTRACT.release_id,
            "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
            **counts,
            "provenance": dict(provenance),
            "crosswalk_package_record_sha256": file_sha256(
                staging / "crosswalk_package_record.json"
            ),
            "prompt_package_record_sha256": crosswalk_record[
                "prompt_package_record_sha256"
            ],
            "prompt_manifest_sha256": crosswalk_record["prompt_manifest_sha256"],
            "model_input_manifest_sha256": crosswalk_record[
                "model_input_manifest_sha256"
            ],
            "geometry_package_record_sha256": crosswalk_record[
                "geometry_package_record_sha256"
            ],
            "geometry_manifest_sha256": crosswalk_record[
                "geometry_manifest_sha256"
            ],
            "proposal_results_sha256": crosswalk_record[
                "proposal_results_sha256"
            ],
            "crosswalk_sha256": crosswalk_record["crosswalk_sha256"],
            "attributes_package_record_sha256": attributes_package_record_sha256,
            "protected_attributes_sha256": protected_attributes_sha256,
            "geometry_content_checksums_sha256": geometry_record[
                "content_checksums_sha256"
            ],
            "attributes_content_checksums_sha256": attributes_record[
                "content_checksums_sha256"
            ],
            "content_file_count": len(content),
            "content_checksums_sha256": sha256(checksums).hexdigest(),
            "all_checks_passed": True,
        }
        _write(staging / "package_record.json", canonical_json_bytes(record) + b"\n")
        verify_ds1_evaluator_package(staging)
        os.replace(staging, output_root)
        return record
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_ds1_evaluator_package(root: Path, *, intended_use: str = INTENDED_USE) -> dict[str, Any]:
    """Verify every evaluator package file and reject model-facing use."""

    if intended_use != INTENDED_USE:
        raise ValueError("The DS1 evaluator package cannot be used by a model")
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("DS1 evaluator package root is missing or unsafe")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("DS1 evaluator package cannot contain symbolic links")
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    if record.get("package_kind") != PACKAGE_KIND or record.get("model_facing_allowed") is not False:
        raise ValueError("DS1 evaluator package declaration is invalid")
    if record.get("cohort_id") != DS1_SCREEN_CONTRACT.cohort_id or record.get("membership_sha256") != DS1_SCREEN_CONTRACT.membership_sha256:
        raise ValueError("DS1 evaluator package identifies another cohort")
    policy = json.loads((root / "usage_policy.json").read_text(encoding="utf-8"))
    if policy.get("model_facing_allowed") is not False or policy.get("intended_use") != INTENDED_USE:
        raise ValueError("DS1 evaluator usage policy is invalid")
    counts = _check_inputs(
        read_jsonl(root / "reference_manifest.jsonl"),
        read_jsonl(root / "protected_scene_attributes.jsonl"),
        read_jsonl(root / "evaluator_crosswalk.jsonl"),
    )
    crosswalk_record_path = root / "crosswalk_package_record.json"
    crosswalk_record = json.loads(crosswalk_record_path.read_text(encoding="utf-8"))
    if (
        crosswalk_record.get("package_kind") != "ds1-sam-evaluator-crosswalk"
        or crosswalk_record.get("intended_use") != INTENDED_USE
        or crosswalk_record.get("model_facing_allowed") is not False
        or crosswalk_record.get("cohort_id") != DS1_SCREEN_CONTRACT.cohort_id
        or crosswalk_record.get("membership_sha256")
        != DS1_SCREEN_CONTRACT.membership_sha256
        or crosswalk_record.get("crosswalk_sha256")
        != file_sha256(root / "evaluator_crosswalk.jsonl")
        or int(crosswalk_record.get("row_count", -1)) != counts["crosswalk_row_count"]
        or crosswalk_record.get("all_checks_passed") is not True
    ):
        raise ValueError("DS1 evaluator crosswalk package record is invalid")
    linked_hashes = {
        "crosswalk_package_record_sha256": file_sha256(crosswalk_record_path),
        "prompt_package_record_sha256": crosswalk_record.get(
            "prompt_package_record_sha256"
        ),
        "prompt_manifest_sha256": crosswalk_record.get("prompt_manifest_sha256"),
        "model_input_manifest_sha256": crosswalk_record.get(
            "model_input_manifest_sha256"
        ),
        "geometry_package_record_sha256": crosswalk_record.get(
            "geometry_package_record_sha256"
        ),
        "geometry_manifest_sha256": crosswalk_record.get(
            "geometry_manifest_sha256"
        ),
        "proposal_results_sha256": crosswalk_record.get("proposal_results_sha256"),
        "crosswalk_sha256": crosswalk_record.get("crosswalk_sha256"),
    }
    if any(record.get(key) != value for key, value in linked_hashes.items()):
        raise ValueError("DS1 evaluator upstream hash chain differs from its record")
    for key, value in counts.items():
        if int(record.get(key, -1)) != value:
            raise ValueError(f"DS1 evaluator package count differs for {key}")
    source_hashes = {
        "geometry_package_record_sha256",
        "geometry_manifest_sha256",
        "geometry_content_checksums_sha256",
        "attributes_package_record_sha256",
        "protected_attributes_sha256",
        "attributes_content_checksums_sha256",
    }
    if any(
        not isinstance(record.get(key), str)
        or not _SHA256_PATTERN.fullmatch(record[key])
        for key in source_hashes
    ):
        raise ValueError("DS1 evaluator source identity chain is incomplete")
    if (
        file_sha256(root / "reference_manifest.jsonl")
        != record["geometry_manifest_sha256"]
        or file_sha256(root / "protected_scene_attributes.jsonl")
        != record["protected_attributes_sha256"]
    ):
        raise ValueError("DS1 evaluator copied sources differ from their identity chain")
    checksums: dict[str, str] = {}
    for line in (root / "content_checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[_safe_relative(relative).as_posix()] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"package_record.json", "content_checksums.sha256"}
    }
    if actual != set(checksums) or len(actual) != int(record["content_file_count"]):
        raise ValueError("DS1 evaluator content allowlist differs from its record")
    for relative, digest in checksums.items():
        path = root / relative
        if file_sha256(path) != digest:
            raise ValueError("DS1 evaluator content checksum mismatch")
    checksum_digest = file_sha256(root / "content_checksums.sha256")
    if checksum_digest != record["content_checksums_sha256"]:
        raise ValueError("DS1 evaluator checksum manifest differs from its record")
    if record.get("all_checks_passed") is not True:
        raise ValueError("DS1 evaluator package is not marked verified")
    return record


__all__ = [
    "INTENDED_USE",
    "PACKAGE_KIND",
    "build_ds1_evaluator_package",
    "verify_ds1_evaluator_package",
]
