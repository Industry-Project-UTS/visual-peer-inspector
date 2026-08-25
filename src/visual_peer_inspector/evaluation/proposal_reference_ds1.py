"""Build trusted DS1 geometry and protected evaluator attributes."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.archive_members import read_indexed_member
from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZE_BYTES,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
)
from visual_peer_inspector.evaluation.proposal_reference import (
    DEFAULT_MASK_THRESHOLD,
    _load_indexed_members,
    _mask_geometry,
)


SCHEMA_VERSION = 1
GEOMETRY_KIND = "ds1-oracle-geometry"
ATTRIBUTES_KIND = "ds1-evaluator-attributes"


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"DS1 reference output already exists: {path}")
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _checksum_bytes(root: Path, paths: Iterable[Path]) -> bytes:
    return "".join(
        f"{file_sha256(root / relative)}  {relative.as_posix()}\n"
        for relative in sorted(set(paths), key=lambda value: value.as_posix())
    ).encode("utf-8")


def build_ds1_reference_packages(
    *,
    model_manifest_path: Path,
    development_source_map_path: Path,
    development_labels_path: Path,
    required_source_members_path: Path,
    source_archive_path: Path,
    archive_index_path: Path,
    geometry_output_root: Path,
    attributes_output_root: Path,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
) -> dict[str, Any]:
    """Extract the 100-scene supplied masks and protected attributes."""

    contract = DS1_SCREEN_CONTRACT
    if not 1 <= int(mask_threshold) <= 255:
        raise ValueError("Mask threshold must be between 1 and 255")
    model_rows = read_jsonl(model_manifest_path)
    selected_ids = {str(row["opaque_scene_id"]) for row in model_rows}
    if len(model_rows) != contract.scene_count or len(selected_ids) != contract.scene_count:
        raise ValueError("DS1 model manifest must contain 100 unique scenes")
    if membership_sha256(selected_ids) != contract.membership_sha256:
        raise ValueError("DS1 reference membership differs from the frozen contract")
    expected_views = [f"V{index:02d}" for index in range(contract.views_per_scene)]
    for row in model_rows:
        if [str(value) for value in row["view_ids"]] != expected_views:
            raise ValueError("Every DS1 scene must contain the frozen 20 views")

    source_rows = [row for row in read_jsonl(development_source_map_path) if str(row.get("opaque_scene_id")) in selected_ids]
    label_rows = [row for row in read_jsonl(development_labels_path) if str(row.get("opaque_scene_id")) in selected_ids]
    source_by_scene = {str(row["opaque_scene_id"]): row for row in source_rows}
    labels_by_scene = {str(row["opaque_scene_id"]): row for row in label_rows}
    if len(source_rows) != len(source_by_scene) or len(label_rows) != len(labels_by_scene):
        raise ValueError("Protected DS1 inputs contain duplicate scenes")
    if set(source_by_scene) != selected_ids or set(labels_by_scene) != selected_ids:
        raise ValueError("Protected development records do not cover DS1")

    required_rows = [
        row for row in read_jsonl(required_source_members_path)
        if str(row.get("opaque_scene_id")) in selected_ids
        and row.get("component") == "object_mask"
        and row.get("include_in_media_package") is True
    ]
    required_keys = [(str(row["opaque_scene_id"]), str(row["view_id"]), str(row["opaque_object_id"])) for row in required_rows]
    if len(required_keys) != len(set(required_keys)):
        raise ValueError("Protected DS1 mask records contain duplicate object views")
    expected_object_views = sum(int(row["object_count"]) * contract.views_per_scene for row in model_rows)
    if len(required_rows) != expected_object_views:
        raise ValueError(f"Expected {expected_object_views} DS1 object masks, found {len(required_rows)}")

    source_archive_path = source_archive_path.expanduser().resolve()
    archive_index_path = archive_index_path.expanduser().resolve()
    if source_archive_path.is_symlink() or source_archive_path.stat().st_size != EXPECTED_ARCHIVE_SIZE_BYTES:
        raise ValueError("Source archive size differs from the accepted ToysAD archive")
    if file_sha256(source_archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Source archive SHA-256 differs from the accepted ToysAD archive")
    if archive_index_path.is_symlink() or not archive_index_path.is_file():
        raise ValueError("ToysAD archive index is missing or unsafe")
    indexed = _load_indexed_members(archive_index_path, (str(row["source_member"]) for row in required_rows))

    geometry_output_root = geometry_output_root.expanduser().resolve()
    attributes_output_root = attributes_output_root.expanduser().resolve()
    if geometry_output_root == attributes_output_root or geometry_output_root.exists() or attributes_output_root.exists():
        raise FileExistsError("DS1 reference outputs require two new, separate roots")
    geometry_staging = geometry_output_root.with_name(geometry_output_root.name + ".incomplete")
    attributes_staging = attributes_output_root.with_name(attributes_output_root.name + ".incomplete")
    if geometry_staging.exists() or attributes_staging.exists():
        raise FileExistsError("Incomplete DS1 reference output already exists")
    geometry_staging.mkdir(parents=True)
    attributes_staging.mkdir(parents=True)
    try:
        reference_rows: list[dict[str, Any]] = []
        mask_paths: list[Path] = []
        visible_count = 0
        with source_archive_path.open("rb") as archive:
            for row in sorted(required_rows, key=lambda item: (str(item["opaque_scene_id"]), str(item["view_id"]), str(item["opaque_object_id"]))):
                member = indexed[str(row["source_member"])]
                if member.crc32 != int(str(row["crc32"]), 16) or member.uncompressed_size != int(row["uncompressed_size"]):
                    raise ValueError("Archive member metadata differs from the protected DS1 record")
                data = read_indexed_member(archive, member)
                box, foreground = _mask_geometry(data, int(mask_threshold))
                relative = Path(str(row["opaque_scene_id"])) / str(row["opaque_object_id"]) / f"{row['view_id']}.png"
                _write(geometry_staging / "masks" / relative, data)
                mask_paths.append(Path("masks") / relative)
                visible_count += int(box is not None)
                reference_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "opaque_scene_id": str(row["opaque_scene_id"]),
                    "view_id": str(row["view_id"]),
                    "opaque_object_id": str(row["opaque_object_id"]),
                    "mask_path": relative.as_posix(),
                    "mask_sha256": sha256(data).hexdigest(),
                    "foreground_pixels": foreground,
                    "visible": box is not None,
                    "box_xyxy": box,
                })
        protected_rows = []
        model_by_scene = {str(row["opaque_scene_id"]): row for row in model_rows}
        for scene_id in sorted(selected_ids):
            source = source_by_scene[scene_id]
            labels = labels_by_scene[scene_id]
            labels_by_object = {str(key): int(value) for key, value in dict(labels["labels_by_object_id"]).items()}
            if set(labels_by_object) != {str(value) for value in model_by_scene[scene_id]["object_ids"]}:
                raise ValueError("Protected DS1 labels do not match the opaque object IDs")
            protected_rows.append({
                "schema_version": SCHEMA_VERSION,
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": str(source["opaque_asset_group_id"]),
                "category": str(source["category"]),
                "object_count": int(source["object_count"]),
                "anomaly_count": int(source["anomaly_count"]),
                "anomaly_families": sorted(str(value) for value in source["raw_anomaly_prefixes"]),
                "labels_by_object_id": labels_by_object,
            })
        reference_path = geometry_staging / "reference_manifest.jsonl"
        attributes_path = attributes_staging / "protected_scene_attributes.jsonl"
        _write(reference_path, _jsonl_bytes(reference_rows))
        _write(attributes_path, _jsonl_bytes(protected_rows))
        geometry_checksums = _checksum_bytes(geometry_staging, [Path("reference_manifest.jsonl"), *mask_paths])
        attributes_checksums = _checksum_bytes(attributes_staging, [Path("protected_scene_attributes.jsonl")])
        _write(geometry_staging / "content_checksums.sha256", geometry_checksums)
        _write(attributes_staging / "content_checksums.sha256", attributes_checksums)
        common = {
            "schema_version": SCHEMA_VERSION,
            "cohort_id": contract.cohort_id,
            "release_id": contract.release_id,
            "membership_sha256": contract.membership_sha256,
            "scene_count": contract.scene_count,
            "views_per_scene": contract.views_per_scene,
            "image_count": contract.image_count,
            "asset_group_count": contract.asset_group_count,
        }
        geometry_record = {
            **common,
            "package_kind": GEOMETRY_KIND,
            "object_view_count": len(reference_rows),
            "visible_object_view_count": visible_count,
            "empty_object_view_count": len(reference_rows) - visible_count,
            "mask_threshold": int(mask_threshold),
            "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "source_archive_size_bytes": EXPECTED_ARCHIVE_SIZE_BYTES,
            "archive_index_sha256": file_sha256(archive_index_path),
            "reference_manifest_sha256": file_sha256(reference_path),
            "content_checksums_sha256": sha256(geometry_checksums).hexdigest(),
            "all_checks_passed": True,
        }
        attributes_record = {
            **common,
            "package_kind": ATTRIBUTES_KIND,
            "protected_attributes_sha256": file_sha256(attributes_path),
            "content_checksums_sha256": sha256(attributes_checksums).hexdigest(),
            "all_checks_passed": True,
        }
        _write(geometry_staging / "package_record.json", json.dumps(geometry_record, indent=2, sort_keys=True).encode() + b"\n")
        _write(attributes_staging / "package_record.json", json.dumps(attributes_record, indent=2, sort_keys=True).encode() + b"\n")
        verify_ds1_geometry_package(geometry_staging)
        verify_ds1_attributes_package(attributes_staging)
        os.replace(geometry_staging, geometry_output_root)
        os.replace(attributes_staging, attributes_output_root)
        return {"geometry": geometry_record, "attributes": attributes_record}
    except Exception:
        shutil.rmtree(geometry_staging, ignore_errors=True)
        shutil.rmtree(attributes_staging, ignore_errors=True)
        raise


def _checksums(root: Path) -> dict[str, str]:
    rows = {}
    for line in (root / "content_checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in rows:
            raise ValueError("DS1 reference checksum manifest contains duplicates")
        rows[relative] = digest
    return rows


def verify_ds1_geometry_package(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("DS1 geometry package is missing or unsafe")
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    contract = DS1_SCREEN_CONTRACT
    for key, expected in {"package_kind": GEOMETRY_KIND, "cohort_id": contract.cohort_id, "release_id": contract.release_id, "membership_sha256": contract.membership_sha256, "scene_count": contract.scene_count, "views_per_scene": contract.views_per_scene, "image_count": contract.image_count, "asset_group_count": contract.asset_group_count}.items():
        if record.get(key) != expected:
            raise ValueError(f"DS1 geometry package mismatch: {key}")
    rows = read_jsonl(root / "reference_manifest.jsonl")
    if len(rows) != int(record["object_view_count"]):
        raise ValueError("DS1 geometry manifest count differs from its record")
    checksums = _checksums(root)
    expected_paths = {"reference_manifest.jsonl", *(f"masks/{row['mask_path']}" for row in rows)}
    if set(checksums) != expected_paths:
        raise ValueError("DS1 geometry checksum allowlist differs")
    keys: set[tuple[str, str, str]] = set()
    images: Counter[tuple[str, str]] = Counter()
    visible = 0
    for row in rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]), str(row["opaque_object_id"]))
        if key in keys:
            raise ValueError("DS1 geometry contains duplicate object views")
        keys.add(key)
        images[key[:2]] += 1
        path = root / "masks" / str(row["mask_path"])
        digest = file_sha256(path)
        if digest != str(row["mask_sha256"]) or checksums[f"masks/{row['mask_path']}"] != digest:
            raise ValueError("DS1 supplied-mask checksum mismatch")
        box, foreground = _mask_geometry(path.read_bytes(), int(record["mask_threshold"]))
        if box != row["box_xyxy"] or foreground != int(row["foreground_pixels"]) or bool(row["visible"]) != (box is not None):
            raise ValueError("DS1 stored geometry differs from its supplied mask")
        visible += int(box is not None)
    scene_ids = {scene_id for scene_id, _ in images}
    expected_views = {f"V{index:02d}" for index in range(contract.views_per_scene)}
    if membership_sha256(scene_ids) != contract.membership_sha256:
        raise ValueError("DS1 geometry rows identify another cohort")
    if any(
        {view_id for candidate, view_id in images if candidate == scene_id}
        != expected_views
        for scene_id in scene_ids
    ):
        raise ValueError("DS1 geometry rows have incomplete view coverage")
    if len(images) != contract.image_count or visible != int(record["visible_object_view_count"]):
        raise ValueError("DS1 geometry coverage differs from its record")
    for relative, digest in checksums.items():
        if file_sha256(root / relative) != digest:
            raise ValueError("DS1 geometry content checksum mismatch")
    if file_sha256(root / "reference_manifest.jsonl") != record["reference_manifest_sha256"] or file_sha256(root / "content_checksums.sha256") != record["content_checksums_sha256"]:
        raise ValueError("DS1 geometry record hash mismatch")
    return record


def verify_ds1_attributes_package(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("DS1 attribute package is missing or unsafe")
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    contract = DS1_SCREEN_CONTRACT
    for key, expected in {"package_kind": ATTRIBUTES_KIND, "cohort_id": contract.cohort_id, "release_id": contract.release_id, "membership_sha256": contract.membership_sha256, "scene_count": contract.scene_count, "asset_group_count": contract.asset_group_count}.items():
        if record.get(key) != expected:
            raise ValueError(f"DS1 attribute package mismatch: {key}")
    rows = read_jsonl(root / "protected_scene_attributes.jsonl")
    scenes = {str(row["opaque_scene_id"]) for row in rows}
    assets = {str(row["opaque_asset_group_id"]) for row in rows}
    if len(rows) != contract.scene_count or membership_sha256(scenes) != contract.membership_sha256 or len(assets) != contract.asset_group_count:
        raise ValueError("DS1 protected attribute coverage differs")
    checksums = _checksums(root)
    if checksums != {"protected_scene_attributes.jsonl": file_sha256(root / "protected_scene_attributes.jsonl")}:
        raise ValueError("DS1 attribute checksum manifest differs")
    if file_sha256(root / "protected_scene_attributes.jsonl") != record["protected_attributes_sha256"] or file_sha256(root / "content_checksums.sha256") != record["content_checksums_sha256"]:
        raise ValueError("DS1 attribute record hash mismatch")
    return record


__all__ = ["build_ds1_reference_packages", "verify_ds1_attributes_package", "verify_ds1_geometry_package"]
