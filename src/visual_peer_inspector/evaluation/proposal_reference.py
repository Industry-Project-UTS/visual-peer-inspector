"""Build the protected DS0 reference package used only by evaluators.

The model path never imports this module. It joins the frozen opaque DS0 IDs
to the supplied object masks and protected development attributes after model
inference has finished.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

from PIL import Image

from visual_peer_inspector.data.model_safe_package import (
    EXPECTED_ARCHIVE_SIZE_BYTES,
    SMOKE20_COHORT_ID,
    SMOKE20_MEMBERSHIP_SHA256,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
)
from visual_peer_inspector.data.archive_members import IndexedMember, read_indexed_member


DS0_REFERENCE_SCHEMA_VERSION = 1
DEFAULT_MASK_THRESHOLD = 128
EXPECTED_DS0_SCENES = 20
EXPECTED_DS0_VIEWS = 20


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_if_consistent(path: Path, data: bytes) -> bool:
    """Atomically create a file, or verify an existing resume file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"Existing evaluator output differs: {path}")
        return False
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return True


def _mask_geometry(data: bytes, threshold: int) -> tuple[list[int] | None, int]:
    """Return a half-open XYXY box and foreground area from one supplied mask."""

    with Image.open(BytesIO(data)) as opened:
        opened.load()
        if opened.size != (256, 256):
            raise ValueError(f"Unexpected supplied-mask size: {opened.size}")
        grey = opened.convert("L")
        binary = grey.point(lambda value: 255 if value >= threshold else 0, mode="L")
        bounds = binary.getbbox()
        histogram = binary.histogram()
        foreground_pixels = int(histogram[255])
    if bounds is None:
        if foreground_pixels != 0:
            raise ValueError("Empty mask bounds disagree with foreground area")
        return None, 0
    x1, y1, x2, y2 = [int(value) for value in bounds]
    if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
        raise ValueError("Derived reference box is outside the image")
    return [x1, y1, x2, y2], foreground_pixels


def _content_checksum_bytes(root: Path, relative_paths: Iterable[Path]) -> bytes:
    lines = [
        f"{file_sha256(root / relative)}  {relative.as_posix()}\n"
        for relative in sorted(set(relative_paths), key=lambda item: item.as_posix())
    ]
    return "".join(lines).encode("utf-8")


def _load_indexed_members(
    database_path: Path, names: Iterable[str]
) -> dict[str, IndexedMember]:
    requested = sorted(set(names))
    records: dict[str, IndexedMember] = {}
    with sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True) as connection:
        for offset in range(0, len(requested), 500):
            chunk = requested[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                "SELECT name, compressed_size, uncompressed_size, crc32, "
                "compression_method, flags, local_header_offset, is_directory, "
                "is_symlink, unsafe_reason FROM members WHERE name IN "
                f"({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                name = str(row[0])
                if name in records:
                    raise ValueError(f"Archive index contains a duplicate member: {name}")
                records[name] = IndexedMember(
                    name=name,
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
    if set(records) != set(requested):
        missing = sorted(set(requested) - set(records))
        raise ValueError(f"Archive index is missing required members: {missing[:3]}")
    return records


def build_ds0_reference_package(
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
    """Create a resumable private mask and oracle-box package for all DS0 views."""

    if not 1 <= int(mask_threshold) <= 255:
        raise ValueError("Mask threshold must be between 1 and 255")
    model_rows = read_jsonl(model_manifest_path)
    if len(model_rows) != EXPECTED_DS0_SCENES:
        raise ValueError(f"Expected {EXPECTED_DS0_SCENES} DS0 scenes")
    selected_ids = {str(row["opaque_scene_id"]) for row in model_rows}
    if membership_sha256(selected_ids) != SMOKE20_MEMBERSHIP_SHA256:
        raise ValueError("Evaluator input does not match the frozen DS0 membership")
    for row in model_rows:
        view_ids = [str(value) for value in row["view_ids"]]
        if len(view_ids) != EXPECTED_DS0_VIEWS or len(set(view_ids)) != len(view_ids):
            raise ValueError("Every DS0 scene must contain 20 unique views")

    source_rows = read_jsonl(development_source_map_path)
    source_by_scene = {
        str(row["opaque_scene_id"]): row
        for row in source_rows
        if str(row.get("opaque_scene_id")) in selected_ids
    }
    label_rows = read_jsonl(development_labels_path)
    labels_by_scene = {
        str(row["opaque_scene_id"]): row
        for row in label_rows
        if str(row.get("opaque_scene_id")) in selected_ids
    }
    if set(source_by_scene) != selected_ids or set(labels_by_scene) != selected_ids:
        raise ValueError("Protected development records do not cover the DS0 scenes")

    required_rows = [
        row
        for row in read_jsonl(required_source_members_path)
        if str(row.get("opaque_scene_id")) in selected_ids
        and row.get("component") == "object_mask"
        and row.get("include_in_media_package") is True
    ]
    required_keys = [
        (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        )
        for row in required_rows
    ]
    if len(required_keys) != len(set(required_keys)):
        raise ValueError("Protected mask records contain duplicate object-view keys")

    expected_object_views = sum(
        int(row["object_count"]) * EXPECTED_DS0_VIEWS for row in model_rows
    )
    if len(required_rows) != expected_object_views:
        raise ValueError(
            f"Expected {expected_object_views} DS0 object masks, found {len(required_rows)}"
        )

    source_archive_path = source_archive_path.expanduser().resolve()
    if source_archive_path.stat().st_size != EXPECTED_ARCHIVE_SIZE_BYTES:
        raise ValueError("Source archive size differs from the accepted ToysAD archive")
    archive_index_path = archive_index_path.expanduser().resolve()
    indexed_members = _load_indexed_members(
        archive_index_path, (str(row["source_member"]) for row in required_rows)
    )
    geometry_output_root = geometry_output_root.expanduser().resolve()
    attributes_output_root = attributes_output_root.expanduser().resolve()
    if geometry_output_root == attributes_output_root:
        raise ValueError("Geometry and evaluator attributes require separate roots")
    geometry_output_root.mkdir(parents=True, exist_ok=True)
    attributes_output_root.mkdir(parents=True, exist_ok=True)

    reference_rows: list[dict[str, Any]] = []
    relative_mask_paths: list[Path] = []
    visible_count = 0
    with source_archive_path.open("rb") as archive:
        for row in sorted(
            required_rows,
            key=lambda item: (
                str(item["opaque_scene_id"]),
                str(item["view_id"]),
                str(item["opaque_object_id"]),
            ),
        ):
            scene_id = str(row["opaque_scene_id"])
            view_id = str(row["view_id"])
            object_id = str(row["opaque_object_id"])
            source_member = str(row["source_member"])
            member = indexed_members[source_member]
            expected_crc = int(str(row["crc32"]), 16)
            expected_size = int(row["uncompressed_size"])
            if member.crc32 != expected_crc or member.uncompressed_size != expected_size:
                raise ValueError("Archive member metadata differs from the protected record")
            data = read_indexed_member(archive, member)
            box, foreground_pixels = _mask_geometry(data, int(mask_threshold))
            relative_mask = Path("masks") / scene_id / object_id / f"{view_id}.png"
            _write_if_consistent(geometry_output_root / relative_mask, data)
            relative_mask_paths.append(relative_mask)
            visible = box is not None
            visible_count += int(visible)
            reference_rows.append(
                {
                    "schema_version": DS0_REFERENCE_SCHEMA_VERSION,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "opaque_object_id": object_id,
                    "mask_path": relative_mask.as_posix(),
                    "mask_sha256": sha256(data).hexdigest(),
                    "foreground_pixels": foreground_pixels,
                    "visible": visible,
                    "box_xyxy": box,
                }
            )
    protected_rows: list[dict[str, Any]] = []
    for scene_id in sorted(selected_ids):
        source = source_by_scene[scene_id]
        labels = labels_by_scene[scene_id]
        labels_by_object = {
            str(key): int(value)
            for key, value in dict(labels["labels_by_object_id"]).items()
        }
        expected_ids = {
            str(value)
            for value in next(
                row["object_ids"] for row in model_rows if row["opaque_scene_id"] == scene_id
            )
        }
        if set(labels_by_object) != expected_ids:
            raise ValueError("Protected labels do not match the opaque object IDs")
        protected_rows.append(
            {
                "schema_version": DS0_REFERENCE_SCHEMA_VERSION,
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": str(source["opaque_asset_group_id"]),
                "category": str(source["category"]),
                "object_count": int(source["object_count"]),
                "anomaly_count": int(source["anomaly_count"]),
                "anomaly_families": sorted(str(value) for value in source["raw_anomaly_prefixes"]),
                "labels_by_object_id": labels_by_object,
            }
        )

    manifest_relative = Path("reference_manifest.jsonl")
    _write_if_consistent(
        geometry_output_root / manifest_relative, _jsonl_bytes(reference_rows)
    )
    checksum_relative = Path("content_checksums.sha256")
    geometry_checksum_bytes = _content_checksum_bytes(
        geometry_output_root, [manifest_relative, *relative_mask_paths]
    )
    _write_if_consistent(
        geometry_output_root / checksum_relative, geometry_checksum_bytes
    )
    geometry_record = {
        "schema_version": DS0_REFERENCE_SCHEMA_VERSION,
        "package_kind": "ds0-oracle-geometry",
        "cohort_id": SMOKE20_COHORT_ID,
        "membership_sha256": membership_sha256(selected_ids),
        "scene_count": EXPECTED_DS0_SCENES,
        "view_count": EXPECTED_DS0_VIEWS,
        "image_count": EXPECTED_DS0_SCENES * EXPECTED_DS0_VIEWS,
        "object_view_count": len(reference_rows),
        "visible_object_view_count": visible_count,
        "empty_object_view_count": len(reference_rows) - visible_count,
        "mask_threshold": int(mask_threshold),
        "archive_index_sha256": file_sha256(archive_index_path),
        "reference_manifest_sha256": file_sha256(
            geometry_output_root / manifest_relative
        ),
        "content_checksums_sha256": sha256(geometry_checksum_bytes).hexdigest(),
        "all_checks_passed": True,
    }
    _write_if_consistent(
        geometry_output_root / "package_record.json",
        json.dumps(geometry_record, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )

    protected_relative = Path("protected_scene_attributes.jsonl")
    _write_if_consistent(
        attributes_output_root / protected_relative, _jsonl_bytes(protected_rows)
    )
    attributes_checksum_bytes = _content_checksum_bytes(
        attributes_output_root, [protected_relative]
    )
    _write_if_consistent(
        attributes_output_root / checksum_relative, attributes_checksum_bytes
    )
    attributes_record = {
        "schema_version": DS0_REFERENCE_SCHEMA_VERSION,
        "package_kind": "ds0-evaluator-attributes",
        "cohort_id": SMOKE20_COHORT_ID,
        "membership_sha256": membership_sha256(selected_ids),
        "scene_count": EXPECTED_DS0_SCENES,
        "protected_attributes_sha256": file_sha256(
            attributes_output_root / protected_relative
        ),
        "content_checksums_sha256": sha256(attributes_checksum_bytes).hexdigest(),
        "all_checks_passed": True,
    }
    _write_if_consistent(
        attributes_output_root / "package_record.json",
        json.dumps(attributes_record, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )
    verify_ds0_geometry_package(geometry_output_root)
    verify_ds0_attributes_package(attributes_output_root)
    return {"geometry": geometry_record, "attributes": attributes_record}


def verify_ds0_geometry_package(root: Path) -> dict[str, Any]:
    """Verify every private DS0 reference mask, box, manifest and checksum."""

    root = root.expanduser().resolve()
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    if record.get("package_kind") != "ds0-oracle-geometry":
        raise ValueError("Expected the DS0 oracle geometry package")
    if record.get("cohort_id") != SMOKE20_COHORT_ID:
        raise ValueError("Reference package identifies another cohort")
    if record.get("membership_sha256") != SMOKE20_MEMBERSHIP_SHA256:
        raise ValueError("Reference package membership differs from frozen DS0")
    threshold = int(record["mask_threshold"])
    reference_rows = read_jsonl(root / "reference_manifest.jsonl")
    if len(reference_rows) != int(record["object_view_count"]):
        raise ValueError("Reference manifest count differs from the package record")

    checksums: dict[str, str] = {}
    for line in (root / "content_checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in checksums:
            raise ValueError("Reference checksum manifest has duplicate paths")
        checksums[relative] = digest
    expected_paths = {
        "reference_manifest.jsonl",
        *(str(row["mask_path"]) for row in reference_rows),
    }
    if set(checksums) != expected_paths:
        raise ValueError("Reference checksum paths do not match the package")

    visible_count = 0
    image_keys: Counter[tuple[str, str]] = Counter()
    object_view_keys: set[tuple[str, str, str]] = set()
    for row in reference_rows:
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        )
        if key in object_view_keys:
            raise ValueError("Reference manifest contains duplicate object views")
        object_view_keys.add(key)
        image_keys[key[:2]] += 1
        relative = str(row["mask_path"])
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Reference mask is missing or unsafe")
        if file_sha256(path) != str(row["mask_sha256"]):
            raise ValueError("Reference mask hash differs from its row")
        if checksums[relative] != str(row["mask_sha256"]):
            raise ValueError("Reference mask checksum sources disagree")
        box, foreground_pixels = _mask_geometry(path.read_bytes(), threshold)
        if box != row["box_xyxy"] or foreground_pixels != int(row["foreground_pixels"]):
            raise ValueError("Stored reference geometry differs from the mask")
        if bool(row["visible"]) != (box is not None):
            raise ValueError("Stored visibility differs from the mask")
        visible_count += int(box is not None)

    if len(image_keys) != int(record["image_count"]):
        raise ValueError("Reference masks do not cover every DS0 image")
    if visible_count != int(record["visible_object_view_count"]):
        raise ValueError("Visible mask count differs from the package record")
    for relative, expected_digest in checksums.items():
        if file_sha256(root / relative) != expected_digest:
            raise ValueError("Reference package content checksum mismatch")
    if file_sha256(root / "reference_manifest.jsonl") != record["reference_manifest_sha256"]:
        raise ValueError("Reference manifest hash differs from the package record")
    if file_sha256(root / "content_checksums.sha256") != record["content_checksums_sha256"]:
        raise ValueError("Reference checksum hash differs from the package record")
    return record


def verify_ds0_attributes_package(root: Path) -> dict[str, Any]:
    """Verify the separate evaluator-only attributes package."""

    root = root.expanduser().resolve()
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    if record.get("package_kind") != "ds0-evaluator-attributes":
        raise ValueError("Expected the DS0 evaluator attributes package")
    if record.get("cohort_id") != SMOKE20_COHORT_ID:
        raise ValueError("Evaluator attributes identify another cohort")
    if record.get("membership_sha256") != SMOKE20_MEMBERSHIP_SHA256:
        raise ValueError("Evaluator attribute membership differs from frozen DS0")
    rows = read_jsonl(root / "protected_scene_attributes.jsonl")
    if len(rows) != int(record["scene_count"]):
        raise ValueError("Protected scene count differs from the package record")
    scene_ids = [str(row["opaque_scene_id"]) for row in rows]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Protected attributes contain duplicate scenes")
    checksums = (root / "content_checksums.sha256").read_text(encoding="utf-8")
    expected_line = (
        f"{file_sha256(root / 'protected_scene_attributes.jsonl')}  "
        "protected_scene_attributes.jsonl\n"
    )
    if checksums != expected_line:
        raise ValueError("Evaluator attribute checksum manifest differs")
    if file_sha256(root / "protected_scene_attributes.jsonl") != record["protected_attributes_sha256"]:
        raise ValueError("Protected attribute hash differs from the package record")
    if file_sha256(root / "content_checksums.sha256") != record["content_checksums_sha256"]:
        raise ValueError("Evaluator attribute checksum hash differs")
    return record


__all__ = [
    "DEFAULT_MASK_THRESHOLD",
    "DS0_REFERENCE_SCHEMA_VERSION",
    "build_ds0_reference_package",
    "verify_ds0_attributes_package",
    "verify_ds0_geometry_package",
]
