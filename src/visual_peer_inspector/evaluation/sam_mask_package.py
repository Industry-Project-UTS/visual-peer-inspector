"""Build the private DS0 package used to evaluate SAM masks.

This package contains supplied reference masks and protected evaluator metadata.
It must never be mounted by a model-facing notebook or uploaded to ClearML.
"""

from __future__ import annotations

from binascii import crc32
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.archive_members import IndexedMember, read_indexed_member
from visual_peer_inspector.data.model_safe_package import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZE_BYTES,
    SMOKE20_COHORT_ID,
    SMOKE20_MEMBERSHIP_SHA256,
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.evaluation.proposal_reference import (
    verify_ds0_attributes_package,
    verify_ds0_geometry_package,
)
from visual_peer_inspector.evaluation.sam_crosswalk import (
    verify_sam_evaluator_crosswalk,
)


MASK_EVALUATOR_SCHEMA_VERSION = 1
MASK_EVALUATOR_PACKAGE_KIND = "ds0-sam-mask-evaluator"
EVALUATOR_ONLY_USE = "private_evaluation_only"
EXPECTED_DS0_SCENES = 20
EXPECTED_DS0_VIEWS_PER_SCENE = 20
EXPECTED_DS0_IMAGES = 400
EXPECTED_VISIBLE_MASKS = 1_640
EXPECTED_REFERENCE_PROMPTS = 1_640
EXPECTED_AUTOMATIC_PROMPTS = 1_616
EXPECTED_SAM_CROSSWALK_ROWS = 3_256

_UPSTREAM_FILES = {
    "oracle_geometry": (
        "package_record.json",
        "content_checksums.sha256",
        "reference_manifest.jsonl",
    ),
    "sam_crosswalk": (
        "package_record.json",
        "content_checksums.sha256",
        "sam_evaluator_crosswalk.jsonl",
    ),
    "scene_attributes": (
        "package_record.json",
        "content_checksums.sha256",
        "protected_scene_attributes.jsonl",
    ),
}


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_atomic(path: Path, data: bytes) -> None:
    """Write one new file through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Private package output already exists: {path}")
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Incomplete private package file already exists: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _safe_relative_path(value: Any, *, expected_prefix: str | None = None) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if not text or text.startswith("/") or ".." in pure.parts or "\\" in text:
        raise ValueError(f"Unsafe private package path: {text}")
    path = Path(*pure.parts)
    if expected_prefix is not None and (not pure.parts or pure.parts[0] != expected_prefix):
        raise ValueError(f"Private package path must start with {expected_prefix}/")
    return path


def _indexed_member(row: tuple[Any, ...]) -> IndexedMember:
    return IndexedMember(
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


def _load_archive_candidates(
    database_path: Path,
    fingerprints: set[tuple[int, int]],
) -> dict[tuple[int, int], list[IndexedMember]]:
    """Scan the index once for members matching a frozen mask fingerprint."""

    candidates: dict[tuple[int, int], list[IndexedMember]] = defaultdict(list)
    query = (
        "SELECT name, compressed_size, uncompressed_size, crc32, "
        "compression_method, flags, local_header_offset, is_directory, "
        "is_symlink, unsafe_reason FROM members"
    )
    with sqlite3.connect(
        f"file:{database_path.expanduser().resolve()}?mode=ro", uri=True
    ) as connection:
        for row in connection.execute(query):
            member = _indexed_member(row)
            fingerprint = (member.crc32, member.uncompressed_size)
            if fingerprint in fingerprints:
                candidates[fingerprint].append(member)
    return candidates


def _read_package_record(root: Path) -> dict[str, Any]:
    value = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Package record must be an object: {root}")
    return value


def _private_input_state(
    *,
    geometry_root: Path,
    crosswalk_root: Path,
    attributes_root: Path,
) -> dict[str, Any]:
    """Validate the three frozen private inputs and their shared DS0 coverage."""

    geometry_root = geometry_root.expanduser().resolve()
    crosswalk_root = crosswalk_root.expanduser().resolve()
    attributes_root = attributes_root.expanduser().resolve()
    geometry_record = verify_ds0_geometry_package(geometry_root)
    crosswalk_record = verify_sam_evaluator_crosswalk(crosswalk_root)
    attributes_record = verify_ds0_attributes_package(attributes_root)

    if int(geometry_record.get("scene_count", -1)) != EXPECTED_DS0_SCENES:
        raise ValueError(f"Expected {EXPECTED_DS0_SCENES} DS0 scenes")
    if int(geometry_record.get("view_count", -1)) != EXPECTED_DS0_VIEWS_PER_SCENE:
        raise ValueError(f"Expected {EXPECTED_DS0_VIEWS_PER_SCENE} views per scene")
    if int(geometry_record.get("image_count", -1)) != EXPECTED_DS0_IMAGES:
        raise ValueError(f"Expected {EXPECTED_DS0_IMAGES} DS0 images")
    if int(geometry_record.get("object_view_count", -1)) != EXPECTED_VISIBLE_MASKS:
        raise ValueError(f"Expected {EXPECTED_VISIBLE_MASKS} supplied masks")
    if int(geometry_record.get("visible_object_view_count", -1)) != EXPECTED_VISIBLE_MASKS:
        raise ValueError(f"Expected {EXPECTED_VISIBLE_MASKS} visible supplied masks")
    if int(geometry_record.get("empty_object_view_count", -1)) != 0:
        raise ValueError("DS0 contains an unexpected empty supplied mask")
    if int(crosswalk_record.get("row_count", -1)) != EXPECTED_SAM_CROSSWALK_ROWS:
        raise ValueError(f"Expected {EXPECTED_SAM_CROSSWALK_ROWS} SAM crosswalk rows")
    if int(crosswalk_record.get("reference_prompt_count", -1)) != EXPECTED_REFERENCE_PROMPTS:
        raise ValueError(f"Expected {EXPECTED_REFERENCE_PROMPTS} reference prompts")
    if int(crosswalk_record.get("automatic_prompt_count", -1)) != EXPECTED_AUTOMATIC_PROMPTS:
        raise ValueError(f"Expected {EXPECTED_AUTOMATIC_PROMPTS} automatic prompts")
    if int(attributes_record.get("scene_count", -1)) != EXPECTED_DS0_SCENES:
        raise ValueError("Protected attributes do not cover all DS0 scenes")

    geometry_rows = read_jsonl(geometry_root / "reference_manifest.jsonl")
    geometry_keys: set[tuple[str, str, str]] = set()
    geometry_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    image_keys: set[tuple[str, str]] = set()
    scene_views: dict[str, set[str]] = defaultdict(set)
    scene_objects: dict[str, set[str]] = defaultdict(set)
    for row in geometry_rows:
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        object_id = str(row["opaque_object_id"])
        key = (scene_id, view_id, object_id)
        if key in geometry_keys:
            raise ValueError("Oracle geometry contains duplicate object-view keys")
        geometry_keys.add(key)
        geometry_by_key[key] = row
        image_keys.add((scene_id, view_id))
        scene_views[scene_id].add(view_id)
        scene_objects[scene_id].add(object_id)
        if row.get("visible") is not True:
            raise ValueError("Every DS0 evaluator mask must be visible")
        if int(row.get("foreground_pixels", 0)) <= 0 or row.get("box_xyxy") is None:
            raise ValueError("A visible DS0 mask has no foreground geometry")
        relative = _safe_relative_path(row["mask_path"], expected_prefix="masks")
        expected = Path("masks") / scene_id / object_id / f"{view_id}.png"
        if relative != expected:
            raise ValueError("Oracle mask path does not match its opaque identity")

    if len(geometry_rows) != EXPECTED_VISIBLE_MASKS or len(image_keys) != EXPECTED_DS0_IMAGES:
        raise ValueError("Oracle geometry has incomplete DS0 coverage")
    if len(scene_views) != EXPECTED_DS0_SCENES or any(
        len(views) != EXPECTED_DS0_VIEWS_PER_SCENE for views in scene_views.values()
    ):
        raise ValueError("Oracle geometry does not contain the expected scene-view grid")

    attribute_rows = read_jsonl(attributes_root / "protected_scene_attributes.jsonl")
    attributes_by_scene: dict[str, Mapping[str, Any]] = {}
    for row in attribute_rows:
        scene_id = str(row["opaque_scene_id"])
        if scene_id in attributes_by_scene:
            raise ValueError("Protected attributes contain duplicate scenes")
        attributes_by_scene[scene_id] = row
        labels = {str(key) for key in dict(row["labels_by_object_id"])}
        if labels != scene_objects.get(scene_id, set()):
            raise ValueError("Protected labels do not match oracle object identities")
        if int(row["object_count"]) != len(labels):
            raise ValueError("Protected object count differs from its labels")
    if set(attributes_by_scene) != set(scene_views):
        raise ValueError("Protected scene membership differs from oracle geometry")

    crosswalk_rows = read_jsonl(crosswalk_root / "sam_evaluator_crosswalk.jsonl")
    crosswalk_keys: set[tuple[str, str, str, str]] = set()
    route_counts: Counter[str] = Counter()
    route_images: dict[str, set[tuple[str, str]]] = defaultdict(set)
    reference_keys: set[tuple[str, str, str]] = set()
    for row in crosswalk_rows:
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        route = str(row["route"])
        prompt_key = str(row["prompt_key"])
        key = (scene_id, view_id, route, prompt_key)
        if key in crosswalk_keys:
            raise ValueError("SAM crosswalk contains duplicate prompt identities")
        if (scene_id, view_id) not in image_keys:
            raise ValueError("SAM crosswalk contains an image outside frozen DS0")
        crosswalk_keys.add(key)
        route_counts[route] += 1
        route_images[route].add((scene_id, view_id))
        if route == "reference_box":
            object_id = str(row["opaque_object_id"])
            reference_key = (scene_id, view_id, object_id)
            reference = geometry_by_key.get(reference_key)
            if reference is None or row["prompt_box_xyxy"] != reference["box_xyxy"]:
                raise ValueError("A reference SAM prompt differs from oracle geometry")
            if row.get("proposal_index") is not None:
                raise ValueError("A reference SAM prompt cannot identify a proposal")
            if str(row.get("matched_opaque_object_id")) != object_id:
                raise ValueError("A reference SAM prompt must match its oracle object")
            if float(row.get("matched_box_iou", -1.0)) != 1.0:
                raise ValueError("A reference SAM prompt must have box IoU 1.0")
            reference_keys.add(reference_key)
        elif route == "automatic_box":
            if row.get("opaque_object_id") is not None:
                raise ValueError("An automatic SAM prompt cannot expose an oracle identity")
            proposal_index = row.get("proposal_index")
            if (
                not isinstance(proposal_index, int)
                or isinstance(proposal_index, bool)
                or proposal_index < 0
            ):
                raise ValueError("An automatic SAM prompt has an invalid proposal index")
            matched_id = row.get("matched_opaque_object_id")
            matched_iou = row.get("matched_box_iou")
            if (matched_id is None) != (matched_iou is None):
                raise ValueError("Automatic prompt match identity and IoU must be paired")
            if matched_id is not None:
                if str(matched_id) not in scene_objects[scene_id]:
                    raise ValueError("Automatic prompt matches an unknown oracle object")
                if not 0.0 <= float(matched_iou) <= 1.0:
                    raise ValueError("Automatic prompt match IoU is outside [0, 1]")
        else:
            raise ValueError(f"Unexpected SAM crosswalk route: {route}")

    if len(crosswalk_rows) != EXPECTED_SAM_CROSSWALK_ROWS:
        raise ValueError("SAM crosswalk row count is incomplete")
    if route_counts != Counter(
        {
            "reference_box": EXPECTED_REFERENCE_PROMPTS,
            "automatic_box": EXPECTED_AUTOMATIC_PROMPTS,
        }
    ):
        raise ValueError("SAM crosswalk route counts are incomplete")
    if reference_keys != geometry_keys:
        raise ValueError("Reference SAM prompts do not cover every supplied mask")
    if route_images["reference_box"] != image_keys or route_images["automatic_box"] != image_keys:
        raise ValueError("Both SAM prompt routes must cover all DS0 images")

    return {
        "geometry_root": geometry_root,
        "crosswalk_root": crosswalk_root,
        "attributes_root": attributes_root,
        "geometry_record": geometry_record,
        "crosswalk_record": crosswalk_record,
        "attributes_record": attributes_record,
        "geometry_rows": geometry_rows,
        "scene_ids": set(scene_views),
        "image_keys": image_keys,
    }


def _copy_private_metadata(
    *,
    staging_root: Path,
    geometry_root: Path,
    crosswalk_root: Path,
    attributes_root: Path,
) -> list[Path]:
    copied: list[Path] = []
    sources = {
        "oracle_geometry": geometry_root,
        "sam_crosswalk": crosswalk_root,
        "scene_attributes": attributes_root,
    }
    for target_name, source_root in sources.items():
        for filename in _UPSTREAM_FILES[target_name]:
            relative = Path(target_name) / filename
            _write_atomic(staging_root / relative, (source_root / filename).read_bytes())
            copied.append(relative)
    return copied


def _content_checksum_bytes(root: Path, relative_paths: Iterable[Path]) -> bytes:
    lines = [
        f"{file_sha256(root / relative)}  {relative.as_posix()}\n"
        for relative in sorted(set(relative_paths), key=lambda item: item.as_posix())
    ]
    return "".join(lines).encode("utf-8")


def _same_source_record(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = (
        "source_archive_sha256",
        "archive_index_sha256",
        "oracle_geometry_package_record_sha256",
        "sam_crosswalk_package_record_sha256",
        "scene_attributes_package_record_sha256",
    )
    return all(existing.get(key) == expected.get(key) for key in keys)


def build_ds0_mask_evaluator_package(
    *,
    geometry_root: Path,
    crosswalk_root: Path,
    attributes_root: Path,
    source_archive_path: Path,
    archive_index_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Publish an evaluator-only package after exact archive and mask checks."""

    state = _private_input_state(
        geometry_root=geometry_root,
        crosswalk_root=crosswalk_root,
        attributes_root=attributes_root,
    )
    geometry_root = state["geometry_root"]
    crosswalk_root = state["crosswalk_root"]
    attributes_root = state["attributes_root"]
    source_archive_path = source_archive_path.expanduser().resolve()
    archive_index_path = archive_index_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    for private_root in (geometry_root, crosswalk_root, attributes_root):
        if private_root == output_root or private_root in output_root.parents:
            raise ValueError("Evaluator output must be outside every private input root")
    if source_archive_path.is_symlink() or not source_archive_path.is_file():
        raise ValueError("The accepted ToysAD archive is missing or unsafe")
    if archive_index_path.is_symlink() or not archive_index_path.is_file():
        raise ValueError("The ToysAD archive index is missing or unsafe")
    if source_archive_path.stat().st_size != EXPECTED_ARCHIVE_SIZE_BYTES:
        raise ValueError("Source archive size differs from the accepted ToysAD archive")
    source_archive_digest = file_sha256(source_archive_path)
    if source_archive_digest != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Source archive SHA-256 differs from the accepted ToysAD archive")
    archive_index_digest = file_sha256(archive_index_path)
    if archive_index_digest != str(state["geometry_record"]["archive_index_sha256"]):
        raise ValueError("Archive index SHA-256 differs from the frozen oracle package")

    provenance = {
        "source_archive_sha256": source_archive_digest,
        "archive_index_sha256": archive_index_digest,
        "oracle_geometry_package_record_sha256": file_sha256(
            geometry_root / "package_record.json"
        ),
        "sam_crosswalk_package_record_sha256": file_sha256(
            crosswalk_root / "package_record.json"
        ),
        "scene_attributes_package_record_sha256": file_sha256(
            attributes_root / "package_record.json"
        ),
    }
    if output_root.exists() or output_root.is_symlink():
        existing = verify_ds0_mask_evaluator_package(output_root)
        if not _same_source_record(existing, provenance):
            raise ValueError("Existing evaluator package was built from other inputs")
        return existing

    staging_root = output_root.with_name(output_root.name + ".incomplete")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(f"Incomplete evaluator package already exists: {staging_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()
    try:
        metadata_paths = _copy_private_metadata(
            staging_root=staging_root,
            geometry_root=geometry_root,
            crosswalk_root=crosswalk_root,
            attributes_root=attributes_root,
        )

        frozen_masks: list[dict[str, Any]] = []
        fingerprints: set[tuple[int, int]] = set()
        for row in state["geometry_rows"]:
            relative = _safe_relative_path(row["mask_path"], expected_prefix="masks")
            frozen_path = geometry_root / relative
            data = frozen_path.read_bytes()
            expected_digest = str(row["mask_sha256"])
            if sha256(data).hexdigest() != expected_digest:
                raise ValueError("Frozen oracle mask differs from its SHA-256")
            fingerprint = (crc32(data) & 0xFFFFFFFF, len(data))
            fingerprints.add(fingerprint)
            frozen_masks.append(
                {
                    "row": row,
                    "relative": relative,
                    "fingerprint": fingerprint,
                    "expected_sha256": expected_digest,
                }
            )
        candidates = _load_archive_candidates(archive_index_path, fingerprints)

        archive_rows: list[dict[str, Any]] = []
        mask_paths: list[Path] = []
        used_source_members: set[str] = set()
        with source_archive_path.open("rb") as archive:
            for item in sorted(
                frozen_masks,
                key=lambda value: (
                    str(value["row"]["opaque_scene_id"]),
                    str(value["row"]["view_id"]),
                    str(value["row"]["opaque_object_id"]),
                ),
            ):
                matching_members: list[tuple[IndexedMember, bytes]] = []
                for member in sorted(
                    candidates.get(item["fingerprint"], []), key=lambda value: value.name
                ):
                    if "/segmentations/" not in member.name.replace("\\", "/"):
                        continue
                    data = read_indexed_member(archive, member)
                    if sha256(data).hexdigest() == item["expected_sha256"]:
                        matching_members.append((member, data))
                if len(matching_members) != 1:
                    raise ValueError(
                        "A frozen supplied mask does not resolve to exactly one archive member"
                    )
                member, data = matching_members[0]
                if member.name in used_source_members:
                    raise ValueError("One archive member resolves to multiple oracle masks")
                used_source_members.add(member.name)
                package_relative = Path("oracle_geometry") / item["relative"]
                _write_atomic(staging_root / package_relative, data)
                mask_paths.append(package_relative)
                row = item["row"]
                archive_rows.append(
                    {
                        "schema_version": MASK_EVALUATOR_SCHEMA_VERSION,
                        "opaque_scene_id": str(row["opaque_scene_id"]),
                        "view_id": str(row["view_id"]),
                        "opaque_object_id": str(row["opaque_object_id"]),
                        "package_mask_path": package_relative.as_posix(),
                        "source_member": member.name,
                        "mask_sha256": item["expected_sha256"],
                        "crc32": f"{member.crc32:08x}",
                        "compressed_size": member.compressed_size,
                        "uncompressed_size": member.uncompressed_size,
                        "compression_method": member.compression_method,
                        "local_header_offset": member.local_header_offset,
                    }
                )

        archive_manifest_relative = Path("archive_mask_manifest.jsonl")
        _write_atomic(
            staging_root / archive_manifest_relative, _jsonl_bytes(archive_rows)
        )
        usage_policy = {
            "schema_version": MASK_EVALUATOR_SCHEMA_VERSION,
            "package_kind": MASK_EVALUATOR_PACKAGE_KIND,
            "intended_use": EVALUATOR_ONLY_USE,
            "model_facing_allowed": False,
            "contains_supplied_reference_masks": True,
            "contains_protected_scene_attributes": True,
            "contains_private_crosswalk": True,
            "clearml_full_package_upload_allowed": False,
            "allowed_external_evidence": [
                "aggregate_metrics",
                "sanitized_tables",
                "sanitized_figures",
            ],
        }
        usage_relative = Path("usage_policy.json")
        _write_atomic(staging_root / usage_relative, _json_bytes(usage_policy))

        content_paths = [
            *metadata_paths,
            *mask_paths,
            archive_manifest_relative,
            usage_relative,
        ]
        checksum_bytes = _content_checksum_bytes(staging_root, content_paths)
        checksum_relative = Path("content_checksums.sha256")
        _write_atomic(staging_root / checksum_relative, checksum_bytes)
        record = {
            "schema_version": MASK_EVALUATOR_SCHEMA_VERSION,
            "package_kind": MASK_EVALUATOR_PACKAGE_KIND,
            "intended_use": EVALUATOR_ONLY_USE,
            "model_facing_allowed": False,
            "cohort_id": SMOKE20_COHORT_ID,
            "membership_sha256": SMOKE20_MEMBERSHIP_SHA256,
            "scene_count": EXPECTED_DS0_SCENES,
            "view_count_per_scene": EXPECTED_DS0_VIEWS_PER_SCENE,
            "image_count": EXPECTED_DS0_IMAGES,
            "supplied_mask_count": len(mask_paths),
            "visible_mask_count": len(mask_paths),
            "crosswalk_row_count": len(
                read_jsonl(crosswalk_root / "sam_evaluator_crosswalk.jsonl")
            ),
            "reference_prompt_count": int(
                state["crosswalk_record"]["reference_prompt_count"]
            ),
            "automatic_prompt_count": int(
                state["crosswalk_record"]["automatic_prompt_count"]
            ),
            "source_archive_size_bytes": source_archive_path.stat().st_size,
            "archive_mask_manifest_sha256": file_sha256(
                staging_root / archive_manifest_relative
            ),
            "usage_policy_sha256": file_sha256(staging_root / usage_relative),
            "content_file_count": len(content_paths),
            "content_checksums_sha256": sha256(checksum_bytes).hexdigest(),
            **provenance,
            "all_checks_passed": True,
        }
        _write_atomic(staging_root / "package_record.json", _json_bytes(record))
        verify_ds0_mask_evaluator_package(staging_root)
        os.replace(staging_root, output_root)
        return record
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _read_top_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"Malformed private checksum line {line_number}") from error
        safe = _safe_relative_path(relative).as_posix()
        if safe in checksums:
            raise ValueError("Private checksum manifest contains duplicate paths")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Private checksum manifest contains an invalid SHA-256")
        checksums[safe] = digest
    if not checksums:
        raise ValueError("Private checksum manifest is empty")
    return checksums


def verify_ds0_mask_evaluator_package(
    root: Path,
    *,
    intended_use: str = EVALUATOR_ONLY_USE,
) -> dict[str, Any]:
    """Verify package integrity and fail closed for any model-facing use."""

    if intended_use != EVALUATOR_ONLY_USE:
        raise ValueError(
            "The DS0 mask evaluator package is evaluator-only and cannot be used "
            "as model-facing input"
        )
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("DS0 mask evaluator package root is missing or unsafe")
    record = _read_package_record(root)
    if record.get("package_kind") != MASK_EVALUATOR_PACKAGE_KIND:
        raise ValueError("Expected the private DS0 SAM mask evaluator package")
    if record.get("intended_use") != EVALUATOR_ONLY_USE:
        raise ValueError("Private DS0 package has an invalid intended-use declaration")
    if record.get("model_facing_allowed") is not False:
        raise ValueError("Private DS0 package must explicitly reject model-facing use")
    if record.get("cohort_id") != SMOKE20_COHORT_ID:
        raise ValueError("Private DS0 package identifies another cohort")
    if record.get("membership_sha256") != SMOKE20_MEMBERSHIP_SHA256:
        raise ValueError("Private DS0 package membership differs from frozen DS0")
    if record.get("source_archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Private DS0 package identifies another ToysAD archive")
    if int(record.get("source_archive_size_bytes", -1)) != EXPECTED_ARCHIVE_SIZE_BYTES:
        raise ValueError("Private DS0 package records another ToysAD archive size")

    expected_counts = {
        "scene_count": EXPECTED_DS0_SCENES,
        "view_count_per_scene": EXPECTED_DS0_VIEWS_PER_SCENE,
        "image_count": EXPECTED_DS0_IMAGES,
        "supplied_mask_count": EXPECTED_VISIBLE_MASKS,
        "visible_mask_count": EXPECTED_VISIBLE_MASKS,
        "crosswalk_row_count": EXPECTED_SAM_CROSSWALK_ROWS,
        "reference_prompt_count": EXPECTED_REFERENCE_PROMPTS,
        "automatic_prompt_count": EXPECTED_AUTOMATIC_PROMPTS,
    }
    for key, expected in expected_counts.items():
        if int(record.get(key, -1)) != expected:
            raise ValueError(f"Private DS0 package count differs for {key}")

    geometry_root = root / "oracle_geometry"
    crosswalk_root = root / "sam_crosswalk"
    attributes_root = root / "scene_attributes"
    state = _private_input_state(
        geometry_root=geometry_root,
        crosswalk_root=crosswalk_root,
        attributes_root=attributes_root,
    )
    if record.get("archive_index_sha256") != state["geometry_record"].get(
        "archive_index_sha256"
    ):
        raise ValueError("Private DS0 package identifies another archive index")
    if file_sha256(geometry_root / "package_record.json") != record[
        "oracle_geometry_package_record_sha256"
    ]:
        raise ValueError("Embedded oracle package record differs from provenance")
    if file_sha256(crosswalk_root / "package_record.json") != record[
        "sam_crosswalk_package_record_sha256"
    ]:
        raise ValueError("Embedded crosswalk package record differs from provenance")
    if file_sha256(attributes_root / "package_record.json") != record[
        "scene_attributes_package_record_sha256"
    ]:
        raise ValueError("Embedded scene attributes record differs from provenance")

    policy = json.loads((root / "usage_policy.json").read_text(encoding="utf-8"))
    if (
        policy.get("package_kind") != MASK_EVALUATOR_PACKAGE_KIND
        or policy.get("intended_use") != EVALUATOR_ONLY_USE
        or policy.get("model_facing_allowed") is not False
        or policy.get("clearml_full_package_upload_allowed") is not False
    ):
        raise ValueError("Evaluator-only usage policy is invalid")
    if file_sha256(root / "usage_policy.json") != record["usage_policy_sha256"]:
        raise ValueError("Evaluator-only usage policy hash differs from its record")

    archive_rows = read_jsonl(root / "archive_mask_manifest.jsonl")
    if len(archive_rows) != EXPECTED_VISIBLE_MASKS:
        raise ValueError("Archive mask manifest does not contain every supplied mask")
    archive_keys: set[tuple[str, str, str]] = set()
    archive_paths: set[str] = set()
    source_members: set[str] = set()
    geometry_by_key = {
        (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        ): row
        for row in state["geometry_rows"]
    }
    for row in archive_rows:
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        )
        if key in archive_keys or key not in geometry_by_key:
            raise ValueError("Archive mask manifest has an invalid object-view key")
        archive_keys.add(key)
        relative = _safe_relative_path(row["package_mask_path"], expected_prefix="oracle_geometry")
        expected_relative = Path("oracle_geometry") / str(geometry_by_key[key]["mask_path"])
        if relative != expected_relative:
            raise ValueError("Archive mask path differs from oracle geometry")
        relative_text = relative.as_posix()
        if relative_text in archive_paths:
            raise ValueError("Archive mask manifest contains duplicate package paths")
        archive_paths.add(relative_text)
        source_member = str(row["source_member"])
        if (
            source_member in source_members
            or "/segmentations/" not in source_member.replace("\\", "/")
        ):
            raise ValueError("Archive mask manifest has an invalid source member")
        source_members.add(source_member)
        path = root / relative
        expected_digest = str(geometry_by_key[key]["mask_sha256"])
        if str(row["mask_sha256"]) != expected_digest or file_sha256(path) != expected_digest:
            raise ValueError("Extracted supplied mask differs from its exact SHA-256")
        data = path.read_bytes()
        if f"{crc32(data) & 0xFFFFFFFF:08x}" != str(row["crc32"]):
            raise ValueError("Extracted supplied mask differs from its archive CRC")
        if len(data) != int(row["uncompressed_size"]):
            raise ValueError("Extracted supplied mask differs from its archive size")
    if archive_keys != set(geometry_by_key):
        raise ValueError("Archive mask manifest does not cover all oracle masks")
    if file_sha256(root / "archive_mask_manifest.jsonl") != record[
        "archive_mask_manifest_sha256"
    ]:
        raise ValueError("Archive mask manifest hash differs from its record")

    checksums = _read_top_checksums(root / "content_checksums.sha256")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Private DS0 package cannot contain symbolic links")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in {"package_record.json", "content_checksums.sha256"}:
                actual_files.add(relative)
    if set(checksums) != actual_files:
        raise ValueError("Private DS0 package file allowlist differs from checksums")
    if len(actual_files) != int(record["content_file_count"]):
        raise ValueError("Private DS0 content file count differs from its record")
    for relative, expected_digest in checksums.items():
        if file_sha256(root / relative) != expected_digest:
            raise ValueError("Private DS0 package content checksum mismatch")
    if file_sha256(root / "content_checksums.sha256") != record[
        "content_checksums_sha256"
    ]:
        raise ValueError("Private DS0 checksum manifest hash differs from its record")
    if record.get("all_checks_passed") is not True:
        raise ValueError("Private DS0 package record is not marked verified")
    return record


def reject_model_facing_mask_evaluator_package(root: Path) -> None:
    """Provide an explicit guard for loaders that inspect an input package."""

    verify_ds0_mask_evaluator_package(root, intended_use="model_facing")


__all__ = [
    "EVALUATOR_ONLY_USE",
    "MASK_EVALUATOR_PACKAGE_KIND",
    "MASK_EVALUATOR_SCHEMA_VERSION",
    "build_ds0_mask_evaluator_package",
    "reject_model_facing_mask_evaluator_package",
    "verify_ds0_mask_evaluator_package",
]
