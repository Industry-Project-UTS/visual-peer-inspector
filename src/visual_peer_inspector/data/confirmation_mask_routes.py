"""Label-free supplied, reference-box, and automatic mask routes for DS2."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.evaluation.confirmation_reference import (
    verify_ds2_geometry_package,
)
from visual_peer_inspector.models.sam2 import (
    binary_mask_sha256,
    decode_binary_mask,
    encode_binary_mask,
)


ROUTES = ("supplied_mask", "reference_box", "automatic_box")
PACKAGE_KIND = "ds2-confirmation-mask-routes"
ALLOWED_FILES = {"mask_routes.jsonl", "package_record.json", "content_checksums.sha256"}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _load_supplied_item(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    mask_path = root / "masks" / str(row["mask_path"])
    if file_sha256(mask_path) != str(row["mask_sha256"]):
        raise ValueError("A supplied DS2 mask differs from its trusted digest")
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
    if int(mask.sum()) != int(row["foreground_pixels"]):
        raise ValueError("A supplied DS2 mask has inconsistent foreground geometry")
    object_id = str(row["opaque_object_id"])
    return {
        "prompt_key": object_id,
        "peer_key": object_id,
        "box_xyxy": [int(value) for value in row["box_xyxy"]],
        "predicted_quality_score": None,
        "foreground_pixels": int(mask.sum()),
        "mask_status": "non_empty",
        "mask_sha256": binary_mask_sha256(mask),
        "mask_rle": encode_binary_mask(mask),
    }


def _normalise_sam_item(item: Mapping[str, Any]) -> dict[str, Any]:
    mask = decode_binary_mask(item["mask_rle"])
    if binary_mask_sha256(mask) != str(item["mask_sha256"]):
        raise ValueError("A DS2 SAM result differs from its mask digest")
    object_id = str(item["prompt_key"])
    return {
        "prompt_key": object_id,
        "peer_key": object_id,
        "box_xyxy": [int(value) for value in item["box_xyxy"]],
        "predicted_quality_score": float(item["predicted_quality_score"]),
        "foreground_pixels": int(item["foreground_pixels"]),
        "mask_status": str(item["mask_status"]),
        "mask_sha256": str(item["mask_sha256"]),
        "mask_rle": dict(item["mask_rle"]),
    }


def build_ds2_mask_route_package(
    *,
    rgb_root: Path,
    geometry_root: Path,
    sam_results_path: Path,
    automatic_sam_results_path: Path | None = None,
    output_root: Path,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    """Join all frozen mask routes while keeping automatic peers label-free."""

    rgb_root = rgb_root.expanduser().resolve()
    geometry_root = geometry_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    rgb_audit = verify_model_input_package(rgb_root, contract=DS2_CONFIRM_CONTRACT)
    geometry_record = verify_ds2_geometry_package(geometry_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"DS2 mask-route output already exists: {output_root}")
    if not provenance or any(len(str(value)) != 64 for value in provenance.values()):
        raise ValueError("DS2 mask-route provenance requires SHA-256 values")

    geometry_by_image: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(geometry_root / "reference_manifest.jsonl"):
        if bool(row["visible"]):
            geometry_by_image[(str(row["opaque_scene_id"]), str(row["view_id"]))].append(row)

    sam_by_image: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in read_jsonl(sam_results_path):
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in sam_by_image:
            raise ValueError("DS2 SAM results contain a duplicate image")
        if set(row.get("routes", {})) != {"reference_box"}:
            raise ValueError("DS2 reference SAM file must contain only reference-box results")
        sam_by_image[key] = row

    if automatic_sam_results_path is None:
        raise ValueError("DS2 three-route packaging requires automatic SAM results")
    automatic_sam_results_path = automatic_sam_results_path.expanduser().resolve()
    automatic_by_image: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in read_jsonl(automatic_sam_results_path):
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in automatic_by_image:
            raise ValueError("DS2 automatic SAM results contain a duplicate image")
        if set(row.get("routes", {})) != {"automatic_box"}:
            raise ValueError("DS2 automatic SAM file must contain only automatic-box results")
        automatic_by_image[key] = row

    rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    for scene in read_jsonl(rgb_root / "neutral_corpus_manifest.jsonl"):
        scene_id = str(scene["opaque_scene_id"])
        for view_id in map(str, scene["view_ids"]):
            key = (scene_id, view_id)
            supplied_rows = sorted(
                geometry_by_image.get(key, ()),
                key=lambda value: str(value["opaque_object_id"]),
            )
            sam_row = sam_by_image.get(key)
            automatic_row = automatic_by_image.get(key)
            if not supplied_rows or sam_row is None or automatic_row is None:
                raise ValueError("DS2 mask routes do not cover every RGB image")
            supplied = [_load_supplied_item(geometry_root, row) for row in supplied_rows]
            reference = sorted(
                (_normalise_sam_item(item) for item in sam_row["routes"]["reference_box"]),
                key=lambda value: value["prompt_key"],
            )
            expected_ids = [item["prompt_key"] for item in supplied]
            if [item["prompt_key"] for item in reference] != expected_ids:
                raise ValueError("DS2 SAM prompts do not match the supplied object identities")
            automatic = sorted(
                (
                    _normalise_sam_item(item)
                    for item in automatic_row["routes"]["automatic_box"]
                ),
                key=lambda value: value["prompt_key"],
            )
            automatic_ids = [str(item["prompt_key"]) for item in automatic]
            if (
                len(automatic_ids) != len(set(automatic_ids))
                or any(not value.startswith("P") for value in automatic_ids)
                or set(automatic_ids).intersection(expected_ids)
            ):
                raise ValueError("DS2 automatic SAM prompts expose or duplicate reference identities")
            image_path = Path("media") / scene_id / "rgb" / f"{view_id}.png"
            image_digest = file_sha256(rgb_root / image_path)
            if (
                image_digest != str(sam_row["image_sha256"])
                or image_digest != str(automatic_row["image_sha256"])
            ):
                raise ValueError("DS2 SAM and RGB package identify different images")
            route_counts.update(
                {
                    "supplied_mask": len(supplied),
                    "reference_box": len(reference),
                    "automatic_box": len(automatic),
                }
            )
            rows.append(
                {
                    "schema_version": 1,
                    "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "image_path": image_path.as_posix(),
                    "image_sha256": image_digest,
                    "routes": {
                        "supplied_mask": supplied,
                        "reference_box": reference,
                        "automatic_box": automatic,
                    },
                }
            )

    if len(rows) != DS2_CONFIRM_CONTRACT.image_count:
        raise ValueError("DS2 mask routes must cover all 6,000 images")
    output_root.mkdir(parents=True)
    manifest_path = output_root / "mask_routes.jsonl"
    _write_jsonl(manifest_path, rows)
    checksum_line = f"{file_sha256(manifest_path)}  mask_routes.jsonl\n"
    checksum_path = output_root / "content_checksums.sha256"
    checksum_path.write_text(checksum_line, encoding="utf-8")
    record = {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "release_id": DS2_CONFIRM_CONTRACT.release_id,
        "membership_sha256": DS2_CONFIRM_CONTRACT.membership_sha256,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "image_count": DS2_CONFIRM_CONTRACT.image_count,
        "views_per_scene": DS2_CONFIRM_CONTRACT.views_per_scene,
        "asset_group_count": DS2_CONFIRM_CONTRACT.asset_group_count,
        "route_counts": dict(route_counts),
        "rgb_manifest_sha256": rgb_audit.manifest_sha256,
        "geometry_manifest_sha256": geometry_record["reference_manifest_sha256"],
        "sam_results_sha256": file_sha256(sam_results_path),
        "reference_sam_results_sha256": file_sha256(sam_results_path),
        "automatic_sam_results_sha256": file_sha256(automatic_sam_results_path),
        "mask_routes_sha256": file_sha256(manifest_path),
        "content_checksums_sha256": file_sha256(checksum_path),
        "provenance": dict(provenance),
        "all_checks_passed": True,
    }
    _write_json(output_root / "package_record.json", record)
    return verify_ds2_mask_route_package(output_root)


def verify_ds2_mask_route_package(root: Path) -> dict[str, Any]:
    """Fail closed on route schemas, hashes, identity, and cohort coverage."""

    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("DS2 mask-route package is missing or unsafe")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != ALLOWED_FILES:
        raise ValueError("DS2 mask-route package file allowlist differs")
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    expected = {
        "package_kind": PACKAGE_KIND,
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "release_id": DS2_CONFIRM_CONTRACT.release_id,
        "membership_sha256": DS2_CONFIRM_CONTRACT.membership_sha256,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "image_count": DS2_CONFIRM_CONTRACT.image_count,
        "views_per_scene": DS2_CONFIRM_CONTRACT.views_per_scene,
        "asset_group_count": DS2_CONFIRM_CONTRACT.asset_group_count,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"DS2 mask-route package mismatch: {key}")
    manifest_path = root / "mask_routes.jsonl"
    if file_sha256(manifest_path) != record.get("mask_routes_sha256"):
        raise ValueError("DS2 mask-route manifest digest differs")
    checksum_path = root / "content_checksums.sha256"
    if file_sha256(checksum_path) != record.get("content_checksums_sha256"):
        raise ValueError("DS2 mask-route checksum file differs")
    checksum_text = checksum_path.read_text(encoding="utf-8")
    if checksum_text != f"{file_sha256(manifest_path)}  mask_routes.jsonl\n":
        raise ValueError("DS2 mask-route checksum declaration differs")

    rows = read_jsonl(manifest_path)
    if len(rows) != DS2_CONFIRM_CONTRACT.image_count:
        raise ValueError("DS2 mask-route image coverage differs")
    keys: set[tuple[str, str]] = set()
    scenes: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in keys or set(row.get("routes", {})) != set(ROUTES):
            raise ValueError("DS2 mask-route row identity or route schema is invalid")
        keys.add(key)
        scenes.add(key[0])
        for route in ROUTES:
            items = row["routes"][route]
            ids = [str(item["peer_key"]) for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError("Every DS2 route requires unique peer keys")
            if route != "automatic_box" and len(ids) < 3:
                raise ValueError("Supplied and reference DS2 routes require at least three peers")
            if route == "automatic_box" and any(not value.startswith("P") for value in ids):
                raise ValueError("Automatic DS2 peers require proposal-local keys")
            if any(str(item["prompt_key"]) != str(item["peer_key"]) for item in items):
                raise ValueError("DS2 mask routes must preserve their route-local peer keys")
            for item in items:
                mask = decode_binary_mask(item["mask_rle"])
                if binary_mask_sha256(mask) != str(item["mask_sha256"]):
                    raise ValueError("DS2 mask-route RLE digest differs")
                if int(mask.sum()) != int(item["foreground_pixels"]):
                    raise ValueError("DS2 mask-route foreground count differs")
            counts[route] += len(items)
        if [item["peer_key"] for item in row["routes"]["supplied_mask"]] != [
            item["peer_key"] for item in row["routes"]["reference_box"]
        ]:
            raise ValueError("DS2 supplied and assisted routes are not aligned")
    if membership_sha256(scenes) != DS2_CONFIRM_CONTRACT.membership_sha256:
        raise ValueError("DS2 mask-route membership differs")
    if dict(counts) != record.get("route_counts"):
        raise ValueError("DS2 mask-route counts differ from the package record")
    return record


__all__ = [
    "ROUTES",
    "build_ds2_mask_route_package",
    "verify_ds2_mask_route_package",
]
