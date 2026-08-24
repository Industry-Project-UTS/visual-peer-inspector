"""Model-safe mask routes emitted by the DS0 segmentation evaluator.

The package contains no anomaly labels, categories, source identifiers,
evaluator matches or protected attributes. It preserves every route candidate
without exposing how automatic proposals were matched during evaluation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.model_safe_package import (
    SMOKE20_COHORT_ID,
    SMOKE20_MEMBERSHIP_SHA256,
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.models.sam2 import binary_mask_sha256, decode_binary_mask


MASK_ROUTE_SCHEMA_VERSION = 1
MASK_ROUTE_PACKAGE_KIND = "ds0-model-safe-mask-routes"
MASK_ROUTE_INTENDED_USE = "model_facing_representation"
EXPECTED_SCENES = 20
EXPECTED_IMAGES = 400
EXPECTED_ROUTE_COUNTS = {
    "supplied_mask": 1_640,
    "reference_box": 1_640,
    "automatic_box": 1_616,
}
EXPECTED_PROVENANCE_KEYS = {
    "evaluator_snapshot_sha256",
    "sam_run_record_sha256",
    "sam_prompt_package_sha256",
    "proposal_results_sha256",
    "supplied_mask_record_sha256",
}
SCENE_PATTERN = re.compile(r"^S\d{6}$")
VIEW_PATTERN = re.compile(r"^V\d{2}$")
PROMPT_PATTERN = re.compile(r"^[RP]\d{3}$")
_FORBIDDEN_KEY_PARTS = (
    "anomaly",
    "category",
    "label",
    "matched",
    "object_id",
    "asset_group",
    "crosswalk",
    "source_",
)
_FORBIDDEN_VALUE_PARTS = (
    "/users/",
    "/kaggle/input/",
    "labels_by_object",
    "protected_scene",
    "sam_evaluator_crosswalk",
)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Mask-route package file already exists: {path}")
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _safe_value(value: Any, location: str = "package") -> None:
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        lowered = value.lower()
        if any(part in lowered for part in _FORBIDDEN_VALUE_PARTS):
            raise ValueError(f"Protected value found at {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Mask-route key at {location} must be a string")
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"Protected key found at {location}.{key}")
            _safe_value(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _safe_value(item, f"{location}[{index}]")
        return
    raise TypeError(f"Unsupported mask-route value at {location}")


def _safe_relative(value: Any) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or "\\" in text:
        raise ValueError("Mask-route package contains an unsafe path")
    return Path(*pure.parts)


def _content_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Mask-route package cannot contain symbolic links")
        if path.is_file() and path.name not in {
            "package_record.json",
            "content_checksums.sha256",
        }:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def _checksum_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        f"{row['sha256']}  {row['path']}\n" for row in rows
    ).encode("utf-8")


def _validate_route_item(
    item: Mapping[str, Any],
    *,
    route: str,
    seen_prompt_keys: set[str],
) -> None:
    required = {
        "prompt_key",
        "box_xyxy",
        "foreground_pixels",
        "mask_status",
        "mask_sha256",
        "mask_rle",
    }
    optional = {"predicted_quality_score"}
    if set(item) - required - optional or not required.issubset(item):
        raise ValueError("Mask-route item does not follow the accepted schema")
    prompt_key = str(item["prompt_key"])
    expected_prefix = "R" if route in {"supplied_mask", "reference_box"} else "P"
    if (
        not PROMPT_PATTERN.fullmatch(prompt_key)
        or not prompt_key.startswith(expected_prefix)
        or prompt_key in seen_prompt_keys
    ):
        raise ValueError("Mask-route prompt identity is invalid")
    seen_prompt_keys.add(prompt_key)
    box = item["box_xyxy"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
    ):
        raise ValueError("Mask-route prompt box is invalid")
    x1, y1, x2, y2 = [int(value) for value in box]
    if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
        raise ValueError("Mask-route prompt box is outside the image")
    mask_rle = item["mask_rle"]
    if not isinstance(mask_rle, Mapping) or mask_rle.get("size") != [256, 256]:
        raise ValueError("Mask-route RLE has unexpected dimensions")
    mask = decode_binary_mask(mask_rle)
    if mask.shape != (256, 256):
        raise ValueError("Mask-route RLE has unexpected dimensions")
    foreground = int(mask.sum())
    if foreground != int(item["foreground_pixels"]):
        raise ValueError("Mask-route foreground count differs from its RLE")
    if binary_mask_sha256(mask) != str(item["mask_sha256"]):
        raise ValueError("Mask-route SHA-256 differs from its RLE")
    expected_status = "empty" if foreground == 0 else "non_empty"
    if item["mask_status"] != expected_status:
        raise ValueError("Mask-route mask status is inconsistent")
    score = item.get("predicted_quality_score")
    if route == "supplied_mask" and score is not None:
        raise ValueError("Supplied masks cannot have model quality scores")
    if route != "supplied_mask" and score is None:
        raise ValueError("Predicted masks require model quality scores")
    if route != "supplied_mask" and (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ValueError("Predicted mask quality score must be finite")


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    if set(provenance) != EXPECTED_PROVENANCE_KEYS:
        raise ValueError("Mask-route provenance schema differs")
    for digest in provenance.values():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Mask-route provenance requires SHA-256 values")


def _validate_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    image_keys: set[tuple[str, str]] = set()
    route_counts: Counter[str] = Counter()
    route_images: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        if set(row) != {
            "schema_version",
            "opaque_scene_id",
            "view_id",
            "image_sha256",
            "routes",
        }:
            raise ValueError("Mask-route image row does not follow the accepted schema")
        if int(row["schema_version"]) != MASK_ROUTE_SCHEMA_VERSION:
            raise ValueError("Mask-route image row has another schema version")
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        if not SCENE_PATTERN.fullmatch(scene_id) or not VIEW_PATTERN.fullmatch(view_id):
            raise ValueError("Mask-route image identity is not opaque")
        key = (scene_id, view_id)
        if key in image_keys:
            raise ValueError("Mask-route package contains a duplicate image")
        image_keys.add(key)
        image_digest = str(row["image_sha256"])
        if len(image_digest) != 64 or any(
            character not in "0123456789abcdef" for character in image_digest
        ):
            raise ValueError("Mask-route image SHA-256 is invalid")
        routes = row["routes"]
        if not isinstance(routes, dict) or set(routes) != set(EXPECTED_ROUTE_COUNTS):
            raise ValueError("Mask-route image row has unexpected routes")
        reference_keys: set[str] = set()
        supplied_keys: set[str] = set()
        for route in EXPECTED_ROUTE_COUNTS:
            items = routes[route]
            if not isinstance(items, list) or not items:
                raise ValueError("Every mask route must retain its image candidates")
            seen: set[str] = set()
            for item in items:
                _validate_route_item(item, route=route, seen_prompt_keys=seen)
            route_counts[route] += len(items)
            route_images[route].add(key)
            if route == "supplied_mask":
                supplied_keys = seen
            elif route == "reference_box":
                reference_keys = seen
        if supplied_keys != reference_keys:
            raise ValueError("Supplied and reference-box routes have different prompts")
    if len(rows) != EXPECTED_IMAGES or len(image_keys) != EXPECTED_IMAGES:
        raise ValueError("Mask-route package image coverage is incomplete")
    if len({scene_id for scene_id, _ in image_keys}) != EXPECTED_SCENES:
        raise ValueError("Mask-route package scene coverage is incomplete")
    if route_counts != Counter(EXPECTED_ROUTE_COUNTS):
        raise ValueError("Mask-route package candidate coverage is incomplete")
    if any(images != image_keys for images in route_images.values()):
        raise ValueError("Every mask route must cover every image")
    return {
        "scene_count": len({scene_id for scene_id, _ in image_keys}),
        "image_count": len(image_keys),
        "route_counts": dict(route_counts),
    }


def build_mask_route_package(
    *,
    rows: Iterable[Mapping[str, Any]],
    output_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a new hash-locked model-facing mask-route package."""

    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Mask-route package already exists: {output_root}")
    staging_root = output_root.with_name(output_root.name + ".incomplete")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(f"Incomplete mask-route package exists: {staging_root}")
    row_list = [dict(row) for row in rows]
    _safe_value(row_list, "mask_route_manifest")
    counts = _validate_rows(row_list)
    safe_provenance = dict(provenance)
    _safe_value(safe_provenance, "provenance")
    _validate_provenance(safe_provenance)
    staging_root.mkdir(parents=True)
    try:
        manifest_bytes = _jsonl_bytes(row_list)
        _write_atomic(staging_root / "mask_route_manifest.jsonl", manifest_bytes)
        policy = {
            "schema_version": MASK_ROUTE_SCHEMA_VERSION,
            "package_kind": MASK_ROUTE_PACKAGE_KIND,
            "intended_use": MASK_ROUTE_INTENDED_USE,
            "model_facing_allowed": True,
            "protected_fields_included": False,
            "content_policy": (
                "label-free routes without semantic classes, raw identifiers, "
                "match metadata or protected attributes"
            ),
            "candidate_policy": "retain every route candidate without trimming",
        }
        _write_atomic(
            staging_root / "usage_policy.json",
            json.dumps(policy, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        content_rows = _content_manifest(staging_root)
        checksum_bytes = _checksum_bytes(content_rows)
        _write_atomic(staging_root / "content_checksums.sha256", checksum_bytes)
        record = {
            "schema_version": MASK_ROUTE_SCHEMA_VERSION,
            "package_kind": MASK_ROUTE_PACKAGE_KIND,
            "intended_use": MASK_ROUTE_INTENDED_USE,
            "model_facing_allowed": True,
            "cohort_id": SMOKE20_COHORT_ID,
            "membership_sha256": SMOKE20_MEMBERSHIP_SHA256,
            **counts,
            "manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "usage_policy_sha256": file_sha256(staging_root / "usage_policy.json"),
            "content_checksums_sha256": sha256(checksum_bytes).hexdigest(),
            "content_file_count": len(content_rows),
            "provenance": safe_provenance,
            "all_checks_passed": True,
        }
        _safe_value(record, "package_record")
        _write_atomic(
            staging_root / "package_record.json",
            json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        verify_mask_route_package(staging_root)
        os.replace(staging_root, output_root)
        return record
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def verify_mask_route_package(root: Path) -> dict[str, Any]:
    """Verify integrity, model-facing policy and label-free route coverage."""

    expanded_root = root.expanduser()
    if expanded_root.is_symlink():
        raise ValueError("Mask-route package root is missing or unsafe")
    root = expanded_root.resolve()
    if not root.is_dir():
        raise ValueError("Mask-route package root is missing or unsafe")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Mask-route package cannot contain symbolic links")
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    _safe_value(record, "package_record")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Mask-route package provenance is missing")
    _validate_provenance(provenance)
    required = {
        "package_kind": MASK_ROUTE_PACKAGE_KIND,
        "intended_use": MASK_ROUTE_INTENDED_USE,
        "model_facing_allowed": True,
        "cohort_id": SMOKE20_COHORT_ID,
        "membership_sha256": SMOKE20_MEMBERSHIP_SHA256,
        "scene_count": EXPECTED_SCENES,
        "image_count": EXPECTED_IMAGES,
        "route_counts": EXPECTED_ROUTE_COUNTS,
        "all_checks_passed": True,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"Mask-route package record differs for {key}")
    policy = json.loads((root / "usage_policy.json").read_text(encoding="utf-8"))
    _safe_value(policy, "usage_policy")
    if (
        policy.get("model_facing_allowed") is not True
        or policy.get("protected_fields_included") is not False
        or policy.get("candidate_policy")
        != "retain every route candidate without trimming"
    ):
        raise ValueError("Mask-route package policy is invalid")
    if file_sha256(root / "usage_policy.json") != record["usage_policy_sha256"]:
        raise ValueError("Mask-route usage policy hash differs")
    rows = read_jsonl(root / "mask_route_manifest.jsonl")
    _safe_value(rows, "mask_route_manifest")
    counts = _validate_rows(rows)
    if any(record.get(key) != value for key, value in counts.items()):
        raise ValueError("Mask-route package counts differ from its manifest")
    if file_sha256(root / "mask_route_manifest.jsonl") != record["manifest_sha256"]:
        raise ValueError("Mask-route manifest hash differs")

    checksum_path = root / "content_checksums.sha256"
    checksum_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative_text = line.split("  ", 1)
        relative = _safe_relative(relative_text)
        if relative.as_posix() in seen_paths:
            raise ValueError("Mask-route checksum manifest has duplicate paths")
        seen_paths.add(relative.as_posix())
        path = root / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError("Mask-route content checksum differs")
        checksum_rows.append(
            {
                "path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    expected_rows = _content_manifest(root)
    if checksum_rows != expected_rows:
        raise ValueError("Mask-route checksum allowlist differs from package files")
    if len(expected_rows) != int(record["content_file_count"]):
        raise ValueError("Mask-route content file count differs")
    checksum_bytes = _checksum_bytes(checksum_rows)
    if sha256(checksum_bytes).hexdigest() != record["content_checksums_sha256"]:
        raise ValueError("Mask-route checksum manifest hash differs")
    return record


__all__ = [
    "MASK_ROUTE_INTENDED_USE",
    "MASK_ROUTE_PACKAGE_KIND",
    "build_mask_route_package",
    "verify_mask_route_package",
]
