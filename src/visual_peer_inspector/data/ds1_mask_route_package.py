"""Label-free three-route mask package for downstream DS1 representation."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import canonical_json_bytes, file_sha256, membership_sha256, read_jsonl
from visual_peer_inspector.models.sam2 import binary_mask_sha256, decode_binary_mask


PACKAGE_KIND = "toysad8k-ds1-model-safe-mask-routes"
ROUTES = ("supplied_mask", "reference_box", "automatic_box")
_FORBIDDEN = ("anomaly", "asset", "category", "crosswalk", "label", "matched", "object", "role", "source")


def _check_safe(value: Any, location: str = "package") -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if any(part in str(key).lower() for part in _FORBIDDEN):
                raise ValueError(f"Protected model-facing field at {location}.{key}")
            _check_safe(child, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _check_safe(child, f"{location}[{index}]")
        return
    raise TypeError(f"Unsupported model-facing value at {location}")


def _validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contract = DS1_SCREEN_CONTRACT
    if len(rows) != contract.image_count:
        raise ValueError("DS1 mask routes must contain all 2,000 images")
    keys: set[tuple[str, str]] = set()
    counts = {route: 0 for route in ROUTES}
    automatic_alignment = {"aligned": 0, "unaligned": 0}
    for row in rows:
        if set(row) != {"schema_version", "opaque_scene_id", "view_id", "image_sha256", "routes"}:
            raise ValueError("DS1 mask-route image schema is invalid")
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in keys or not re.fullmatch(r"S\d{6}", key[0]) or not re.fullmatch(r"V\d{2}", key[1]):
            raise ValueError("DS1 mask-route image identity is invalid")
        keys.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["image_sha256"])):
            raise ValueError("DS1 mask-route image SHA-256 is invalid")
        if set(row["routes"]) != set(ROUTES):
            raise ValueError("DS1 mask-route package must preserve three routes")
        route_prompt_keys: dict[str, set[str]] = {}
        route_prompt_boxes: dict[str, dict[str, tuple[int, ...]]] = {}
        route_peer_keys: dict[str, set[str]] = {}
        for route in ROUTES:
            items = row["routes"][route]
            if not isinstance(items, list):
                raise ValueError("DS1 mask route must be a list")
            prompt_keys: set[str] = set()
            prompt_boxes: dict[str, tuple[int, ...]] = {}
            peer_keys: set[str] = set()
            for item in items:
                if set(item) - {"prompt_key", "peer_key", "box_xyxy", "foreground_pixels", "mask_status", "mask_sha256", "mask_rle", "predicted_quality_score"}:
                    raise ValueError("DS1 mask route item schema is invalid")
                prompt_key = str(item["prompt_key"])
                if prompt_key in prompt_keys:
                    raise ValueError("DS1 mask route contains a duplicate prompt")
                prompt_keys.add(prompt_key)
                raw_peer_key = item.get("peer_key")
                if route in {"supplied_mask", "reference_box"} and raw_peer_key is None:
                    raise ValueError("DS1 reference-derived routes require a peer key")
                if raw_peer_key is not None:
                    peer_key = str(raw_peer_key)
                    if not re.fullmatch(r"R\d{3}", peer_key) or peer_key in peer_keys:
                        raise ValueError("DS1 mask route peer identity is invalid")
                    peer_keys.add(peer_key)
                    if route in {"supplied_mask", "reference_box"} and peer_key != prompt_key:
                        raise ValueError(
                            "DS1 reference-derived prompt and peer identities differ"
                        )
                if route == "automatic_box":
                    automatic_alignment[
                        "aligned" if raw_peer_key is not None else "unaligned"
                    ] += 1
                box = tuple(int(value) for value in item["box_xyxy"])
                if len(box) != 4 or list(box) != item["box_xyxy"]:
                    raise ValueError("DS1 mask route contains an invalid prompt box")
                prompt_boxes[prompt_key] = box
                mask = decode_binary_mask(item["mask_rle"])
                if mask.shape != (256, 256) or binary_mask_sha256(mask) != item["mask_sha256"]:
                    raise ValueError("DS1 mask route RLE differs from its digest")
                if int(mask.sum()) != int(item["foreground_pixels"]):
                    raise ValueError("DS1 mask route foreground count is inconsistent")
                counts[route] += 1
            route_prompt_keys[route] = prompt_keys
            route_prompt_boxes[route] = prompt_boxes
            route_peer_keys[route] = peer_keys
        if route_prompt_keys["supplied_mask"] != route_prompt_keys["reference_box"]:
            raise ValueError("DS1 supplied and reference mask routes are not aligned")
        if any(
            route_prompt_boxes["supplied_mask"][key]
            != route_prompt_boxes["reference_box"][key]
            for key in route_prompt_keys["supplied_mask"]
        ):
            raise ValueError("DS1 supplied and reference prompt boxes differ")
        if route_peer_keys["supplied_mask"] != route_peer_keys["reference_box"]:
            raise ValueError("DS1 supplied and reference peer identities differ")
        if not route_peer_keys["automatic_box"].issubset(
            route_peer_keys["reference_box"]
        ):
            raise ValueError("DS1 automatic route has an unknown peer identity")
    scenes = {scene for scene, _ in keys}
    if len(scenes) != contract.scene_count or membership_sha256(scenes) != contract.membership_sha256:
        raise ValueError("DS1 mask-route membership differs from the contract")
    _check_safe(rows)
    return {
        "scene_count": len(scenes),
        "image_count": len(keys),
        "route_counts": counts,
        "automatic_alignment": automatic_alignment,
    }


def build_ds1_mask_route_package(*, rows: Iterable[Mapping[str, Any]], output_root: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Write and verify the model-facing DS1 mask routes."""

    materialized = [dict(row) for row in rows]
    coverage = _validate(materialized)
    _check_safe(provenance, "provenance")
    if not provenance:
        raise ValueError("DS1 mask-route provenance cannot be empty")
    for key, value in provenance.items():
        digest = str(value)
        if not str(key).endswith("_sha256") or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("DS1 mask-route provenance must contain SHA-256 values only")
    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"DS1 mask-route output already exists: {output_root}")
    output_root.mkdir(parents=True)
    manifest = output_root / "mask_routes.jsonl"
    manifest.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in materialized))
    record = {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "intended_use": "model_facing_representation",
        "model_facing_allowed": True,
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
        **coverage,
        "manifest_sha256": file_sha256(manifest),
        "provenance": dict(provenance),
        "all_checks_passed": True,
    }
    (output_root / "package_record.json").write_bytes(canonical_json_bytes(record) + b"\n")
    verify_ds1_mask_route_package(output_root)
    return record


def verify_ds1_mask_route_package(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("DS1 mask-route package root is missing or unsafe")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("DS1 mask-route package cannot contain symbolic links")
    actual_files = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if actual_files != {"mask_routes.jsonl", "package_record.json"}:
        raise ValueError("DS1 mask-route package contains an unexpected file")
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    if record.get("package_kind") != PACKAGE_KIND or record.get("model_facing_allowed") is not True:
        raise ValueError("DS1 mask-route package declaration is invalid")
    rows = read_jsonl(root / "mask_routes.jsonl")
    coverage = _validate(rows)
    if record.get("manifest_sha256") != file_sha256(root / "mask_routes.jsonl"):
        raise ValueError("DS1 mask-route manifest SHA-256 mismatch")
    if coverage != {key: record[key] for key in coverage} or record.get("all_checks_passed") is not True:
        raise ValueError("DS1 mask-route package coverage differs from its record")
    return record


__all__ = ["build_ds1_mask_route_package", "verify_ds1_mask_route_package"]
