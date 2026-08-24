"""Build and verify the model-safe prompt package for DS0 SAM inference."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model_safe_package import (
    SMOKE20_COHORT_ID,
    SMOKE20_MEMBERSHIP_SHA256,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
)


SAM_PROMPT_SCHEMA_VERSION = 1
ALLOWED_ROUTE_NAMES = {"reference_box", "automatic_box"}
FORBIDDEN_FIELD_FRAGMENTS = {
    "anomaly",
    "asset",
    "category",
    "crosswalk",
    "label",
    "mask",
    "object_id",
    "raw_response",
    "source",
}


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_if_consistent(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"Existing SAM prompt output differs: {path}")
        return
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _normalise_box(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("SAM prompts require four-element XYXY boxes")
    box = [int(item) for item in value]
    if box != value:
        raise ValueError("SAM prompt coordinates must be integers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
        raise ValueError(f"SAM prompt box is outside the image: {box}")
    return box


def _assert_model_safe(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
                raise ValueError(f"Forbidden model-facing field at {path}.{key}")
            _assert_model_safe(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_model_safe(child, f"{path}[{index}]")


def _geometric_prompts(boxes: Iterable[Any], prefix: str) -> list[dict[str, Any]]:
    normalised = sorted(_normalise_box(value) for value in boxes)
    if len({tuple(value) for value in normalised}) != len(normalised):
        raise ValueError("Duplicate boxes are not allowed in a SAM prompt route")
    return [
        {"prompt_key": f"{prefix}{index:03d}", "box_xyxy": box}
        for index, box in enumerate(normalised)
    ]


def build_sam_prompt_package(
    *,
    model_manifest_path: Path,
    media_root: Path,
    geometry_manifest_path: Path,
    proposal_results_path: Path,
    proposal_decision_record_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create opaque prompts without copying reference masks or evaluator fields."""

    model_rows = read_jsonl(model_manifest_path)
    selected_ids = {str(row["opaque_scene_id"]) for row in model_rows}
    if membership_sha256(selected_ids) != SMOKE20_MEMBERSHIP_SHA256:
        raise ValueError("SAM inputs do not match the frozen DS0 membership")
    expected_images = {
        (str(row["opaque_scene_id"]), str(view_id))
        for row in model_rows
        for view_id in row["view_ids"]
    }
    if len(expected_images) != 400:
        raise ValueError("SAM requires all 400 DS0 images")

    decision = json.loads(proposal_decision_record_path.read_text(encoding="utf-8"))
    if decision.get("proceed_to_ds0_sam") is not True:
        raise ValueError("The frozen proposal evaluator did not approve DS0 SAM")
    proposal_results_digest = file_sha256(proposal_results_path)
    if str(decision.get("proposal_results_sha256")) != proposal_results_digest:
        raise ValueError("Proposal decision record identifies another result file")

    reference_by_image: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for row in read_jsonl(geometry_manifest_path):
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key not in expected_images:
            raise ValueError("Geometry manifest contains an unexpected DS0 image")
        if bool(row["visible"]):
            reference_by_image[key].append(_normalise_box(row["box_xyxy"]))
    if set(reference_by_image) != expected_images:
        raise ValueError("Every DS0 image must contain at least one visible reference box")

    proposal_rows = read_jsonl(proposal_results_path)
    proposal_by_image: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in proposal_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in proposal_by_image:
            raise ValueError("Frozen proposals contain a duplicate image")
        if key not in expected_images or str(row.get("status")) != "ok":
            raise ValueError("Frozen proposal coverage or status is invalid")
        if int(row["proposal_count"]) != len(row["boxes_xyxy"]):
            raise ValueError("Frozen proposal count disagrees with its boxes")
        proposal_by_image[key] = row
    if set(proposal_by_image) != expected_images:
        raise ValueError("Frozen proposals do not cover all DS0 images")
    model_manifest_digest = file_sha256(model_manifest_path)
    if {str(row["input_manifest_sha256"]) for row in proposal_rows} != {
        model_manifest_digest
    }:
        raise ValueError("Frozen proposals identify another model-input manifest")
    if sorted({str(row["configuration_sha256"]) for row in proposal_rows}) != sorted(
        str(value) for value in decision["proposal_configuration_sha256"]
    ):
        raise ValueError("Frozen proposal configuration differs from the decision")
    if sorted({str(row["model_id"]) for row in proposal_rows}) != sorted(
        str(value) for value in decision["proposal_model_ids"]
    ):
        raise ValueError("Frozen proposal model differs from the decision")
    if sorted({str(row["model_revision"]) for row in proposal_rows}) != sorted(
        str(value) for value in decision["proposal_model_revisions"]
    ):
        raise ValueError("Frozen proposal revision differs from the decision")

    media_root = media_root.expanduser().resolve()
    manifest_rows: list[dict[str, Any]] = []
    reference_prompt_count = 0
    automatic_prompt_count = 0
    for scene_id, view_id in sorted(expected_images):
        image_relative = Path(scene_id) / "rgb" / f"{view_id}.png"
        image_path = media_root / image_relative
        image_digest = file_sha256(image_path)
        proposal_row = proposal_by_image[(scene_id, view_id)]
        if image_digest != str(proposal_row["image_sha256"]):
            raise ValueError("SAM image hash differs from the frozen proposal input")
        reference_prompts = _geometric_prompts(
            reference_by_image[(scene_id, view_id)], "R"
        )
        automatic_prompts = _geometric_prompts(proposal_row["boxes_xyxy"], "P")
        reference_prompt_count += len(reference_prompts)
        automatic_prompt_count += len(automatic_prompts)
        manifest_rows.append(
            {
                "schema_version": SAM_PROMPT_SCHEMA_VERSION,
                "cohort_id": SMOKE20_COHORT_ID,
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "image_path": image_relative.as_posix(),
                "image_sha256": image_digest,
                "routes": {
                    "reference_box": reference_prompts,
                    "automatic_box": automatic_prompts,
                },
            }
        )

    _assert_model_safe(manifest_rows)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _jsonl_bytes(manifest_rows)
    manifest_path = output_root / "sam_prompt_manifest.jsonl"
    _write_if_consistent(manifest_path, manifest_bytes)
    decision_summary = {
        "schema_version": SAM_PROMPT_SCHEMA_VERSION,
        "decision": "proceed_to_ds0_sam",
        "decision_value": True,
        "evaluation_scope": str(decision["evaluation_scope"]),
        "proposal_results_sha256": proposal_results_digest,
        "proposal_decision_record_sha256": file_sha256(proposal_decision_record_path),
        "proposal_configuration_sha256": decision["proposal_configuration_sha256"],
        "proposal_model_ids": decision["proposal_model_ids"],
        "proposal_model_revisions": decision["proposal_model_revisions"],
    }
    _assert_model_safe(decision_summary)
    decision_bytes = json.dumps(decision_summary, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    decision_path = output_root / "upstream_decision.json"
    _write_if_consistent(decision_path, decision_bytes)
    checksum_lines = "".join(
        f"{file_sha256(path)}  {path.name}\n"
        for path in (manifest_path, decision_path)
    )
    checksum_path = output_root / "content_checksums.sha256"
    _write_if_consistent(checksum_path, checksum_lines.encode("utf-8"))
    record = {
        "schema_version": SAM_PROMPT_SCHEMA_VERSION,
        "package_kind": "ds0-sam-prompts",
        "cohort_id": SMOKE20_COHORT_ID,
        "membership_sha256": membership_sha256(selected_ids),
        "image_count": len(manifest_rows),
        "reference_prompt_count": reference_prompt_count,
        "automatic_prompt_count": automatic_prompt_count,
        "model_manifest_sha256": model_manifest_digest,
        "prompt_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "upstream_decision_sha256": sha256(decision_bytes).hexdigest(),
        "content_checksums_sha256": file_sha256(checksum_path),
        "all_checks_passed": True,
    }
    _write_if_consistent(
        output_root / "package_record.json",
        json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    verify_sam_prompt_package(output_root)
    return record


def verify_sam_prompt_package(root: Path) -> dict[str, Any]:
    """Fail closed on prompt-package coverage, hashes and private fields."""

    root = root.expanduser().resolve()
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    if record.get("package_kind") != "ds0-sam-prompts":
        raise ValueError("Expected the DS0 SAM prompt package")
    if record.get("membership_sha256") != SMOKE20_MEMBERSHIP_SHA256:
        raise ValueError("SAM prompt membership differs from frozen DS0")
    rows = read_jsonl(root / "sam_prompt_manifest.jsonl")
    _assert_model_safe(rows)
    keys: set[tuple[str, str]] = set()
    reference_count = 0
    automatic_count = 0
    for row in rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in keys:
            raise ValueError("SAM prompt manifest contains duplicate images")
        keys.add(key)
        if set(row["routes"]) != ALLOWED_ROUTE_NAMES:
            raise ValueError("SAM prompt manifest contains an unexpected route")
        for route_name, expected_prefix in (
            ("reference_box", "R"),
            ("automatic_box", "P"),
        ):
            prompts = row["routes"][route_name]
            identifiers = []
            boxes = []
            for prompt in prompts:
                identifiers.append(str(prompt["prompt_key"]))
                boxes.append(_normalise_box(prompt["box_xyxy"]))
            expected_identifiers = [
                f"{expected_prefix}{index:03d}" for index in range(len(prompts))
            ]
            if identifiers != expected_identifiers or boxes != sorted(boxes):
                raise ValueError("SAM prompts are not in canonical geometric order")
        reference_count += len(row["routes"]["reference_box"])
        automatic_count += len(row["routes"]["automatic_box"])
    if len(rows) != int(record["image_count"]) or len(keys) != 400:
        raise ValueError("SAM prompt package does not cover 400 unique images")
    if reference_count != int(record["reference_prompt_count"]):
        raise ValueError("Reference prompt count differs from the package record")
    if automatic_count != int(record["automatic_prompt_count"]):
        raise ValueError("Automatic prompt count differs from the package record")
    if file_sha256(root / "sam_prompt_manifest.jsonl") != record["prompt_manifest_sha256"]:
        raise ValueError("SAM prompt manifest hash differs from the package record")
    if file_sha256(root / "upstream_decision.json") != record["upstream_decision_sha256"]:
        raise ValueError("SAM upstream decision hash differs from the package record")
    if file_sha256(root / "content_checksums.sha256") != record["content_checksums_sha256"]:
        raise ValueError("SAM prompt checksum hash differs from the package record")
    return record


__all__ = [
    "SAM_PROMPT_SCHEMA_VERSION",
    "build_sam_prompt_package",
    "verify_sam_prompt_package",
]
