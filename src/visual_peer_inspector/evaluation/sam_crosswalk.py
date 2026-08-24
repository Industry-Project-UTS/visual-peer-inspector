"""Build the protected bridge from SAM prompt keys to evaluator identities."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.data.sam_prompt_package import verify_sam_prompt_package


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def build_sam_evaluator_crosswalk(
    *,
    prompt_root: Path,
    geometry_manifest_path: Path,
    proposal_results_path: Path,
    proposal_matches_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create a private prompt-to-reference bridge for the later evaluator."""

    prompt_record = verify_sam_prompt_package(prompt_root)
    prompt_rows = read_jsonl(prompt_root / "sam_prompt_manifest.jsonl")
    geometry: dict[tuple[str, str, tuple[int, ...]], str] = {}
    for row in read_jsonl(geometry_manifest_path):
        if not bool(row["visible"]):
            continue
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            tuple(int(value) for value in row["box_xyxy"]),
        )
        if key in geometry:
            raise ValueError("Reference geometry contains a duplicate visible box")
        geometry[key] = str(row["opaque_object_id"])

    proposals: dict[tuple[str, str, tuple[int, ...]], int] = {}
    for row in read_jsonl(proposal_results_path):
        image_key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        for proposal_index, box in enumerate(row["boxes_xyxy"]):
            key = (*image_key, tuple(int(value) for value in box))
            if key in proposals:
                raise ValueError("Frozen proposals contain a duplicate box")
            proposals[key] = proposal_index

    matches: dict[tuple[str, str, int], tuple[str, float]] = {}
    for row in read_jsonl(proposal_matches_path):
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            int(row["proposal_index"]),
        )
        if key in matches:
            raise ValueError("Proposal matches contain a duplicate proposal")
        matches[key] = (str(row["opaque_object_id"]), float(row["box_iou"]))

    crosswalk_rows: list[dict[str, Any]] = []
    for image in prompt_rows:
        scene_id = str(image["opaque_scene_id"])
        view_id = str(image["view_id"])
        for prompt in image["routes"]["reference_box"]:
            box = tuple(int(value) for value in prompt["box_xyxy"])
            object_id = geometry.get((scene_id, view_id, box))
            if object_id is None:
                raise ValueError("A reference prompt has no protected object identity")
            crosswalk_rows.append(
                {
                    "schema_version": 1,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "route": "reference_box",
                    "prompt_key": str(prompt["prompt_key"]),
                    "prompt_box_xyxy": list(box),
                    "opaque_object_id": object_id,
                    "proposal_index": None,
                    "matched_opaque_object_id": object_id,
                    "matched_box_iou": 1.0,
                }
            )
        for prompt in image["routes"]["automatic_box"]:
            box = tuple(int(value) for value in prompt["box_xyxy"])
            proposal_index = proposals.get((scene_id, view_id, box))
            if proposal_index is None:
                raise ValueError("An automatic prompt has no frozen proposal identity")
            matched = matches.get((scene_id, view_id, proposal_index))
            crosswalk_rows.append(
                {
                    "schema_version": 1,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "route": "automatic_box",
                    "prompt_key": str(prompt["prompt_key"]),
                    "prompt_box_xyxy": list(box),
                    "opaque_object_id": None,
                    "proposal_index": proposal_index,
                    "matched_opaque_object_id": matched[0] if matched else None,
                    "matched_box_iou": matched[1] if matched else None,
                }
            )

    expected_count = int(prompt_record["reference_prompt_count"]) + int(
        prompt_record["automatic_prompt_count"]
    )
    if len(crosswalk_rows) != expected_count:
        raise ValueError("SAM evaluator crosswalk coverage is incomplete")
    keys = {
        (row["opaque_scene_id"], row["view_id"], row["route"], row["prompt_key"])
        for row in crosswalk_rows
    }
    if len(keys) != len(crosswalk_rows):
        raise ValueError("SAM evaluator crosswalk contains duplicate prompt keys")

    output_root = output_root.expanduser().resolve()
    crosswalk_bytes = _jsonl_bytes(crosswalk_rows)
    crosswalk_path = output_root / "sam_evaluator_crosswalk.jsonl"
    _write_atomic(crosswalk_path, crosswalk_bytes)
    checksum_path = output_root / "content_checksums.sha256"
    _write_atomic(
        checksum_path,
        f"{sha256(crosswalk_bytes).hexdigest()}  {crosswalk_path.name}\n".encode(
            "utf-8"
        ),
    )
    record = {
        "schema_version": 1,
        "package_kind": "ds0-sam-evaluator-crosswalk",
        "prompt_package_sha256": file_sha256(prompt_root / "package_record.json"),
        "geometry_manifest_sha256": file_sha256(geometry_manifest_path),
        "proposal_results_sha256": file_sha256(proposal_results_path),
        "proposal_matches_sha256": file_sha256(proposal_matches_path),
        "row_count": len(crosswalk_rows),
        "reference_prompt_count": int(prompt_record["reference_prompt_count"]),
        "automatic_prompt_count": int(prompt_record["automatic_prompt_count"]),
        "crosswalk_sha256": sha256(crosswalk_bytes).hexdigest(),
        "content_checksums_sha256": file_sha256(checksum_path),
    }
    _write_atomic(
        output_root / "package_record.json",
        json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    verify_sam_evaluator_crosswalk(output_root)
    return record


def verify_sam_evaluator_crosswalk(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    if record.get("package_kind") != "ds0-sam-evaluator-crosswalk":
        raise ValueError("Expected the protected DS0 SAM evaluator crosswalk")
    rows = read_jsonl(root / "sam_evaluator_crosswalk.jsonl")
    keys = [
        (row["opaque_scene_id"], row["view_id"], row["route"], row["prompt_key"])
        for row in rows
    ]
    if len(rows) != int(record["row_count"]) or len(keys) != len(set(keys)):
        raise ValueError("SAM evaluator crosswalk coverage is invalid")
    if file_sha256(root / "sam_evaluator_crosswalk.jsonl") != record["crosswalk_sha256"]:
        raise ValueError("SAM evaluator crosswalk hash differs from its record")
    if file_sha256(root / "content_checksums.sha256") != record["content_checksums_sha256"]:
        raise ValueError("SAM evaluator checksum hash differs from its record")
    return record


__all__ = ["build_sam_evaluator_crosswalk", "verify_sam_evaluator_crosswalk"]
