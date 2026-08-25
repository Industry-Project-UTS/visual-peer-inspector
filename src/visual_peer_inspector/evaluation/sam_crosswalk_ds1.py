"""Private DS1 prompt-to-reference bridge built after model inference."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import canonical_json_bytes, file_sha256, membership_sha256, read_jsonl
from visual_peer_inspector.data.sam_prompt_package_ds1 import verify_ds1_sam_prompt_package
from visual_peer_inspector.evaluation.proposal_reference_ds1 import verify_ds1_geometry_package
from visual_peer_inspector.evaluation.proposals import evaluate_proposal_dataset


PACKAGE_KIND = "ds1-sam-evaluator-crosswalk"


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"DS1 crosswalk output already exists: {path}")
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def build_ds1_sam_evaluator_crosswalk(
    *,
    prompt_root: Path,
    geometry_root: Path,
    proposal_results_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Join exact DS1 prompts to deterministic evaluator-only box matches."""

    prompt_root = prompt_root.expanduser().resolve()
    geometry_root = geometry_root.expanduser().resolve()
    proposal_results_path = proposal_results_path.expanduser().resolve()
    prompt_record = verify_ds1_sam_prompt_package(prompt_root)
    geometry_record = verify_ds1_geometry_package(geometry_root)
    prompt_rows = read_jsonl(prompt_root / "sam_prompt_manifest.jsonl")
    geometry_rows = read_jsonl(geometry_root / "reference_manifest.jsonl")
    proposal_rows = read_jsonl(proposal_results_path)
    upstream = json.loads((prompt_root / "upstream_decision.json").read_text(encoding="utf-8"))
    if upstream.get("geometry_manifest_sha256") != file_sha256(geometry_root / "reference_manifest.jsonl"):
        raise ValueError("DS1 prompt package identifies another geometry manifest")
    if upstream.get("proposal_results_sha256") != file_sha256(proposal_results_path):
        raise ValueError("DS1 prompt package identifies another proposal result set")

    evaluation = evaluate_proposal_dataset(reference_rows=geometry_rows, proposal_rows=proposal_rows)
    match_by_proposal = {
        (str(row["opaque_scene_id"]), str(row["view_id"]), int(row["proposal_index"])): (str(row["opaque_object_id"]), float(row["box_iou"]))
        for row in evaluation["match_records"]
    }
    geometry_by_box: dict[tuple[str, str, tuple[int, ...]], str] = {}
    for row in geometry_rows:
        if not bool(row["visible"]):
            continue
        key = (str(row["opaque_scene_id"]), str(row["view_id"]), tuple(int(value) for value in row["box_xyxy"]))
        if key in geometry_by_box:
            raise ValueError("DS1 reference geometry contains a duplicate visible box")
        geometry_by_box[key] = str(row["opaque_object_id"])
    proposal_by_box: dict[tuple[str, str, tuple[int, ...]], int] = {}
    for row in proposal_rows:
        scene_id, view_id = str(row["opaque_scene_id"]), str(row["view_id"])
        if row.get("status") != "ok" or int(row["proposal_count"]) != len(row["boxes_xyxy"]):
            raise ValueError("DS1 crosswalk requires complete successful proposals")
        for index, box in enumerate(row["boxes_xyxy"]):
            key = (scene_id, view_id, tuple(int(value) for value in box))
            if key in proposal_by_box:
                raise ValueError("DS1 proposals contain a duplicate image box")
            proposal_by_box[key] = index

    rows: list[dict[str, Any]] = []
    for image in prompt_rows:
        scene_id, view_id = str(image["opaque_scene_id"]), str(image["view_id"])
        for prompt in image["routes"]["reference_box"]:
            box = tuple(int(value) for value in prompt["box_xyxy"])
            object_id = geometry_by_box.get((scene_id, view_id, box))
            if object_id is None:
                raise ValueError("DS1 reference prompt has no evaluator identity")
            rows.append({
                "schema_version": 1,
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "image_sha256": str(image["image_sha256"]),
                "route": "reference_box",
                "prompt_key": str(prompt["prompt_key"]),
                "prompt_box_xyxy": list(box),
                "opaque_object_id": object_id,
                "proposal_index": None,
                "matched_opaque_object_id": object_id,
                "matched_box_iou": 1.0,
            })
        for prompt in image["routes"]["automatic_box"]:
            box = tuple(int(value) for value in prompt["box_xyxy"])
            proposal_index = proposal_by_box.get((scene_id, view_id, box))
            if proposal_index is None:
                raise ValueError("DS1 automatic prompt has no frozen proposal identity")
            matched = match_by_proposal.get((scene_id, view_id, proposal_index))
            rows.append({
                "schema_version": 1,
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "image_sha256": str(image["image_sha256"]),
                "route": "automatic_box",
                "prompt_key": str(prompt["prompt_key"]),
                "prompt_box_xyxy": list(box),
                "opaque_object_id": matched[0] if matched else None,
                "proposal_index": proposal_index,
                "matched_opaque_object_id": matched[0] if matched else None,
                "matched_box_iou": matched[1] if matched else None,
            })
    expected = int(prompt_record["reference_prompt_count"]) + int(prompt_record["automatic_prompt_count"])
    keys = {(row["opaque_scene_id"], row["view_id"], row["route"], row["prompt_key"]) for row in rows}
    if len(rows) != expected or len(keys) != len(rows):
        raise ValueError("DS1 evaluator crosswalk coverage is incomplete")

    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"DS1 evaluator crosswalk output exists: {output_root}")
    output_root.mkdir(parents=True)
    crosswalk_bytes = _jsonl_bytes(rows)
    _write(output_root / "evaluator_crosswalk.jsonl", crosswalk_bytes)
    checksum_bytes = f"{sha256(crosswalk_bytes).hexdigest()}  evaluator_crosswalk.jsonl\n".encode()
    _write(output_root / "content_checksums.sha256", checksum_bytes)
    record = {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "intended_use": "evaluator_only",
        "model_facing_allowed": False,
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "release_id": DS1_SCREEN_CONTRACT.release_id,
        "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
        "scene_count": DS1_SCREEN_CONTRACT.scene_count,
        "image_count": DS1_SCREEN_CONTRACT.image_count,
        "row_count": len(rows),
        "reference_prompt_count": int(prompt_record["reference_prompt_count"]),
        "automatic_prompt_count": int(prompt_record["automatic_prompt_count"]),
        "matched_automatic_prompt_count": sum(row["route"] == "automatic_box" and row["opaque_object_id"] is not None for row in rows),
        "prompt_package_record_sha256": file_sha256(prompt_root / "package_record.json"),
        "prompt_manifest_sha256": prompt_record["prompt_manifest_sha256"],
        "model_input_manifest_sha256": prompt_record["model_manifest_sha256"],
        "geometry_package_record_sha256": file_sha256(geometry_root / "package_record.json"),
        "geometry_manifest_sha256": geometry_record["reference_manifest_sha256"],
        "proposal_results_sha256": file_sha256(proposal_results_path),
        "crosswalk_sha256": sha256(crosswalk_bytes).hexdigest(),
        "content_checksums_sha256": sha256(checksum_bytes).hexdigest(),
        "all_checks_passed": True,
    }
    _write(output_root / "package_record.json", json.dumps(record, indent=2, sort_keys=True).encode() + b"\n")
    verify_ds1_sam_evaluator_crosswalk(output_root)
    return record


def verify_ds1_sam_evaluator_crosswalk(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("DS1 evaluator crosswalk is missing or unsafe")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != {"evaluator_crosswalk.jsonl", "content_checksums.sha256", "package_record.json"}:
        raise ValueError("DS1 evaluator crosswalk file allowlist differs")
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    contract = DS1_SCREEN_CONTRACT
    for key, expected in {"package_kind": PACKAGE_KIND, "intended_use": "evaluator_only", "model_facing_allowed": False, "cohort_id": contract.cohort_id, "release_id": contract.release_id, "membership_sha256": contract.membership_sha256, "scene_count": contract.scene_count, "image_count": contract.image_count, "all_checks_passed": True}.items():
        if record.get(key) != expected:
            raise ValueError(f"DS1 evaluator crosswalk mismatch: {key}")
    rows = read_jsonl(root / "evaluator_crosswalk.jsonl")
    keys = [(row["opaque_scene_id"], row["view_id"], row["route"], row["prompt_key"]) for row in rows]
    if len(rows) != int(record["row_count"]) or len(keys) != len(set(keys)):
        raise ValueError("DS1 evaluator crosswalk row coverage is invalid")
    image_keys = {(str(row["opaque_scene_id"]), str(row["view_id"])) for row in rows}
    image_digests: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        digest = str(row.get("image_sha256", ""))
        if len(digest) != 64 or (
            key in image_digests and image_digests[key] != digest
        ):
            raise ValueError("DS1 evaluator crosswalk RGB identity is invalid")
        image_digests[key] = digest
    scene_ids = {scene_id for scene_id, _ in image_keys}
    expected_views = {f"V{index:02d}" for index in range(contract.views_per_scene)}
    if len(image_keys) != contract.image_count or membership_sha256(scene_ids) != contract.membership_sha256:
        raise ValueError("DS1 evaluator crosswalk identifies another cohort")
    if any(
        {view_id for candidate, view_id in image_keys if candidate == scene_id}
        != expected_views
        for scene_id in scene_ids
    ):
        raise ValueError("DS1 evaluator crosswalk view coverage is incomplete")
    route_counts = {route: sum(row["route"] == route for row in rows) for route in ("reference_box", "automatic_box")}
    if route_counts["reference_box"] != int(record["reference_prompt_count"]) or route_counts["automatic_box"] != int(record["automatic_prompt_count"]):
        raise ValueError("DS1 evaluator crosswalk route counts differ")
    for key in (
        "prompt_package_record_sha256",
        "prompt_manifest_sha256",
        "model_input_manifest_sha256",
        "geometry_package_record_sha256",
        "geometry_manifest_sha256",
        "proposal_results_sha256",
    ):
        if not isinstance(record.get(key), str) or len(record[key]) != 64:
            raise ValueError(f"DS1 evaluator crosswalk digest is invalid: {key}")
    if file_sha256(root / "evaluator_crosswalk.jsonl") != record["crosswalk_sha256"] or file_sha256(root / "content_checksums.sha256") != record["content_checksums_sha256"]:
        raise ValueError("DS1 evaluator crosswalk checksum mismatch")
    expected_line = f"{record['crosswalk_sha256']}  evaluator_crosswalk.jsonl\n"
    if (root / "content_checksums.sha256").read_text(encoding="utf-8") != expected_line:
        raise ValueError("DS1 evaluator crosswalk checksum manifest differs")
    return record


__all__ = ["build_ds1_sam_evaluator_crosswalk", "verify_ds1_sam_evaluator_crosswalk"]
