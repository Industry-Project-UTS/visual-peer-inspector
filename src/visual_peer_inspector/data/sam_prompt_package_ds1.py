"""Build the frozen model-facing SAM prompt package for the DS1 screen."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.data.sam_prompt_package import (
    ALLOWED_ROUTE_NAMES,
    SAM_PROMPT_SCHEMA_VERSION,
    _assert_model_safe,
    _geometric_prompts,
    _normalise_box,
)
from visual_peer_inspector.models.locateanything import (
    MODEL_PRODUCER_KIND,
    ProposalResultStore,
)


PACKAGE_KIND = "ds1-sam-prompts"
EXPECTED_VIEW_IDS = tuple(
    f"V{index:02d}" for index in range(DS1_SCREEN_CONTRACT.views_per_scene)
)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_if_consistent(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"Existing DS1 prompt output differs: {path}")
        return
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _verify_safe_artifact_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    safe_root = path.parent
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise ValueError("LocateAnything safe artifact manifest is malformed")
    observed_paths: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        if not relative or relative in observed_paths or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("LocateAnything safe artifact manifest contains an unsafe path")
        observed_paths.add(relative)
        artifact = safe_root / relative
        if not artifact.is_file():
            raise ValueError("LocateAnything safe artifact manifest is incomplete")
        if file_sha256(artifact) != str(row.get("sha256")):
            raise ValueError("LocateAnything safe artifact hash mismatch")
        if artifact.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError("LocateAnything safe artifact size mismatch")
    if "run_record.json" not in observed_paths:
        raise ValueError("LocateAnything safe artifact manifest omits the run record")
    content_digest = sha256(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    ).hexdigest()
    if content_digest != str(manifest.get("manifest_content_sha256")):
        raise ValueError("LocateAnything safe artifact manifest content hash mismatch")


def _validate_upstream_run(
    run_record: Mapping[str, Any],
    latest_proposal_rows: list[dict[str, Any]],
    proposal_history_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    contract = DS1_SCREEN_CONTRACT
    if run_record.get("stage") != "ds1_locateanything_screen":
        raise ValueError("Expected the DS1 LocateAnything screen run")
    if run_record.get("status") != "completed" or run_record.get("completion_gate_passed") is not True:
        raise ValueError("LocateAnything DS1 run is not complete")
    if run_record.get("run_mode") != "model_inference" or run_record.get("producer_kind") != MODEL_PRODUCER_KIND:
        raise ValueError("DS1 prompts require real LocateAnything model inference")
    if run_record.get("inference_stage") != "full":
        raise ValueError("DS1 prompts require the full proposal stage")
    provenance = run_record.get("provenance", {})
    if (
        provenance.get("git_dirty") is not False
        or provenance.get("precommit_candidate") is not False
        or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("git_commit", "")))
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(provenance.get("snapshot_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(provenance.get("notebook_sha256", ""))
        )
    ):
        raise ValueError("DS1 prompts require clean committed LocateAnything provenance")
    cohort = run_record.get("cohort", {})
    expected = {
        "cohort_id": contract.cohort_id,
        "release_id": contract.release_id,
        "role": contract.role,
        "pilot_stage": contract.pilot_stage,
        "membership_sha256": contract.membership_sha256,
        "scene_count": contract.scene_count,
        "views_per_scene": contract.views_per_scene,
        "image_count": contract.image_count,
        "asset_group_count": contract.asset_group_count,
    }
    for key, value in expected.items():
        if cohort.get(key) != value:
            raise ValueError(f"LocateAnything DS1 run has another cohort {key}")
    summary = run_record.get("summary", {})
    if int(summary.get("job_count", -1)) != contract.image_count:
        raise ValueError("LocateAnything DS1 run has incomplete image coverage")
    if int(summary.get("successful_job_count", -1)) != contract.image_count or int(summary.get("failed_job_count", -1)) != 0:
        raise ValueError("LocateAnything DS1 run contains failed images")
    configuration_sha256 = str(run_record.get("configuration_sha256", ""))
    configuration = run_record.get("configuration", {})
    model_id = str(configuration.get("model_id", ""))
    model_revision = str(configuration.get("model_revision", ""))
    if not configuration_sha256 or not model_id or not model_revision:
        raise ValueError("LocateAnything DS1 run is missing frozen model configuration")
    if {str(row["configuration_sha256"]) for row in proposal_history_rows} != {configuration_sha256}:
        raise ValueError("LocateAnything results use another configuration")
    if {str(row["model_id"]) for row in proposal_history_rows} != {model_id}:
        raise ValueError("LocateAnything results use another model")
    if {str(row["model_revision"]) for row in proposal_history_rows} != {model_revision}:
        raise ValueError("LocateAnything results use another model revision")
    output_identity = run_record.get("output_identity", {})
    proposal_results_sha256 = str(
        output_identity.get("proposal_results_sha256", "")
    )
    if (
        output_identity.get("result_filename") != "proposal_results.jsonl"
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(output_identity.get("result_sha256", ""))
        )
        or not re.fullmatch(r"[0-9a-f]{64}", proposal_results_sha256)
        or proposal_results_sha256 != output_identity.get("result_sha256")
        or int(output_identity.get("result_count", -1))
        != len(latest_proposal_rows)
    ):
        raise ValueError("LocateAnything run record does not commit its current result set")
    return configuration_sha256, model_id, model_revision


def _validate_proposal_history_sequence(
    history_rows: list[dict[str, Any]],
) -> None:
    """Require every append-only retry history to use consecutive attempts."""

    previous_attempt_by_image: dict[tuple[str, str], int] = {}
    previous_row_by_image: dict[tuple[str, str], dict[str, Any]] = {}
    for row in history_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        previous_attempt = previous_attempt_by_image.get(key, 0)
        if int(row["attempt"]) != previous_attempt + 1:
            raise ValueError("LocateAnything proposal retry history is not sequential")
        if previous_attempt and previous_row_by_image[key]["status"] == "ok":
            raise ValueError(
                "LocateAnything proposal retry history continues after success"
            )
        previous_attempt_by_image[key] = int(row["attempt"])
        previous_row_by_image[key] = row


def build_ds1_sam_prompt_package(
    *,
    model_input_root: Path,
    geometry_manifest_path: Path,
    geometry_package_record_path: Path,
    proposal_results_path: Path,
    proposal_run_record_path: Path,
    proposal_artifact_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Join frozen DS1 media, trusted boxes and accepted automatic proposals."""

    contract = DS1_SCREEN_CONTRACT
    model_input_root = model_input_root.expanduser().resolve()
    model_audit = verify_model_input_package(model_input_root, contract=contract)
    model_manifest_path = model_input_root / "neutral_corpus_manifest.jsonl"
    model_manifest_sha256 = file_sha256(model_manifest_path)
    expected_images: dict[tuple[str, str], tuple[str, Path]] = {}
    for row in read_jsonl(model_manifest_path):
        scene_id = str(row["opaque_scene_id"])
        view_ids = tuple(str(value) for value in row["view_ids"])
        if view_ids != EXPECTED_VIEW_IDS:
            raise ValueError("DS1 model input views are not in the frozen order")
        for view_id in view_ids:
            relative = Path(scene_id) / "rgb" / f"{view_id}.png"
            expected_images[(scene_id, view_id)] = (
                relative.as_posix(),
                model_input_root / "media" / relative,
            )
    if len(expected_images) != contract.image_count:
        raise ValueError("DS1 model input does not contain 2,000 unique images")

    geometry_record = json.loads(geometry_package_record_path.read_text(encoding="utf-8"))
    for key, expected in {
        "cohort_id": contract.cohort_id,
        "release_id": contract.release_id,
        "membership_sha256": contract.membership_sha256,
    }.items():
        if geometry_record.get(key) != expected:
            raise ValueError(f"Trusted DS1 geometry has another {key}")
    if geometry_record.get("all_checks_passed") is not True:
        raise ValueError("Trusted DS1 geometry package is not verified")
    geometry_digest = file_sha256(geometry_manifest_path)
    recorded_geometry_digest = geometry_record.get("reference_manifest_sha256") or geometry_record.get("manifest_sha256")
    if recorded_geometry_digest != geometry_digest:
        raise ValueError("Trusted DS1 geometry manifest hash mismatch")
    reference_by_image: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for row in read_jsonl(geometry_manifest_path):
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key not in expected_images:
            raise ValueError("Trusted geometry contains an image outside DS1")
        if bool(row["visible"]):
            reference_by_image[key].append(_normalise_box(row["box_xyxy"]))
    if set(reference_by_image) != set(expected_images):
        raise ValueError("Every DS1 image must have at least one visible reference box")

    _verify_safe_artifact_manifest(proposal_artifact_manifest_path)
    run_record = json.loads(proposal_run_record_path.read_text(encoding="utf-8"))
    if proposal_run_record_path.resolve() != (proposal_artifact_manifest_path.parent / "run_record.json").resolve():
        raise ValueError("Proposal run record is outside the verified safe evidence root")
    proposal_results_sha256 = file_sha256(proposal_results_path)
    output_identity = run_record.get("output_identity", {})
    if (
        output_identity.get("result_sha256") != proposal_results_sha256
        or output_identity.get("proposal_results_sha256")
        != proposal_results_sha256
    ):
        raise ValueError("LocateAnything run record identifies another proposal file")
    configuration_sha256 = str(run_record.get("configuration_sha256", ""))
    proposal_store = ProposalResultStore(
        proposal_results_path,
        configuration_sha256,
        MODEL_PRODUCER_KIND,
    )
    proposal_history_rows = proposal_store.history_rows
    _validate_proposal_history_sequence(proposal_history_rows)
    proposal_rows = proposal_store.rows
    expected_image_sha256: dict[tuple[str, str], str] = {}
    for row in proposal_history_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key not in expected_images:
            raise ValueError("LocateAnything history contains an unexpected image")
        if row["cohort_id"] != contract.cohort_id:
            raise ValueError("LocateAnything history identifies another cohort")
        _, image_path = expected_images[key]
        if key not in expected_image_sha256:
            expected_image_sha256[key] = file_sha256(image_path)
        expected_digest = expected_image_sha256[key]
        if str(row["image_sha256"]) != expected_digest:
            raise ValueError("LocateAnything history identifies another image")
    proposal_by_image: dict[tuple[str, str], dict[str, Any]] = {}
    for row in proposal_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in proposal_by_image or key not in expected_images:
            raise ValueError("LocateAnything results contain duplicate or unexpected images")
        if row["status"] != "ok" or row["producer_kind"] != MODEL_PRODUCER_KIND:
            raise ValueError("LocateAnything results are not accepted model outputs")
        proposal_by_image[key] = row
    if set(proposal_by_image) != set(expected_images):
        raise ValueError("LocateAnything results do not cover all DS1 images")
    configuration_sha256, model_id, model_revision = _validate_upstream_run(
        run_record, proposal_rows, proposal_history_rows
    )
    if {str(row["input_manifest_sha256"]) for row in proposal_history_rows} != {model_manifest_sha256}:
        raise ValueError("LocateAnything results identify another model-input manifest")

    manifest_rows: list[dict[str, Any]] = []
    reference_prompt_count = 0
    automatic_prompt_count = 0
    for key in sorted(expected_images):
        scene_id, view_id = key
        image_relative, image_path = expected_images[key]
        image_sha256 = expected_image_sha256[key]
        proposal_row = proposal_by_image[key]
        if image_sha256 != str(proposal_row["image_sha256"]):
            raise ValueError("A DS1 image differs from the frozen proposal input")
        reference_prompts = _geometric_prompts(reference_by_image[key], "R")
        automatic_prompts = _geometric_prompts(proposal_row["boxes_xyxy"], "P")
        if not automatic_prompts:
            raise ValueError("Every DS1 image must have at least one automatic proposal")
        reference_prompt_count += len(reference_prompts)
        automatic_prompt_count += len(automatic_prompts)
        manifest_rows.append(
            {
                "schema_version": SAM_PROMPT_SCHEMA_VERSION,
                "cohort_id": contract.cohort_id,
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "image_path": image_relative,
                "image_sha256": image_sha256,
                "routes": {
                    "reference_box": reference_prompts,
                    "automatic_box": automatic_prompts,
                },
            }
        )
    _assert_model_safe(manifest_rows)

    output_root = output_root.expanduser().resolve()
    manifest_bytes = _jsonl_bytes(manifest_rows)
    manifest_path = output_root / "sam_prompt_manifest.jsonl"
    _write_if_consistent(manifest_path, manifest_bytes)
    upstream_record = {
        "schema_version": SAM_PROMPT_SCHEMA_VERSION,
        "decision": "use_frozen_ds1_proposals",
        "cohort_id": contract.cohort_id,
        "release_id": contract.release_id,
        "pilot_stage": contract.pilot_stage,
        "membership_sha256": contract.membership_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "model_package_record_sha256": file_sha256(model_input_root / "package_record.json"),
        "geometry_manifest_sha256": geometry_digest,
        "geometry_package_record_sha256": file_sha256(geometry_package_record_path),
        "proposal_results_sha256": file_sha256(proposal_results_path),
        "proposal_run_record_sha256": file_sha256(proposal_run_record_path),
        "proposal_artifact_manifest_sha256": file_sha256(proposal_artifact_manifest_path),
        "proposal_configuration_sha256": configuration_sha256,
        "proposal_model_id": model_id,
        "proposal_model_revision": model_revision,
    }
    _assert_model_safe(upstream_record)
    upstream_bytes = json.dumps(upstream_record, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    upstream_path = output_root / "upstream_decision.json"
    _write_if_consistent(upstream_path, upstream_bytes)
    checksum_bytes = "".join(
        f"{file_sha256(path)}  {path.name}\n" for path in (manifest_path, upstream_path)
    ).encode("utf-8")
    checksum_path = output_root / "content_checksums.sha256"
    _write_if_consistent(checksum_path, checksum_bytes)
    record = {
        "schema_version": SAM_PROMPT_SCHEMA_VERSION,
        "package_kind": PACKAGE_KIND,
        "cohort_id": contract.cohort_id,
        "release_id": contract.release_id,
        "role": contract.role,
        "pilot_stage": contract.pilot_stage,
        "membership_sha256": contract.membership_sha256,
        "scene_count": contract.scene_count,
        "views_per_scene": contract.views_per_scene,
        "image_count": contract.image_count,
        "asset_group_count": contract.asset_group_count,
        "reference_prompt_count": reference_prompt_count,
        "automatic_prompt_count": automatic_prompt_count,
        "model_manifest_sha256": model_manifest_sha256,
        "prompt_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "upstream_decision_sha256": sha256(upstream_bytes).hexdigest(),
        "content_checksums_sha256": sha256(checksum_bytes).hexdigest(),
        "all_checks_passed": model_audit.all_checks_passed,
    }
    _write_if_consistent(
        output_root / "package_record.json",
        json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    verify_ds1_sam_prompt_package(output_root)
    return record


def verify_ds1_sam_prompt_package(root: Path) -> dict[str, Any]:
    """Verify exact DS1 coverage, routes and content hashes."""

    root = root.expanduser().resolve()
    record = json.loads((root / "package_record.json").read_text(encoding="utf-8"))
    contract = DS1_SCREEN_CONTRACT
    expected_record = {
        "package_kind": PACKAGE_KIND,
        "cohort_id": contract.cohort_id,
        "release_id": contract.release_id,
        "role": contract.role,
        "pilot_stage": contract.pilot_stage,
        "membership_sha256": contract.membership_sha256,
        "scene_count": contract.scene_count,
        "views_per_scene": contract.views_per_scene,
        "image_count": contract.image_count,
        "asset_group_count": contract.asset_group_count,
        "all_checks_passed": True,
    }
    for key, value in expected_record.items():
        if record.get(key) != value:
            raise ValueError(f"DS1 SAM prompt package record mismatch: {key}")
    expected_files = {
        "content_checksums.sha256",
        "package_record.json",
        "sam_prompt_manifest.jsonl",
        "upstream_decision.json",
    }
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_symlink() for path in root.iterdir()):
        raise ValueError("DS1 SAM prompt package file allowlist mismatch")
    manifest_path = root / "sam_prompt_manifest.jsonl"
    upstream_path = root / "upstream_decision.json"
    checksum_path = root / "content_checksums.sha256"
    if file_sha256(manifest_path) != str(record["prompt_manifest_sha256"]):
        raise ValueError("DS1 SAM prompt manifest hash mismatch")
    if file_sha256(upstream_path) != str(record["upstream_decision_sha256"]):
        raise ValueError("DS1 SAM upstream decision hash mismatch")
    if file_sha256(checksum_path) != str(record["content_checksums_sha256"]):
        raise ValueError("DS1 SAM checksum manifest hash mismatch")
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    expected_upstream = {
        "schema_version",
        "decision",
        "cohort_id",
        "release_id",
        "pilot_stage",
        "membership_sha256",
        "model_manifest_sha256",
        "model_package_record_sha256",
        "geometry_manifest_sha256",
        "geometry_package_record_sha256",
        "proposal_results_sha256",
        "proposal_run_record_sha256",
        "proposal_artifact_manifest_sha256",
        "proposal_configuration_sha256",
        "proposal_model_id",
        "proposal_model_revision",
    }
    if set(upstream) != expected_upstream:
        raise ValueError("DS1 SAM upstream decision schema mismatch")
    for key, expected in {
        "schema_version": SAM_PROMPT_SCHEMA_VERSION,
        "decision": "use_frozen_ds1_proposals",
        "cohort_id": contract.cohort_id,
        "release_id": contract.release_id,
        "pilot_stage": contract.pilot_stage,
        "membership_sha256": contract.membership_sha256,
        "model_manifest_sha256": record["model_manifest_sha256"],
    }.items():
        if upstream.get(key) != expected:
            raise ValueError(f"DS1 SAM upstream decision mismatch: {key}")
    if not upstream["proposal_model_id"] or not upstream["proposal_model_revision"]:
        raise ValueError("DS1 SAM upstream model provenance is incomplete")
    for key, value in upstream.items():
        if key.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError(f"DS1 SAM upstream digest is invalid: {key}")
    _assert_model_safe(upstream)
    checksum_rows = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in checksum_rows:
            raise ValueError("DS1 SAM checksum manifest contains duplicates")
        checksum_rows[relative] = digest
    if set(checksum_rows) != {"sam_prompt_manifest.jsonl", "upstream_decision.json"}:
        raise ValueError("DS1 SAM checksum manifest path mismatch")
    for relative, digest in checksum_rows.items():
        if file_sha256(root / relative) != digest:
            raise ValueError("DS1 SAM content checksum mismatch")

    rows = read_jsonl(manifest_path)
    _assert_model_safe(rows)
    keys: set[tuple[str, str]] = set()
    scenes: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("cohort_id") != contract.cohort_id:
            raise ValueError("DS1 SAM prompt row identifies another cohort")
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in keys:
            raise ValueError("DS1 SAM prompt manifest contains duplicate images")
        keys.add(key)
        scenes[key[0]] += 1
        if set(row.get("routes", {})) != ALLOWED_ROUTE_NAMES:
            raise ValueError("DS1 SAM prompt row must contain exactly two routes")
        for route, prefix in (("reference_box", "R"), ("automatic_box", "P")):
            prompts = row["routes"][route]
            boxes = [_normalise_box(prompt["box_xyxy"]) for prompt in prompts]
            identifiers = [str(prompt["prompt_key"]) for prompt in prompts]
            if identifiers != [f"{prefix}{index:03d}" for index in range(len(prompts))] or boxes != sorted(boxes):
                raise ValueError("DS1 SAM prompts are not in canonical geometric order")
            if not prompts:
                raise ValueError("Both DS1 SAM prompt routes must be non-empty")
            route_counts[route] += len(prompts)
    if len(keys) != contract.image_count or len(scenes) != contract.scene_count:
        raise ValueError("DS1 SAM prompt package coverage is incomplete")
    if set(scenes.values()) != {contract.views_per_scene}:
        raise ValueError("Every DS1 SAM prompt scene must contain all views")
    for route, field in (("reference_box", "reference_prompt_count"), ("automatic_box", "automatic_prompt_count")):
        if route_counts[route] != int(record[field]):
            raise ValueError(f"DS1 SAM {route} count mismatch")
    return record


__all__ = [
    "PACKAGE_KIND",
    "build_ds1_sam_prompt_package",
    "verify_ds1_sam_prompt_package",
]
