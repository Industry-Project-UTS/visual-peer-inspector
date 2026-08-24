"""Evaluator-only DS0 comparison of frozen SAM 2.1 mask routes.

This workflow starts after Notebook 05 has finished. It verifies the private
evaluator package and the complete SAM output before reading either source.
Per-object records remain under ``private``. Only aggregate tables, charts and
provenance hashes are written under ``safe``.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_peer_inspector.data.model_safe_package import (
    SMOKE20_COHORT_ID,
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.data.mask_route_package import build_mask_route_package
from visual_peer_inspector.evaluation.masks import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    DEFAULT_MASK_IOU_THRESHOLD,
    evaluate_mask_route_image,
    paired_asset_clustered_bootstrap,
    summarize_mask_routes,
)
from visual_peer_inspector.evaluation.proposals import (
    evaluate_proposal_dataset,
    summarize_proposal_evaluation,
)
from visual_peer_inspector.evaluation.sam_mask_package import (
    verify_ds0_mask_evaluator_package,
)
from visual_peer_inspector.evidence import (
    save_bar_chart,
    save_histogram,
    save_scatter_plot,
    write_csv_rows,
)
from visual_peer_inspector.models.locateanything import LocateAnythingConfiguration
from visual_peer_inspector.models.sam2 import (
    CONTRACT_PRODUCER_KIND,
    MODEL_PRODUCER_KIND,
    SAM2_CHECKPOINT_SHA256,
    binary_mask_sha256,
    decode_binary_mask,
    encode_binary_mask,
)


EVALUATION_SCHEMA_VERSION = 1
EVALUATION_STAGE = "ds0_sam21_mask_evaluation"
MODEL_EVIDENCE_MODE = "model_evidence"
CONTRACT_TEST_MODE = "contract_test"
EXPECTED_SCENES = 20
EXPECTED_VIEWS_PER_SCENE = 20
EXPECTED_IMAGES = 400
EXPECTED_REFERENCE_PROMPTS = 1_640
EXPECTED_AUTOMATIC_PROMPTS = 1_616
EXPECTED_TOTAL_PROMPTS = 3_256
ROUTES = ("reference_box", "automatic_box")
PROPOSAL_EXACT_COUNT_MINIMUM = 0.90
PROPOSAL_RECALL_AT_0_50_MINIMUM = 0.95
REPEAT_AGREEMENT_MINIMUM = 0.98
REFERENCE_MASK_RECALL_MINIMUM = 0.90
REFERENCE_MEAN_MATCHED_IOU_MINIMUM = 0.90
AUTOMATIC_MASK_RECALL_MINIMUM = 0.90
AUTOMATIC_MEAN_MATCHED_IOU_MINIMUM = 0.90
EVALUATOR_PROVENANCE_KEYS = {
    "git_commit",
    "git_dirty",
    "snapshot_sha256",
    "precommit_candidate",
    "notebook_sha256",
    "kaggle_notebook_slug",
    "kaggle_notebook_version",
    "code_dataset_version",
}

ROUTE_SUMMARY_COLUMNS = [
    "route",
    "reference_count",
    "prediction_count",
    "matched_reference_count",
    "missed_reference_count",
    "unmatched_prediction_count",
    "detection_aware_recall_at_0_75",
    "mean_matched_iou",
    "detection_aware_mean_iou",
    "mean_matched_dice",
    "detection_aware_mean_dice",
    "mean_matched_boundary_precision",
    "mean_matched_boundary_recall",
    "mean_matched_boundary_f1",
    "detection_aware_mean_boundary_precision",
    "detection_aware_mean_boundary_recall",
    "detection_aware_mean_boundary_f1",
    "mean_prediction_spill_fraction",
    "prediction_spill_rate",
    "mask_iou_threshold",
    "boundary_tolerance_pixels",
]
VIEW_ROUTE_SUMMARY_COLUMNS = [
    "view_id",
    "route",
    "reference_count",
    "prediction_count",
    "matched_reference_count",
    "missed_reference_count",
    "unmatched_prediction_count",
    "detection_aware_recall_at_0_75",
    "detection_aware_mean_iou",
    "detection_aware_mean_dice",
    "detection_aware_mean_boundary_f1",
]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _safe_relative_path(value: Any) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or "\\" in text
        or ".." in pure.parts
        or pure.is_absolute()
    ):
        raise ValueError(f"Unsafe artifact path: {text}")
    return Path(*pure.parts)


def _validated_evaluator_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(provenance)
    if set(value) != EVALUATOR_PROVENANCE_KEYS:
        raise ValueError("Evaluator provenance does not follow the accepted schema")
    if not isinstance(value["git_dirty"], bool) or not isinstance(
        value["precommit_candidate"], bool
    ):
        raise ValueError("Evaluator provenance Git flags must be Boolean")
    hexadecimal_fields = {
        "git_commit": 40,
        "snapshot_sha256": 64,
        "notebook_sha256": 64,
    }
    for key, length in hexadecimal_fields.items():
        text = value[key]
        if (
            not isinstance(text, str)
            or len(text) != length
            or any(character not in "0123456789abcdef" for character in text)
        ):
            raise ValueError(f"Evaluator provenance {key} is invalid")
    expected_slug = (
        "esreekarreddy/toysad-8k-ds0-proposal-and-mask-evaluation"
    )
    if value["kaggle_notebook_slug"] != expected_slug:
        raise ValueError("Evaluator provenance notebook slug is invalid")
    for key in (
        "kaggle_notebook_version",
        "code_dataset_version",
    ):
        text = value[key]
        if not isinstance(text, str) or not text or "/" in text or "\\" in text:
            raise ValueError(f"Evaluator provenance {key} is invalid")
    return value


def _manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact roots cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def _write_artifact_manifest(root: Path) -> None:
    rows = _manifest_rows(root)
    _write_json(
        root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
            "manifest_content_sha256": sha256(
                b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
            ).hexdigest(),
        },
    )


def verify_artifact_manifest(root: Path) -> dict[str, Any]:
    """Verify one complete immutable artifact root and its file allowlist."""

    expanded_root = root.expanduser()
    if expanded_root.is_symlink():
        raise ValueError("Artifact root is missing or unsafe")
    root = expanded_root.resolve()
    if not root.is_dir():
        raise ValueError("Artifact root is missing or unsafe")
    manifest_path = root / "artifact_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Artifact manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if (
        not isinstance(rows, list)
        or int(manifest.get("artifact_count", -1)) != len(rows)
    ):
        raise ValueError("Artifact manifest count is inconsistent")
    listed_paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError(
                "Artifact manifest row does not follow the accepted schema"
            )
        relative = _safe_relative_path(row["path"])
        relative_text = relative.as_posix()
        if relative_text in listed_paths:
            raise ValueError("Artifact manifest contains a duplicate path")
        listed_paths.append(relative_text)
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or root not in path.resolve().parents
        ):
            raise ValueError("Artifact manifest points to a missing or unsafe file")
        if file_sha256(path) != str(row["sha256"]):
            raise ValueError(f"Artifact SHA-256 differs for {relative_text}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Artifact size differs for {relative_text}")
    if listed_paths != sorted(listed_paths):
        raise ValueError("Artifact manifest paths are not in deterministic order")
    actual_paths = [row["path"] for row in _manifest_rows(root)]
    if actual_paths != listed_paths:
        raise ValueError("Artifact manifest does not match the complete file set")
    content_digest = sha256(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    ).hexdigest()
    if content_digest != manifest.get("manifest_content_sha256"):
        raise ValueError("Artifact manifest content hash differs")
    return manifest


def _read_reference_mask(
    evaluator_root: Path,
    row: Mapping[str, Any],
    *,
    threshold: int,
) -> np.ndarray:
    relative = _safe_relative_path(row["mask_path"])
    path = evaluator_root / "oracle_geometry" / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("A supplied reference mask is missing or unsafe")
    if file_sha256(path) != str(row["mask_sha256"]):
        raise ValueError("A supplied reference mask differs from its SHA-256")
    with Image.open(path) as opened:
        opened.load()
        if opened.size != (256, 256):
            raise ValueError("A supplied reference mask has unexpected dimensions")
        grey = np.asarray(opened.convert("L"), dtype=np.uint8)
    mask = grey >= threshold
    if int(mask.sum()) != int(row["foreground_pixels"]):
        raise ValueError("A supplied reference mask has unexpected foreground area")
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if box != [int(value) for value in row["box_xyxy"]]:
        raise ValueError("A supplied reference mask has unexpected geometry")
    return mask


def _verify_locateanything_manifest(root: Path) -> dict[str, str]:
    """Verify the Notebook 04 flat SHA-256 manifest and complete file set."""

    manifest_path = root / "report/artifact_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Notebook 04 artifact manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Notebook 04 artifact manifest is invalid")
    listed: dict[str, str] = {}
    for relative_text, digest in manifest.items():
        relative = _safe_relative_path(relative_text)
        safe_text = relative.as_posix()
        if safe_text in listed:
            raise ValueError("Notebook 04 artifact manifest has duplicate paths")
        path = root / relative
        resolved = path.resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or root not in resolved.parents
            or len(str(digest)) != 64
            or file_sha256(path) != str(digest)
        ):
            raise ValueError(f"Notebook 04 artifact differs for {safe_text}")
        listed[safe_text] = str(digest)
    actual = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Notebook 04 output cannot contain symbolic links")
        if path.is_file() and path != manifest_path:
            actual.append(path.relative_to(root).as_posix())
    if actual != sorted(listed):
        raise ValueError("Notebook 04 artifact manifest differs from its file set")
    return listed


def _load_verified_locateanything(
    proposal_output_root: Path,
    *,
    expected_image_keys: set[tuple[str, str]],
    geometry_rows: Sequence[Mapping[str, Any]],
    expected_proposal_results_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Verify the full model proposal run and recompute its protected gate."""

    expanded_root = proposal_output_root.expanduser()
    if expanded_root.is_symlink():
        raise ValueError("Notebook 04 output root is missing or unsafe")
    proposal_output_root = expanded_root.resolve()
    if not proposal_output_root.is_dir():
        raise ValueError("Notebook 04 output root is missing or unsafe")
    manifest = _verify_locateanything_manifest(proposal_output_root)
    proposal_path = proposal_output_root / "proposal_results.jsonl"
    repeat_path = proposal_output_root / "repeat_pilot_results.jsonl"
    run_path = proposal_output_root / "report/run_record.json"
    gates_path = proposal_output_root / "report/completion_gates.csv"
    if file_sha256(proposal_path) != expected_proposal_results_sha256:
        raise ValueError(
            "Notebook 04 proposal results differ from the frozen crosswalk"
        )
    if manifest.get("proposal_results.jsonl") != expected_proposal_results_sha256:
        raise ValueError(
            "Notebook 04 proposal manifest differs from the frozen crosswalk"
        )
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    required = {
        "stage": "ds0_locateanything_proposals",
        "status": "completed",
        "run_mode": "model_inference",
        "inference_stage": "full",
        "producer_kind": "locateanything_model",
    }
    for key, expected in required.items():
        if run_record.get(key) != expected:
            raise ValueError(f"Notebook 04 model run differs for {key}")
    configuration = run_record.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Notebook 04 configuration is missing")
    configuration_fields = set(LocateAnythingConfiguration.__dataclass_fields__)
    expected_configuration_fields = configuration_fields | {
        "producer_kind",
        "run_mode",
        "inference_stage",
        "maximum_attempts_per_job",
        "maximum_consecutive_failed_jobs",
        "scene_count",
        "views_per_scene",
    }
    if set(configuration) != expected_configuration_fields:
        raise ValueError("Notebook 04 configuration schema differs")
    configuration_sha256 = LocateAnythingConfiguration(
        **{name: configuration[name] for name in configuration_fields}
    ).sha256
    if configuration_sha256 != run_record.get("configuration_sha256"):
        raise ValueError("Notebook 04 configuration hash differs")
    if (
        configuration.get("producer_kind") != "locateanything_model"
        or configuration.get("run_mode") != "model_inference"
        or configuration.get("inference_stage") != "full"
    ):
        raise ValueError("Notebook 04 configuration is not a full model run")
    summary = run_record.get("summary", {})
    required_summary = {
        "job_count": EXPECTED_IMAGES,
        "successful_job_count": EXPECTED_IMAGES,
        "failed_job_count": 0,
        "possibly_truncated_count": 0,
        "scene_count": EXPECTED_SCENES,
        "zero_proposal_image_count": 0,
    }
    for key, expected in required_summary.items():
        if int(summary.get(key, -1)) != expected:
            raise ValueError(f"Notebook 04 full-run coverage differs for {key}")
    repeat = run_record.get("repeat_agreement", {})
    repeat_job_count = int(repeat.get("job_count", -1))
    repeat_matching_count = int(repeat.get("matching_job_count", -1))
    repeat_agreement = float(repeat.get("box_agreement_rate", -1.0))
    if (
        repeat_job_count != EXPECTED_SCENES
        or repeat_matching_count < 0
        or not math.isclose(
            repeat_agreement,
            repeat_matching_count / repeat_job_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or repeat_agreement < REPEAT_AGREEMENT_MINIMUM
        or str(repeat.get("result_file_sha256")) != file_sha256(repeat_path)
    ):
        raise ValueError("Notebook 04 repeat agreement is incomplete")
    with gates_path.open("r", encoding="utf-8", newline="") as handle:
        gate_rows = list(csv.DictReader(handle))
    if not gate_rows or any(row.get("status") != "pass" for row in gate_rows):
        raise ValueError("Notebook 04 operational completion gates did not all pass")

    proposal_rows = read_jsonl(proposal_path)
    if len(proposal_rows) != EXPECTED_IMAGES:
        raise ValueError("Notebook 04 proposal result coverage is incomplete")
    seen: set[tuple[str, str]] = set()
    proposal_by_image: dict[tuple[str, str], Mapping[str, Any]] = {}
    proposal_count = 0
    for row in proposal_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in seen or key not in expected_image_keys:
            raise ValueError("Notebook 04 proposal image identity is invalid")
        seen.add(key)
        if (
            row.get("status") != "ok"
            or row.get("producer_kind") != "locateanything_model"
            or row.get("configuration_sha256") != configuration_sha256
            or row.get("cohort_id") != SMOKE20_COHORT_ID
            or row.get("possibly_truncated") is not False
        ):
            raise ValueError("Notebook 04 proposal row is not an accepted model result")
        if (
            row.get("input_manifest_sha256")
            != run_record.get("input_manifest_sha256")
            or row.get("model_id") != configuration["model_id"]
            or row.get("model_revision") != configuration["model_revision"]
            or row.get("prompt") != configuration["prompt"]
        ):
            raise ValueError("Notebook 04 proposal provenance is inconsistent")
        boxes = row.get("boxes_xyxy")
        if (
            not isinstance(boxes, list)
            or int(row.get("proposal_count", -1)) != len(boxes)
        ):
            raise ValueError("Notebook 04 proposal count is inconsistent")
        proposal_count += len(boxes)
        proposal_by_image[key] = row
    if seen != expected_image_keys or proposal_count != EXPECTED_AUTOMATIC_PROMPTS:
        raise ValueError("Notebook 04 proposal membership differs from frozen DS0")

    repeat_rows = read_jsonl(repeat_path)
    if len(repeat_rows) != repeat_job_count:
        raise ValueError("Notebook 04 repeat result coverage is incomplete")
    repeat_keys: set[tuple[str, str]] = set()
    recomputed_matching_count = 0
    for row in repeat_rows:
        key = (str(row.get("opaque_scene_id")), str(row.get("view_id")))
        if key in repeat_keys or key not in proposal_by_image:
            raise ValueError("Notebook 04 repeat result identity is invalid")
        repeat_keys.add(key)
        full_row = proposal_by_image[key]
        repeat_boxes = row.get("boxes_xyxy")
        if (
            row.get("status") != "ok"
            or row.get("producer_kind") != "locateanything_model"
            or row.get("configuration_sha256") != configuration_sha256
            or row.get("cohort_id") != SMOKE20_COHORT_ID
            or row.get("image_sha256") != full_row.get("image_sha256")
            or not isinstance(repeat_boxes, list)
            or row.get("proposal_count") != len(repeat_boxes)
        ):
            raise ValueError("Notebook 04 repeat result is inconsistent")
        if repeat_boxes == full_row["boxes_xyxy"]:
            recomputed_matching_count += 1
    expected_repeat_keys = {
        (scene_id, "V00") for scene_id, _view_id in expected_image_keys
    }
    if repeat_keys != expected_repeat_keys:
        raise ValueError("Notebook 04 repeat pilot does not cover one view per scene")
    recomputed_agreement = recomputed_matching_count / len(repeat_rows)
    if (
        recomputed_matching_count != repeat_matching_count
        or not math.isclose(
            recomputed_agreement,
            repeat_agreement,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or recomputed_agreement < REPEAT_AGREEMENT_MINIMUM
    ):
        raise ValueError("Notebook 04 repeat agreement differs from frozen results")

    proposal_evaluation = evaluate_proposal_dataset(
        reference_rows=geometry_rows,
        proposal_rows=proposal_rows,
    )
    gate_summary = summarize_proposal_evaluation(
        proposal_evaluation["image_records"]
    )
    if (
        int(gate_summary["image_count"]) != EXPECTED_IMAGES
        or int(gate_summary["reference_count"]) != EXPECTED_REFERENCE_PROMPTS
        or int(gate_summary["proposal_count"]) != EXPECTED_AUTOMATIC_PROMPTS
    ):
        raise ValueError("Recomputed Notebook 04 gate coverage is incomplete")
    gate_passed = (
        float(gate_summary["exact_count_rate"])
        >= PROPOSAL_EXACT_COUNT_MINIMUM
        and float(gate_summary["micro_recall_at_0_50"])
        >= PROPOSAL_RECALL_AT_0_50_MINIMUM
    )
    if not gate_passed:
        raise ValueError("Recomputed Notebook 04 proposal gate did not pass")
    provenance = run_record.get("provenance", {})
    if not isinstance(provenance, dict) or not isinstance(
        provenance.get("git_dirty"), bool
    ):
        raise ValueError("Notebook 04 git-dirty provenance is missing")
    hashes = {
        "proposal_artifact_manifest_sha256": file_sha256(
            proposal_output_root / "report/artifact_manifest.json"
        ),
        "proposal_run_record_sha256": file_sha256(run_path),
        "proposal_results_sha256": file_sha256(proposal_path),
        "proposal_repeat_results_sha256": file_sha256(repeat_path),
    }
    return run_record, proposal_rows, gate_summary, hashes


def _validate_sam_run_record(
    record: Mapping[str, Any],
    *,
    evaluation_mode: str,
    expected_prompt_package_sha256: str,
) -> None:
    if evaluation_mode not in {MODEL_EVIDENCE_MODE, CONTRACT_TEST_MODE}:
        raise ValueError("evaluation_mode must be model_evidence or contract_test")
    required = {
        "status": "completed",
        "inference_stage": "full",
        "completion_gate_passed": True,
        "reference_prompt_count": EXPECTED_REFERENCE_PROMPTS,
        "automatic_prompt_count": EXPECTED_AUTOMATIC_PROMPTS,
        "prompt_package_sha256": expected_prompt_package_sha256,
    }
    if evaluation_mode == MODEL_EVIDENCE_MODE:
        required.update(
            {
                "run_mode": "model_inference",
                "producer_kind": MODEL_PRODUCER_KIND,
                "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
            }
        )
    else:
        required.update(
            {
                "run_mode": "contract_smoke",
                "producer_kind": CONTRACT_PRODUCER_KIND,
                "checkpoint_sha256": None,
            }
        )
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(
                f"Notebook 05 output is not accepted for {evaluation_mode}: {key}"
            )
    summary = record.get("summary", {})
    expected_summary = {
        "scene_count": EXPECTED_SCENES,
        "image_count": EXPECTED_IMAGES,
        "mask_count": EXPECTED_TOTAL_PROMPTS,
    }
    for key, expected in expected_summary.items():
        if int(summary.get(key, -1)) != expected:
            raise ValueError(f"Notebook 05 coverage differs for {key}")
    repeat = record.get("repeat_agreement", {})
    if (
        float(repeat.get("mask_digest_agreement_rate", -1.0)) != 1.0
        or float(repeat.get("score_agreement_rate", -1.0)) != 1.0
    ):
        raise ValueError("Notebook 05 repeat agreement is incomplete")
    configuration = record.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Notebook 05 configuration is missing")
    if sha256(canonical_json_bytes(configuration)).hexdigest() != record.get(
        "configuration_sha256"
    ):
        raise ValueError("Notebook 05 configuration hash differs")
    if evaluation_mode == MODEL_EVIDENCE_MODE:
        if configuration.get("checkpoint_sha256") != SAM2_CHECKPOINT_SHA256:
            raise ValueError("Notebook 05 configuration identifies another checkpoint")


def _validated_prediction(
    item: Mapping[str, Any],
    *,
    expected_box: Sequence[Any],
) -> dict[str, Any]:
    if [int(value) for value in item.get("box_xyxy", [])] != [
        int(value) for value in expected_box
    ]:
        raise ValueError("A SAM result box differs from the frozen prompt")
    mask_rle = item.get("mask_rle")
    if not isinstance(mask_rle, Mapping) or mask_rle.get("size") != [256, 256]:
        raise ValueError("A decoded SAM mask has unexpected dimensions")
    mask = decode_binary_mask(mask_rle)
    if mask.shape != (256, 256):
        raise ValueError("A decoded SAM mask has unexpected dimensions")
    foreground = int(mask.sum())
    if foreground != int(item.get("foreground_pixels", -1)):
        raise ValueError("A decoded SAM mask has unexpected foreground area")
    if binary_mask_sha256(mask) != str(item.get("mask_sha256")):
        raise ValueError("A decoded SAM mask differs from its recorded SHA-256")
    expected_status = "empty" if foreground == 0 else "non_empty"
    if item.get("mask_status") != expected_status:
        raise ValueError("A decoded SAM mask has an inconsistent status")
    score = float(item.get("predicted_quality_score"))
    if not math.isfinite(score):
        raise ValueError("A SAM predicted quality score is not finite")
    return {
        "mask": mask,
        "mask_sha256": str(item["mask_sha256"]),
        "foreground_pixels": foreground,
        "predicted_quality_score": score,
    }


def _load_verified_sam_rows(
    sam_output_root: Path,
    *,
    evaluation_mode: str,
    expected_prompt_package_sha256: str,
    crosswalk_by_key: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    expected_image_keys: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    safe_root = sam_output_root / "safe"
    private_root = sam_output_root / "private"
    safe_manifest = verify_artifact_manifest(safe_root)
    private_manifest = verify_artifact_manifest(private_root)
    run_record = json.loads((safe_root / "run_record.json").read_text(encoding="utf-8"))
    _validate_sam_run_record(
        run_record,
        evaluation_mode=evaluation_mode,
        expected_prompt_package_sha256=expected_prompt_package_sha256,
    )
    index_rows = read_jsonl(private_root / "mask_results.jsonl")
    result_paths = sorted((private_root / "image_results").glob("*/*.json"))
    if len(index_rows) != EXPECTED_IMAGES or len(result_paths) != EXPECTED_IMAGES:
        raise ValueError("Notebook 05 image-result coverage is incomplete")
    index_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in index_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in index_by_key:
            raise ValueError("Notebook 05 mask index contains duplicate images")
        index_by_key[key] = dict(row)
    if set(index_by_key) != expected_image_keys:
        raise ValueError("Notebook 05 image membership differs from the evaluator")

    result_by_image: dict[tuple[str, str], dict[str, Any]] = {}
    seen_prompts: set[tuple[str, str, str, str]] = set()
    route_counts: dict[str, int] = defaultdict(int)
    for path in result_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (str(row.get("opaque_scene_id")), str(row.get("view_id")))
        expected_path = private_root / "image_results" / key[0] / f"{key[1]}.json"
        if (
            path != expected_path
            or key in result_by_image
            or key not in expected_image_keys
        ):
            raise ValueError("Notebook 05 image-result identity is invalid")
        if row != index_by_key[key]:
            raise ValueError("Notebook 05 image result differs from its frozen index")
        required = {
            "schema_version": 1,
            "status": "ok",
            "cohort_id": SMOKE20_COHORT_ID,
            "producer_kind": run_record["producer_kind"],
            "configuration_sha256": run_record["configuration_sha256"],
            "prompt_package_sha256": expected_prompt_package_sha256,
        }
        for field, expected in required.items():
            if row.get(field) != expected:
                raise ValueError(f"Notebook 05 image result differs for {field}")
        if set(row.get("routes", {})) != set(ROUTES):
            raise ValueError("Notebook 05 image result has unexpected routes")
        prompt_count = 0
        for route in ROUTES:
            for item in row["routes"][route]:
                prompt_key = str(item.get("prompt_key"))
                identity = (*key, route, prompt_key)
                if identity in seen_prompts or identity not in crosswalk_by_key:
                    raise ValueError(
                        "Notebook 05 contains a duplicate or unknown prompt"
                    )
                seen_prompts.add(identity)
                route_counts[route] += 1
                prompt_count += 1
        if prompt_count != int(row.get("prompt_count", -1)):
            raise ValueError("Notebook 05 image prompt count is inconsistent")
        result_by_image[key] = row
    if seen_prompts != set(crosswalk_by_key):
        raise ValueError(
            "Notebook 05 prompt coverage differs from the evaluator crosswalk"
        )
    if route_counts != {
        "reference_box": EXPECTED_REFERENCE_PROMPTS,
        "automatic_box": EXPECTED_AUTOMATIC_PROMPTS,
    }:
        raise ValueError("Notebook 05 route coverage is incomplete")

    repeat_path = private_root / "repeat_mask_results.jsonl"
    repeat_rows = read_jsonl(repeat_path)
    repeat_record = run_record["repeat_agreement"]
    expected_repeat_scenes = {
        scene_id for scene_id, _view_id in expected_image_keys
    }
    expected_repeat_views = {
        f"V{index:02d}" for index in range(EXPECTED_VIEWS_PER_SCENE)
    }
    repeat_keys: set[tuple[str, str]] = set()
    repeat_prompt_count = 0
    matching_digest_count = 0
    matching_score_count = 0
    score_tolerance = float(repeat_record.get("score_tolerance", -1.0))
    if score_tolerance < 0.0 or not math.isfinite(score_tolerance):
        raise ValueError("Notebook 05 repeat score tolerance is invalid")
    for repeat_row in repeat_rows:
        key = (
            str(repeat_row.get("opaque_scene_id")),
            str(repeat_row.get("view_id")),
        )
        if key in repeat_keys or key not in result_by_image:
            raise ValueError("Notebook 05 repeat result identity is invalid")
        repeat_keys.add(key)
        main_row = result_by_image[key]
        if (
            repeat_row.get("status") != "ok"
            or repeat_row.get("producer_kind") != run_record["producer_kind"]
            or repeat_row.get("configuration_sha256")
            != run_record["configuration_sha256"]
            or repeat_row.get("prompt_package_sha256")
            != expected_prompt_package_sha256
            or repeat_row.get("image_sha256") != main_row.get("image_sha256")
            or set(repeat_row.get("routes", {})) != set(ROUTES)
        ):
            raise ValueError("Notebook 05 repeat result is inconsistent")
        image_prompt_count = 0
        for route in ROUTES:
            main_items = {
                str(item["prompt_key"]): item
                for item in main_row["routes"][route]
            }
            repeat_items = {
                str(item["prompt_key"]): item
                for item in repeat_row["routes"][route]
            }
            if len(repeat_items) != len(repeat_row["routes"][route]) or set(
                repeat_items
            ) != set(main_items):
                raise ValueError("Notebook 05 repeat prompt membership differs")
            for prompt_key, item in repeat_items.items():
                crosswalk_row = crosswalk_by_key[
                    (*key, route, prompt_key)
                ]
                validated = _validated_prediction(
                    item,
                    expected_box=crosswalk_row["prompt_box_xyxy"],
                )
                main_item = main_items[prompt_key]
                repeat_prompt_count += 1
                image_prompt_count += 1
                if validated["mask_sha256"] == main_item["mask_sha256"]:
                    matching_digest_count += 1
                if math.isclose(
                    validated["predicted_quality_score"],
                    float(main_item["predicted_quality_score"]),
                    rel_tol=0.0,
                    abs_tol=score_tolerance,
                ):
                    matching_score_count += 1
        if image_prompt_count != int(repeat_row.get("prompt_count", -1)):
            raise ValueError("Notebook 05 repeat prompt count is inconsistent")
    if (
        len(repeat_keys) != EXPECTED_SCENES
        or {scene_id for scene_id, _view_id in repeat_keys}
        != expected_repeat_scenes
        or {_view_id for _scene_id, _view_id in repeat_keys}
        != expected_repeat_views
    ):
        raise ValueError(
            "Notebook 05 repeat pilot must cover every scene and view position"
        )
    if repeat_prompt_count <= 0:
        raise ValueError("Notebook 05 repeat pilot contains no prompts")
    digest_agreement = matching_digest_count / repeat_prompt_count
    score_agreement = matching_score_count / repeat_prompt_count
    if (
        int(repeat_record.get("prompt_count", -1)) != repeat_prompt_count
        or int(repeat_record.get("matching_mask_digest_count", -1))
        != matching_digest_count
        or int(repeat_record.get("matching_score_count", -1))
        != matching_score_count
        or not math.isclose(
            float(repeat_record["mask_digest_agreement_rate"]),
            digest_agreement,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(repeat_record["score_agreement_rate"]),
            score_agreement,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Notebook 05 repeat agreement differs from frozen results")
    manifest_hashes = {
        "sam_safe_artifact_manifest_sha256": file_sha256(
            safe_root / "artifact_manifest.json"
        ),
        "sam_output_artifact_manifest_sha256": file_sha256(
            private_root / "artifact_manifest.json"
        ),
        "sam_run_record_sha256": file_sha256(safe_root / "run_record.json"),
        "sam_repeat_results_sha256": file_sha256(repeat_path),
        "sam_safe_manifest_content_sha256": str(
            safe_manifest["manifest_content_sha256"]
        ),
        "sam_output_manifest_content_sha256": str(
            private_manifest["manifest_content_sha256"]
        ),
    }
    return run_record, result_by_image, manifest_hashes


def _group_summaries(
    records: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(str(row[key]) for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        for summary in summarize_mask_routes(rows):
            output.append(
                {
                    **{name: value for name, value in zip(group_keys, key)},
                    **summary,
                }
            )
    return output


def _route_comparison_rows(
    route_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_route = {str(row["route"]): row for row in route_rows}
    metrics = (
        "detection_aware_recall_at_0_75",
        "detection_aware_mean_iou",
        "detection_aware_mean_dice",
        "detection_aware_mean_boundary_f1",
        "mean_prediction_spill_fraction",
    )
    output = []
    for metric in metrics:
        reference = float(by_route["reference_box"][metric])
        automatic = float(by_route["automatic_box"][metric])
        output.append(
            {
                "metric": metric,
                "reference_box": reference,
                "automatic_box": automatic,
                "automatic_minus_reference": automatic - reference,
            }
        )
    return output


def _bootstrap_rows(
    scene_rows: Sequence[Mapping[str, Any]],
    assets_by_scene: Mapping[str, str],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_scene_route = {
        (str(row["opaque_scene_id"]), str(row["route"])): row for row in scene_rows
    }
    metrics = (
        "detection_aware_recall_at_0_75",
        "detection_aware_mean_iou",
        "detection_aware_mean_dice",
        "detection_aware_mean_boundary_f1",
    )
    output = []
    for metric_index, metric in enumerate(metrics):
        paired = [
            {
                "opaque_scene_id": scene_id,
                "opaque_asset_group_id": assets_by_scene[scene_id],
                "baseline": float(by_scene_route[(scene_id, "reference_box")][metric]),
                "candidate": float(by_scene_route[(scene_id, "automatic_box")][metric]),
            }
            for scene_id in sorted(assets_by_scene)
        ]
        interval = paired_asset_clustered_bootstrap(
            paired,
            baseline_key="baseline",
            candidate_key="candidate",
            repetitions=repetitions,
            seed=seed + metric_index,
        )
        output.append({"metric": metric, **interval})
    return output


def _metric_definitions() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "statistical_unit": "scene",
        "views": "repeated observations within each scene",
        "routes": {
            "reference_box": "SAM 2.1 prompted with supplied-mask boxes",
            "automatic_box": "SAM 2.1 prompted with frozen LocateAnything boxes",
        },
        "mask_iou": "intersection divided by union for matched binary masks",
        "dice": "twice the intersection divided by total foreground area",
        "boundary_f1": (
            "boundary precision and recall within the fixed pixel tolerance"
        ),
        "detection_aware_metrics": (
            "missed references contribute zero; unmatched predictions remain counted"
        ),
        "confidence_interval": (
            "paired scene means with complete asset groups resampled together"
        ),
        "scope": "DS0 provisional integration evidence",
        "ranking_metrics": "B0, B1 and P1 are deferred to Notebook 07",
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _verified_media_paths(
    model_input_root: Path,
    expected_image_keys: set[tuple[str, str]],
    sam_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], Path]:
    package_record_path = model_input_root / "package_record.json"
    checksum_path = model_input_root / "content_checksums.sha256"
    if not package_record_path.is_file() or not checksum_path.is_file():
        raise ValueError("The model-input media package is incomplete")
    package_record = json.loads(package_record_path.read_text(encoding="utf-8"))
    if (
        package_record.get("all_checks_passed") is not True
        or int(package_record.get("image_count", -1)) != EXPECTED_IMAGES
        or int(package_record.get("scene_count", -1)) != EXPECTED_SCENES
        or int(package_record.get("view_count", -1)) != EXPECTED_VIEWS_PER_SCENE
    ):
        raise ValueError("The model-input media package does not match DS0")
    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or relative in checksums:
            raise ValueError("The model-input checksum file is invalid")
        checksums[relative] = digest
    media_paths: dict[tuple[str, str], Path] = {}
    for scene_id, view_id in sorted(expected_image_keys):
        relative = f"media/{scene_id}/rgb/{view_id}.png"
        path = model_input_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("A DS0 RGB image is missing or unsafe")
        digest = file_sha256(path)
        if checksums.get(relative) != digest:
            raise ValueError("A DS0 RGB image differs from the media package")
        if str(sam_rows[(scene_id, view_id)]["image_sha256"]) != digest:
            raise ValueError("A DS0 RGB image differs from the SAM input")
        media_paths[(scene_id, view_id)] = path
    if len(media_paths) != EXPECTED_IMAGES:
        raise ValueError("DS0 qualitative media coverage is incomplete")
    return media_paths


def _qualitative_case_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    matched = sorted(
        (
            row
            for row in records
            if row["route"] == "automatic_box"
            and row["record_kind"] == "matched_reference"
        ),
        key=lambda row: (
            float(row["mask_iou"]),
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        ),
    )
    if len(matched) < 6:
        raise ValueError("At least six matched automatic masks are required")
    middle = (len(matched) - 1) / 2
    candidate_groups = (
        ("Strong", list(reversed(matched))),
        (
            "Typical",
            sorted(
                enumerate(matched),
                key=lambda item: (
                    abs(item[0] - middle),
                    str(item[1]["opaque_scene_id"]),
                    str(item[1]["view_id"]),
                    str(item[1]["opaque_object_id"]),
                ),
            ),
        ),
        ("Challenging", matched),
    )
    selected: list[tuple[str, Mapping[str, Any]]] = []
    used_scenes: set[str] = set()
    for group, source in candidate_groups:
        candidates = [item[1] for item in source] if group == "Typical" else source
        group_rows = []
        for row in candidates:
            scene_id = str(row["opaque_scene_id"])
            if scene_id in used_scenes:
                continue
            group_rows.append(row)
            used_scenes.add(scene_id)
            if len(group_rows) == 2:
                break
        if len(group_rows) != 2:
            raise ValueError("Qualitative cases require six distinct scenes")
        selected.extend((group, row) for row in group_rows)
    identities = {
        (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        )
        for _group, row in selected
    }
    if len(identities) != len(selected):
        raise ValueError("Qualitative case selection produced duplicate objects")
    return selected


def _mask_on_rgb(
    rgb: Image.Image,
    mask: np.ndarray,
    colour: tuple[int, int, int, int],
) -> Image.Image:
    base = rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    binary = Image.fromarray(mask.astype("uint8") * 255, mode="L")
    fill = Image.new("RGBA", base.size, colour)
    overlay.alpha_composite(
        Image.composite(fill, Image.new("RGBA", base.size), binary)
    )
    return Image.alpha_composite(base, overlay).convert("RGB")


def _mask_agreement_panel(reference: np.ndarray, prediction: np.ndarray) -> Image.Image:
    panel = np.full((256, 256, 3), 245, dtype=np.uint8)
    reference_only = np.logical_and(reference, np.logical_not(prediction))
    prediction_only = np.logical_and(prediction, np.logical_not(reference))
    overlap = np.logical_and(reference, prediction)
    panel[reference_only] = (38, 139, 94)
    panel[prediction_only] = (214, 69, 84)
    panel[overlap] = (69, 119, 196)
    return Image.fromarray(panel, mode="RGB")


def _save_representative_case_grid(
    path: Path,
    manifest_path: Path,
    records: Sequence[Mapping[str, Any]],
    evaluator_root: Path,
    geometry_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
    prediction_masks: Mapping[tuple[str, str, str, str], np.ndarray],
    media_paths: Mapping[tuple[str, str], Path],
    *,
    threshold: int,
) -> None:
    selected = _qualitative_case_rows(records)
    panel_size = 220
    panel_gap = 8
    label_height = 50
    row_gap = 16
    header_height = 92
    columns = 4
    width = columns * panel_size + (columns - 1) * panel_gap
    height = header_height + len(selected) * (panel_size + label_height + row_gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        "Automatic-route SAM 2.1 examples",
        fill="#1f2933",
        font=_font(22, bold=True),
    )
    draw.text(
        (8, 36),
        "Two deterministic examples from the strong, typical and challenging IoU ranks.",
        fill="#52606d",
        font=_font(13),
    )
    legend = (
        ((38, 139, 94), "Reference only"),
        ((214, 69, 84), "Prediction only"),
        ((69, 119, 196), "Overlap"),
    )
    legend_x = 8
    for colour, label in legend:
        draw.rectangle((legend_x, 64, legend_x + 14, 78), fill=colour)
        draw.text((legend_x + 20, 62), label, fill="#1f2933", font=_font(11))
        legend_x += 145
    manifest_rows = []
    for index, (group, row) in enumerate(selected):
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        object_id = str(row["opaque_object_id"])
        prediction_id = str(row["prediction_id"])
        reference = _read_reference_mask(
            evaluator_root,
            geometry_by_key[(scene_id, view_id, object_id)],
            threshold=threshold,
        )
        prediction = prediction_masks[
            (scene_id, view_id, "automatic_box", prediction_id)
        ]
        with Image.open(media_paths[(scene_id, view_id)]) as opened:
            rgb = opened.convert("RGB")
        panels = (
            ("RGB", rgb),
            ("Supplied mask", _mask_on_rgb(rgb, reference, (38, 139, 94, 105))),
            ("Automatic SAM", _mask_on_rgb(rgb, prediction, (214, 69, 84, 105))),
            ("Mask agreement", _mask_agreement_panel(reference, prediction)),
        )
        y = header_height + index * (panel_size + label_height + row_gap) + 22
        for panel_index, (label, image) in enumerate(panels):
            x = panel_index * (panel_size + panel_gap)
            resampling = (
                Image.Resampling.NEAREST
                if label == "Mask agreement"
                else Image.Resampling.BILINEAR
            )
            resized = image.resize((panel_size, panel_size), resampling)
            canvas.paste(resized, (x, y))
            draw.rectangle(
                (x, y, x + panel_size - 1, y + panel_size - 1),
                outline="#c8d1da",
            )
            draw.text(
                (x, y - 19),
                label,
                fill="#1f2933",
                font=_font(12, bold=True),
            )
        draw.text(
            (0, y + panel_size + 5),
            (
                f"{group}: {scene_id} / {view_id} / {object_id} | "
                f"box IoU {float(row['matched_box_iou']):.3f} | "
                f"mask IoU {float(row['mask_iou']):.3f} | "
                f"boundary F1 {float(row['boundary_f1']):.3f}"
            ),
            fill="#1f2933",
            font=_font(12),
        )
        manifest_rows.append(
            {
                "display_order": index + 1,
                "quality_group": group.lower(),
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "opaque_object_id": object_id,
                "prediction_id": prediction_id,
                "matched_box_iou": row["matched_box_iou"],
                "mask_iou": row["mask_iou"],
                "dice": row["dice"],
                "boundary_f1": row["boundary_f1"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    write_csv_rows(
        manifest_path,
        [
            "display_order",
            "quality_group",
            "opaque_scene_id",
            "view_id",
            "opaque_object_id",
            "prediction_id",
            "matched_box_iou",
            "mask_iou",
            "dice",
            "boundary_f1",
        ],
        manifest_rows,
    )


def _save_worst_case_grid(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    evaluator_root: Path,
    geometry_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
    prediction_masks: Mapping[tuple[str, str, str, str], np.ndarray],
    *,
    threshold: int,
) -> None:
    candidates = [
        row
        for row in records
        if row["route"] == "automatic_box"
        and row["record_kind"] == "matched_reference"
    ]
    selected = sorted(
        candidates,
        key=lambda row: (
            float(row["mask_iou"]),
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
        ),
    )[:12]
    columns = 4
    cell = 292
    label_height = 50
    header_height = 82
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new(
        "RGB",
        (columns * cell, header_height + rows * (cell + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10),
        "Lowest-IoU automatic-route mask matches",
        fill="#1f2933",
        font=_font(22, bold=True),
    )
    legend = (
        ((38, 139, 94), "Reference only"),
        ((214, 69, 84), "Prediction only"),
        ((69, 119, 196), "Overlap"),
    )
    legend_x = 10
    for colour, label in legend:
        draw.rectangle((legend_x, 48, legend_x + 18, 66), fill=colour)
        draw.text(
            (legend_x + 26, 48),
            label,
            fill="#1f2933",
            font=_font(14),
        )
        legend_x += 165
    for index, row in enumerate(selected):
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        object_id = str(row["opaque_object_id"])
        prediction_id = str(row["prediction_id"])
        reference = _read_reference_mask(
            evaluator_root,
            geometry_by_key[(scene_id, view_id, object_id)],
            threshold=threshold,
        )
        prediction = prediction_masks[
            (scene_id, view_id, "automatic_box", prediction_id)
        ]
        panel = np.full((256, 256, 3), 245, dtype=np.uint8)
        reference_only = np.logical_and(reference, np.logical_not(prediction))
        prediction_only = np.logical_and(prediction, np.logical_not(reference))
        overlap = np.logical_and(reference, prediction)
        panel[reference_only] = (38, 139, 94)
        panel[prediction_only] = (214, 69, 84)
        panel[overlap] = (69, 119, 196)
        image = Image.fromarray(panel, mode="RGB").resize((272, 272))
        x = (index % columns) * cell + 10
        y = header_height + (index // columns) * (cell + label_height) + 5
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + 271, y + 271), outline="#c8d1da")
        draw.text(
            (x, y + 278),
            f"{scene_id} / {view_id} / {object_id}",
            fill="#1f2933",
            font=_font(13, bold=True),
        )
        draw.text(
            (x, y + 296),
            f"mask IoU {float(row['mask_iou']):.3f}",
            fill="#1f2933",
            font=_font(13),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _save_safe_charts(
    safe_root: Path,
    records: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    view_rows: Sequence[Mapping[str, Any]],
) -> None:
    matched_by_route = {
        route: [
            row
            for row in records
            if row["route"] == route and row["record_kind"] == "matched_reference"
        ]
        for route in ROUTES
    }
    for route in ROUTES:
        save_histogram(
            [float(row["mask_iou"]) for row in matched_by_route[route]],
            safe_root / f"{route}_mask_iou_distribution.png",
            title=f"{route.replace('_', ' ').title()} matched mask IoU",
            x_label="Mask IoU for matched references",
            bins=12,
            count_label="Matched object-view masks",
        )
        save_histogram(
            [float(row["boundary_f1"]) for row in matched_by_route[route]],
            safe_root / f"{route}_boundary_f1_distribution.png",
            title=f"{route.replace('_', ' ').title()} matched boundary F1",
            x_label="Boundary F1 for matched references",
            bins=12,
            count_label="Matched object-view masks",
        )
        route_views = sorted(
            (row for row in view_rows if row["route"] == route),
            key=lambda row: str(row["view_id"]),
        )
        save_bar_chart(
            [str(row["view_id"]) for row in route_views],
            [float(row["detection_aware_recall_at_0_75"]) for row in route_views],
            safe_root / f"{route}_recall_by_view.png",
            title=f"{route.replace('_', ' ').title()} detection-aware recall by view",
            y_label="Recall at mask IoU 0.75",
        )
    automatic = matched_by_route["automatic_box"]
    save_scatter_plot(
        [float(row["matched_box_iou"]) for row in automatic],
        [float(row["mask_iou"]) for row in automatic],
        safe_root / "automatic_box_prompt_iou_vs_mask_iou.png",
        title="Automatic prompt IoU and resulting mask IoU",
        x_label="Frozen proposal box IoU",
        y_label="SAM mask IoU",
    )
    summary_by_route = {str(row["route"]): row for row in route_rows}
    metric_labels = ["Recall@0.75", "Mean IoU", "Mean Dice", "Boundary F1"]
    values = []
    labels = []
    for route in ROUTES:
        for label, metric in zip(
            metric_labels,
            (
                "detection_aware_recall_at_0_75",
                "detection_aware_mean_iou",
                "detection_aware_mean_dice",
                "detection_aware_mean_boundary_f1",
            ),
        ):
            labels.append(f"{route.split('_')[0]} {label}")
            values.append(float(summary_by_route[route][metric]))
    save_bar_chart(
        labels,
        values,
        safe_root / "route_metric_comparison.png",
        title="SAM 2.1 route comparison",
        y_label="Metric value",
        show_values=True,
    )
    failure_labels = []
    failure_values = []
    for route in ROUTES:
        failure_labels.extend(
            [
                f"{route.split('_')[0]} misses",
                f"{route.split('_')[0]} extras",
            ]
        )
        failure_values.extend(
            [
                int(summary_by_route[route]["missed_reference_count"]),
                int(summary_by_route[route]["unmatched_prediction_count"]),
            ]
        )
    save_bar_chart(
        failure_labels,
        failure_values,
        safe_root / "misses_and_extras_by_route.png",
        title="Retained misses and unmatched predictions",
        y_label="Count",
        integer_ticks=True,
        show_values=True,
    )


def run_ds0_mask_evaluation(
    *,
    evaluator_root: Path,
    model_input_root: Path,
    proposal_output_root: Path,
    sam_output_root: Path,
    output_root: Path,
    evaluator_provenance: Mapping[str, Any],
    evaluation_mode: str = MODEL_EVIDENCE_MODE,
    mask_iou_threshold: float = DEFAULT_MASK_IOU_THRESHOLD,
    boundary_tolerance_pixels: int = DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the complete DS0 mask evaluator after all model inference finishes."""

    verified_evaluator_provenance = _validated_evaluator_provenance(
        evaluator_provenance
    )
    expanded_inputs = (
        evaluator_root.expanduser(),
        model_input_root.expanduser(),
        proposal_output_root.expanduser(),
        sam_output_root.expanduser(),
    )
    if any(path.is_symlink() for path in expanded_inputs):
        raise ValueError(
            "Evaluator, media, proposal and SAM roots cannot be symbolic links"
        )
    evaluator_root, model_input_root, proposal_output_root, sam_output_root = (
        path.resolve() for path in expanded_inputs
    )
    expanded_output = output_root.expanduser()
    if expanded_output.is_symlink():
        raise ValueError("Evaluation output root cannot be a symbolic link")
    output_root = expanded_output.resolve()
    input_roots = (
        evaluator_root,
        model_input_root,
        proposal_output_root,
        sam_output_root,
    )
    for index, first in enumerate(input_roots):
        for second in input_roots[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError(
                    "Evaluator, media, proposal and SAM inputs require separate roots"
                )
    if any(
        output_root == root
        or output_root in root.parents
        or root in output_root.parents
        for root in input_roots
    ):
        raise ValueError("Evaluation output requires a new root")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Evaluation output already exists: {output_root}")
    staging_root = output_root.with_name(output_root.name + ".incomplete")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(
            f"Incomplete evaluation output already exists: {staging_root}"
        )

    evaluator_record = verify_ds0_mask_evaluator_package(evaluator_root)
    geometry_record = json.loads(
        (evaluator_root / "oracle_geometry/package_record.json").read_text(
            encoding="utf-8"
        )
    )
    threshold = int(geometry_record["mask_threshold"])
    geometry_rows = read_jsonl(
        evaluator_root / "oracle_geometry/reference_manifest.jsonl"
    )
    crosswalk_rows = read_jsonl(
        evaluator_root / "sam_crosswalk/sam_evaluator_crosswalk.jsonl"
    )
    attribute_rows = read_jsonl(
        evaluator_root / "scene_attributes/protected_scene_attributes.jsonl"
    )
    geometry_by_key = {
        (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        ): row
        for row in geometry_rows
    }
    geometry_by_image: dict[
        tuple[str, str], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for key, row in geometry_by_key.items():
        geometry_by_image[(key[0], key[1])][key[2]] = row
    expected_image_keys = set(geometry_by_image)
    if (
        len(geometry_by_key) != EXPECTED_REFERENCE_PROMPTS
        or len(expected_image_keys) != EXPECTED_IMAGES
    ):
        raise ValueError("Evaluator geometry coverage is incomplete")

    crosswalk_by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    crosswalk_by_image_route: dict[
        tuple[str, str, str], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in crosswalk_rows:
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["route"]),
            str(row["prompt_key"]),
        )
        if key in crosswalk_by_key:
            raise ValueError("Evaluator crosswalk contains duplicate prompts")
        crosswalk_by_key[key] = row
        crosswalk_by_image_route[key[:3]][key[3]] = row
    if len(crosswalk_by_key) != EXPECTED_TOTAL_PROMPTS:
        raise ValueError("Evaluator crosswalk coverage is incomplete")
    crosswalk_record = json.loads(
        (evaluator_root / "sam_crosswalk/package_record.json").read_text(
            encoding="utf-8"
        )
    )
    expected_prompt_package_sha256 = str(crosswalk_record["prompt_package_sha256"])
    assets_by_scene = {
        str(row["opaque_scene_id"]): str(row["opaque_asset_group_id"])
        for row in attribute_rows
    }
    if len(assets_by_scene) != EXPECTED_SCENES:
        raise ValueError("Evaluator asset-group coverage is incomplete")

    (
        proposal_run_record,
        _proposal_rows,
        proposal_gate_summary,
        proposal_manifest_hashes,
    ) = _load_verified_locateanything(
        proposal_output_root,
        expected_image_keys=expected_image_keys,
        geometry_rows=geometry_rows,
        expected_proposal_results_sha256=str(
            crosswalk_record["proposal_results_sha256"]
        ),
    )

    sam_run_record, sam_rows, sam_manifest_hashes = _load_verified_sam_rows(
        sam_output_root,
        evaluation_mode=evaluation_mode,
        expected_prompt_package_sha256=expected_prompt_package_sha256,
        crosswalk_by_key=crosswalk_by_key,
        expected_image_keys=expected_image_keys,
    )
    media_paths = _verified_media_paths(
        model_input_root,
        expected_image_keys,
        sam_rows,
    )

    staging_root.mkdir(parents=True)
    safe_root = staging_root / "safe"
    private_root = staging_root / "private"
    safe_root.mkdir()
    private_root.mkdir()
    try:
        records: list[dict[str, Any]] = []
        mask_route_rows: list[dict[str, Any]] = []
        prediction_masks: dict[tuple[str, str, str, str], np.ndarray] = {}
        decoded_mask_count = 0
        for scene_id, view_id in sorted(expected_image_keys):
            references = {
                object_id: _read_reference_mask(
                    evaluator_root,
                    row,
                    threshold=threshold,
                )
                for object_id, row in sorted(
                    geometry_by_image[(scene_id, view_id)].items()
                )
            }
            image_row = sam_rows[(scene_id, view_id)]
            route_package_items: dict[str, list[dict[str, Any]]] = {
                "supplied_mask": [],
                "reference_box": [],
                "automatic_box": [],
            }
            for route in ROUTES:
                prediction_rows = []
                prediction_metadata: dict[str, dict[str, Any]] = {}
                crosswalk = crosswalk_by_image_route[(scene_id, view_id, route)]
                for item in image_row["routes"][route]:
                    prompt_key = str(item["prompt_key"])
                    crosswalk_row = crosswalk.get(prompt_key)
                    if crosswalk_row is None:
                        raise ValueError("A SAM result has no evaluator crosswalk row")
                    validated = _validated_prediction(
                        item,
                        expected_box=crosswalk_row["prompt_box_xyxy"],
                    )
                    decoded_mask_count += 1
                    prediction_masks[
                        (scene_id, view_id, route, prompt_key)
                    ] = validated["mask"]
                    route_package_items[route].append(
                        {
                            "prompt_key": prompt_key,
                            "box_xyxy": [int(value) for value in item["box_xyxy"]],
                            "foreground_pixels": validated["foreground_pixels"],
                            "mask_status": item["mask_status"],
                            "mask_sha256": validated["mask_sha256"],
                            "mask_rle": item["mask_rle"],
                            "predicted_quality_score": validated[
                                "predicted_quality_score"
                            ],
                        }
                    )
                    matched_reference = crosswalk_row.get(
                        "matched_opaque_object_id"
                    )
                    prediction_rows.append(
                        {
                            "prediction_id": prompt_key,
                            "matched_reference_id": matched_reference,
                            "mask": validated["mask"],
                            "prompt_box_xyxy": crosswalk_row["prompt_box_xyxy"],
                        }
                    )
                    prediction_metadata[prompt_key] = {
                        "matched_box_iou": crosswalk_row.get("matched_box_iou"),
                        "predicted_quality_score": validated[
                            "predicted_quality_score"
                        ],
                        "mask_sha256": validated["mask_sha256"],
                    }
                route_records = evaluate_mask_route_image(
                    opaque_scene_id=scene_id,
                    view_id=view_id,
                    route=route,
                    reference_masks=references,
                    prediction_rows=prediction_rows,
                    mask_iou_threshold=mask_iou_threshold,
                    boundary_tolerance_pixels=boundary_tolerance_pixels,
                )
                for record in route_records:
                    prediction_id = record.get("prediction_id")
                    metadata = (
                        prediction_metadata[str(prediction_id)]
                        if prediction_id is not None
                        else {
                            "matched_box_iou": None,
                            "predicted_quality_score": None,
                            "mask_sha256": None,
                        }
                    )
                    records.append(
                        {
                            **record,
                            "opaque_asset_group_id": assets_by_scene[scene_id],
                            **metadata,
                        }
                    )
            reference_crosswalk = crosswalk_by_image_route[
                (scene_id, view_id, "reference_box")
            ]
            for prompt_key, crosswalk_row in sorted(reference_crosswalk.items()):
                object_id = str(crosswalk_row["opaque_object_id"])
                mask = references[object_id]
                foreground = int(mask.sum())
                route_package_items["supplied_mask"].append(
                    {
                        "prompt_key": prompt_key,
                        "box_xyxy": [
                            int(value) for value in crosswalk_row["prompt_box_xyxy"]
                        ],
                        "foreground_pixels": foreground,
                        "mask_status": "non_empty",
                        "mask_sha256": binary_mask_sha256(mask),
                        "mask_rle": encode_binary_mask(mask),
                    }
                )
            for route in route_package_items:
                route_package_items[route] = sorted(
                    route_package_items[route], key=lambda item: item["prompt_key"]
                )
            mask_route_rows.append(
                {
                    "schema_version": 1,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "image_sha256": str(image_row["image_sha256"]),
                    "routes": route_package_items,
                }
            )
        if decoded_mask_count != EXPECTED_TOTAL_PROMPTS:
            raise ValueError("Decoded SAM mask coverage is incomplete")

        route_rows = summarize_mask_routes(records)
        image_rows = _group_summaries(records, ("opaque_scene_id", "view_id"))
        scene_rows = _group_summaries(records, ("opaque_scene_id",))
        for row in scene_rows:
            row["opaque_asset_group_id"] = assets_by_scene[
                str(row["opaque_scene_id"])
            ]
        view_rows = _group_summaries(records, ("view_id",))
        route_comparison = _route_comparison_rows(route_rows)
        bootstrap_rows = _bootstrap_rows(
            scene_rows,
            assets_by_scene,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )

        _write_jsonl(private_root / "mask_evaluation_records.jsonl", records)
        _write_jsonl(private_root / "image_route_metrics.jsonl", image_rows)
        _write_jsonl(private_root / "scene_route_metrics.jsonl", scene_rows)
        worst_rows = sorted(
            (
                row
                for row in records
                if row["route"] == "automatic_box"
                and row["record_kind"] != "unmatched_prediction"
            ),
            key=lambda row: (
                float(row["mask_iou"]),
                str(row["opaque_scene_id"]),
                str(row["view_id"]),
            ),
        )[:100]
        write_csv_rows(
            private_root / "worst_automatic_cases.csv",
            [
                "opaque_scene_id",
                "view_id",
                "opaque_object_id",
                "prediction_id",
                "record_kind",
                "matched_box_iou",
                "mask_iou",
                "dice",
                "boundary_f1",
            ],
            [
                {
                    column: row.get(column)
                    for column in (
                        "opaque_scene_id",
                        "view_id",
                        "opaque_object_id",
                        "prediction_id",
                        "record_kind",
                        "matched_box_iou",
                        "mask_iou",
                        "dice",
                        "boundary_f1",
                    )
                }
                for row in worst_rows
            ],
        )
        _save_worst_case_grid(
            private_root / "worst_automatic_mask_pairs.png",
            records,
            evaluator_root,
            geometry_by_key,
            prediction_masks,
            threshold=threshold,
        )
        _save_representative_case_grid(
            private_root / "representative_automatic_mask_cases.png",
            private_root / "qualitative_case_manifest.csv",
            records,
            evaluator_root,
            geometry_by_key,
            prediction_masks,
            media_paths,
            threshold=threshold,
        )

        completion_gates = [
            {
                "gate": "verified evaluator package",
                "observed": bool(evaluator_record.get("all_checks_passed")),
                "expected": True,
                "status": "pass",
            },
            {
                "gate": "verified LocateAnything full model run",
                "observed": proposal_run_record["producer_kind"],
                "expected": "locateanything_model",
                "status": "pass",
            },
            {
                "gate": "LocateAnything repeat agreement",
                "observed": proposal_run_record["repeat_agreement"][
                    "box_agreement_rate"
                ],
                "expected": REPEAT_AGREEMENT_MINIMUM,
                "status": "pass",
            },
            {
                "gate": "accepted Notebook 05 run mode",
                "observed": sam_run_record["producer_kind"],
                "expected": (
                    MODEL_PRODUCER_KIND
                    if evaluation_mode == MODEL_EVIDENCE_MODE
                    else CONTRACT_PRODUCER_KIND
                ),
                "status": "pass",
            },
            {
                "gate": "scene coverage",
                "observed": len(assets_by_scene),
                "expected": EXPECTED_SCENES,
                "status": "pass" if len(assets_by_scene) == EXPECTED_SCENES else "fail",
            },
            {
                "gate": "image coverage",
                "observed": len(sam_rows),
                "expected": EXPECTED_IMAGES,
                "status": "pass" if len(sam_rows) == EXPECTED_IMAGES else "fail",
            },
            {
                "gate": "decoded SAM masks",
                "observed": decoded_mask_count,
                "expected": EXPECTED_TOTAL_PROMPTS,
                "status": (
                    "pass"
                    if decoded_mask_count == EXPECTED_TOTAL_PROMPTS
                    else "fail"
                ),
            },
            {
                "gate": "reference route prompts",
                "observed": next(
                    row for row in route_rows if row["route"] == "reference_box"
                )["prediction_count"],
                "expected": EXPECTED_REFERENCE_PROMPTS,
                "status": "pass",
            },
            {
                "gate": "automatic route prompts",
                "observed": next(
                    row for row in route_rows if row["route"] == "automatic_box"
                )["prediction_count"],
                "expected": EXPECTED_AUTOMATIC_PROMPTS,
                "status": "pass",
            },
        ]
        completion_passed = all(row["status"] == "pass" for row in completion_gates)
        if not completion_passed:
            raise RuntimeError("Notebook 06 completion gates failed")

        route_rows_by_name = {str(row["route"]): row for row in route_rows}
        proposal_repeat = float(
            proposal_run_record["repeat_agreement"]["box_agreement_rate"]
        )
        sam_repeat = min(
            float(sam_run_record["repeat_agreement"]["mask_digest_agreement_rate"]),
            float(sam_run_record["repeat_agreement"]["score_agreement_rate"]),
        )
        decision_gates = [
            {
                "gate": "proposal exact count",
                "observed": float(proposal_gate_summary["exact_count_rate"]),
                "threshold": PROPOSAL_EXACT_COUNT_MINIMUM,
                "status": "pass"
                if float(proposal_gate_summary["exact_count_rate"])
                >= PROPOSAL_EXACT_COUNT_MINIMUM
                else "fail",
            },
            {
                "gate": "proposal box recall at 0.50",
                "observed": float(proposal_gate_summary["micro_recall_at_0_50"]),
                "threshold": PROPOSAL_RECALL_AT_0_50_MINIMUM,
                "status": "pass"
                if float(proposal_gate_summary["micro_recall_at_0_50"])
                >= PROPOSAL_RECALL_AT_0_50_MINIMUM
                else "fail",
            },
            {
                "gate": "proposal repeat agreement",
                "observed": proposal_repeat,
                "threshold": REPEAT_AGREEMENT_MINIMUM,
                "status": "pass"
                if proposal_repeat >= REPEAT_AGREEMENT_MINIMUM
                else "fail",
            },
            {
                "gate": "SAM repeat agreement",
                "observed": sam_repeat,
                "threshold": REPEAT_AGREEMENT_MINIMUM,
                "status": "pass" if sam_repeat >= REPEAT_AGREEMENT_MINIMUM else "fail",
            },
            {
                "gate": "reference route mask recall at 0.75",
                "observed": float(
                    route_rows_by_name["reference_box"][
                        "detection_aware_recall_at_0_75"
                    ]
                ),
                "threshold": REFERENCE_MASK_RECALL_MINIMUM,
                "status": "pass"
                if float(
                    route_rows_by_name["reference_box"][
                        "detection_aware_recall_at_0_75"
                    ]
                )
                >= REFERENCE_MASK_RECALL_MINIMUM
                else "fail",
            },
            {
                "gate": "reference route mean matched mask IoU",
                "observed": float(
                    route_rows_by_name["reference_box"]["mean_matched_iou"]
                ),
                "threshold": REFERENCE_MEAN_MATCHED_IOU_MINIMUM,
                "status": "pass"
                if float(route_rows_by_name["reference_box"]["mean_matched_iou"])
                >= REFERENCE_MEAN_MATCHED_IOU_MINIMUM
                else "fail",
            },
            {
                "gate": "automatic route mask recall at 0.75",
                "observed": float(
                    route_rows_by_name["automatic_box"][
                        "detection_aware_recall_at_0_75"
                    ]
                ),
                "threshold": AUTOMATIC_MASK_RECALL_MINIMUM,
                "status": "pass"
                if float(
                    route_rows_by_name["automatic_box"][
                        "detection_aware_recall_at_0_75"
                    ]
                )
                >= AUTOMATIC_MASK_RECALL_MINIMUM
                else "fail",
            },
            {
                "gate": "automatic route mean matched mask IoU",
                "observed": float(
                    route_rows_by_name["automatic_box"]["mean_matched_iou"]
                ),
                "threshold": AUTOMATIC_MEAN_MATCHED_IOU_MINIMUM,
                "status": "pass"
                if float(route_rows_by_name["automatic_box"]["mean_matched_iou"])
                >= AUTOMATIC_MEAN_MATCHED_IOU_MINIMUM
                else "fail",
            },
        ]
        scientific_gates_passed = all(
            row["status"] == "pass" for row in decision_gates
        )
        write_csv_rows(
            safe_root / "completion_gates.csv",
            ["gate", "observed", "expected", "status"],
            completion_gates,
        )
        write_csv_rows(
            safe_root / "route_summary.csv",
            ROUTE_SUMMARY_COLUMNS,
            route_rows,
        )
        write_csv_rows(
            safe_root / "view_route_summary.csv",
            VIEW_ROUTE_SUMMARY_COLUMNS,
            [
                {column: row.get(column) for column in VIEW_ROUTE_SUMMARY_COLUMNS}
                for row in view_rows
            ],
        )
        write_csv_rows(
            safe_root / "route_comparison.csv",
            ["metric", "reference_box", "automatic_box", "automatic_minus_reference"],
            route_comparison,
        )
        write_csv_rows(
            safe_root / "bootstrap_intervals.csv",
            [
                "metric",
                "direction",
                "scene_count",
                "cluster_count",
                "baseline_mean",
                "candidate_mean",
                "mean_difference",
                "confidence_level",
                "ci_lower",
                "ci_upper",
                "repetitions",
                "seed",
            ],
            bootstrap_rows,
        )
        proposal_gate_rows = [
            {"metric": key, "value": value}
            for key, value in proposal_gate_summary.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        write_csv_rows(
            safe_root / "proposal_gate_summary.csv",
            ["metric", "value"],
            proposal_gate_rows,
        )
        write_csv_rows(
            safe_root / "decision_gates.csv",
            ["gate", "observed", "threshold", "status"],
            decision_gates,
        )
        _write_json(safe_root / "metric_definitions.json", _metric_definitions())
        _save_safe_charts(safe_root, records, route_rows, view_rows)

        mask_route_record = build_mask_route_package(
            rows=mask_route_rows,
            output_root=staging_root / "model_input",
            provenance={
                "sam_run_record_sha256": sam_manifest_hashes[
                    "sam_run_record_sha256"
                ],
                "sam_prompt_package_sha256": expected_prompt_package_sha256,
                "proposal_results_sha256": proposal_manifest_hashes[
                    "proposal_results_sha256"
                ],
                "supplied_mask_record_sha256": file_sha256(
                    evaluator_root / "oracle_geometry/package_record.json"
                ),
                "evaluator_snapshot_sha256": verified_evaluator_provenance[
                    "snapshot_sha256"
                ],
            },
        )
        if mask_route_record.get("all_checks_passed") is not True:
            raise RuntimeError("Notebook 07 mask-route package verification failed")

        evidence_eligible = evaluation_mode == MODEL_EVIDENCE_MODE
        proposal_git_dirty = bool(proposal_run_record["provenance"]["git_dirty"])
        sam_git_dirty_value = sam_run_record.get("provenance", {}).get("git_dirty")
        sam_git_dirty = (
            bool(sam_git_dirty_value)
            if isinstance(sam_git_dirty_value, bool)
            else True
        )
        evaluator_git_dirty = bool(verified_evaluator_provenance["git_dirty"])
        if (
            proposal_git_dirty
            or sam_git_dirty
            or evaluator_git_dirty
            or not evidence_eligible
        ):
            decision_status = "candidate"
        elif scientific_gates_passed:
            decision_status = "ds0_gate_passed"
        else:
            decision_status = "ds0_gate_failed"
        run_record = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "stage": EVALUATION_STAGE,
            "status": "completed",
            "tracking_status": "pending",
            "evaluation_mode": evaluation_mode,
            "evidence_eligible": evidence_eligible,
            "completion_gate_passed": completion_passed,
            "scientific_gates_passed": scientific_gates_passed,
            "quality_claim": (
                "provisional_ds0_integration_evidence"
                if evidence_eligible
                else "engineering_contract_test_only"
            ),
            "sam_producer_kind": sam_run_record["producer_kind"],
            "python_version": platform.python_version(),
            "evaluator_provenance": verified_evaluator_provenance,
            "settings": {
                "mask_iou_threshold": float(mask_iou_threshold),
                "boundary_tolerance_pixels": int(boundary_tolerance_pixels),
                "bootstrap_repetitions": int(bootstrap_repetitions),
                "bootstrap_seed": int(bootstrap_seed),
            },
            "coverage": {
                "scene_count": len(assets_by_scene),
                "view_count_per_scene": EXPECTED_VIEWS_PER_SCENE,
                "image_count": len(sam_rows),
                "reference_mask_count": len(geometry_by_key),
                "sam_mask_count": decoded_mask_count,
                "reference_prompt_count": int(
                    next(row for row in route_rows if row["route"] == "reference_box")[
                        "prediction_count"
                    ]
                ),
                "automatic_prompt_count": int(
                    next(row for row in route_rows if row["route"] == "automatic_box")[
                        "prediction_count"
                    ]
                ),
            },
            "inputs": {
                "evaluator_package_record_sha256": file_sha256(
                    evaluator_root / "package_record.json"
                ),
                "sam_configuration_sha256": str(
                    sam_run_record["configuration_sha256"]
                ),
                "sam_prompt_package_sha256": expected_prompt_package_sha256,
                "mask_route_package_record_sha256": file_sha256(
                    staging_root / "model_input/package_record.json"
                ),
                **sam_manifest_hashes,
                **proposal_manifest_hashes,
            },
            "proposal_gate_summary": proposal_gate_summary,
            "route_summary": route_rows,
            "bootstrap_summary": bootstrap_rows,
            "candidate_decision": {
                "status": decision_status,
                "proposal_git_dirty": proposal_git_dirty,
                "sam_git_dirty": sam_git_dirty,
                "evaluator_git_dirty": evaluator_git_dirty,
                "proceed_to_ds1": bool(scientific_gates_passed and evidence_eligible),
                "final_route_acceptance": False,
                "reason": (
                    "DS0 gates support the next development stage; route "
                    "acceptance remains provisional"
                    if scientific_gates_passed and evidence_eligible
                    else "DS0 gates require review before the next development stage"
                ),
            },
            "ranking_metrics_status": "deferred_to_notebook_07",
        }
        _write_json(safe_root / "run_record.json", run_record)
        _write_artifact_manifest(private_root)
        _write_artifact_manifest(safe_root)
        verify_artifact_manifest(private_root)
        verify_artifact_manifest(safe_root)
        os.replace(staging_root, output_root)
        return run_record
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


__all__ = [
    "CONTRACT_TEST_MODE",
    "EVALUATION_STAGE",
    "MODEL_EVIDENCE_MODE",
    "ROUTE_SUMMARY_COLUMNS",
    "VIEW_ROUTE_SUMMARY_COLUMNS",
    "run_ds0_mask_evaluation",
    "verify_artifact_manifest",
]
