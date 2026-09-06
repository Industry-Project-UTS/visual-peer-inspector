#!/usr/bin/env python3
"""Evaluate the optional full-pool predictions without changing the method."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.frozen_resource import publish_frozen_resource
from visual_peer_inspector.tracking_safe_artifacts import write_tracking_safe_manifest
from visual_peer_inspector.workflows import one_shot_evaluation as evaluator
from visual_peer_inspector.workflows.full_pool_sensitivity import (
    ASSET_GROUP_COUNT,
    COHORT_ID,
    IMAGE_COUNT,
    MODEL_MEMBERSHIP_SHA256,
    SCENE_COUNT,
    STAGE_EVALUATION,
    load_full_pool_release_receipt,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _load_inference_receipt(path: Path, release_sha256: str) -> dict:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    recorded = value.pop("receipt_sha256", None)
    if (
        value.get("schema_version") != "visual-peer-11a-inference-receipt/v1"
        or value.get("status") != "complete"
        or value.get("cohort_id") != COHORT_ID
        or value.get("release_receipt_sha256") != release_sha256
        or value.get("descriptive_sensitivity_only") is not True
        or value.get("no_retuning") is not True
        or recorded != __import__("hashlib").sha256(canonical_json_bytes(value)).hexdigest()
    ):
        raise ValueError("Notebook 11A inference receipt is invalid")
    return {**value, "receipt_sha256": recorded, "file_sha256": file_sha256(source)}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--release-receipt", type=Path, required=True)
    common.add_argument("--inference-receipt", type=Path, required=True)
    common.add_argument("--mount-root", type=Path, action="append", required=True)
    contract = subparsers.add_parser("contract", parents=[common])
    contract.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--evaluation-contract", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    evaluate.add_argument("--panel-count", type=int, default=16)
    track = subparsers.add_parser("track")
    track.add_argument("--result-root", type=Path, required=True)
    track.add_argument("--mode", choices=("online", "offline", "disabled"), default="disabled")
    track.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _configure(args: argparse.Namespace) -> tuple[dict, dict]:
    release, contract = load_full_pool_release_receipt(args.release_receipt)
    inference = _load_inference_receipt(args.inference_receipt, release["sha256"])
    if (
        inference["method_record_sha256"] != release["method_record_sha256"]
        or inference["policy_record_sha256"] != release["policy_record_sha256"]
    ):
        raise ValueError("Full-pool inference changed the frozen method or policy")
    evaluator.EXPECTED_COHORTS["11b"] = {
        COHORT_ID: {
            "role": "post_freeze_sensitivity",
            "split_name": "eligible-full-pool",
            "membership_sha256": MODEL_MEMBERSHIP_SHA256,
            "source_membership_sha256": contract.membership_sha256,
            "scene_count": SCENE_COUNT,
            "image_count": IMAGE_COUNT,
            "asset_group_count": ASSET_GROUP_COUNT,
        }
    }
    evaluator.EVALUATION_RULES["11b"] = {
        "prediction_reference_mapping": "maximum_total_iou_one_to_one",
        "minimum_box_iou": evaluator.OFFICIAL_CORRESPONDENCE_MINIMUM_IOU,
    }
    return release, inference


def _finalize_result(output_root: Path, release: dict, inference: dict) -> dict:
    output = Path(output_root).expanduser().resolve(strict=True)
    safe_root = output / "safe"
    run_path = safe_root / "run_record.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update({
        "stage": STAGE_EVALUATION,
        "split_reporting": "eligible_full_pool_descriptive_only",
        "descriptive_sensitivity_only": True,
        "no_retuning": True,
        "primary_results_unchanged": True,
        "official_results_unchanged": True,
        "release_receipt_sha256": release["sha256"],
        "inference_receipt_sha256": inference["file_sha256"],
    })
    _write_json(run_path, run)
    manifest = write_tracking_safe_manifest(safe_root)
    completion_path = output / "evaluation_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.update({
        "stage": STAGE_EVALUATION,
        "safe_manifest_sha256": file_sha256(safe_root / "artifact_manifest.json"),
        "descriptive_sensitivity_only": True,
        "no_retuning": True,
        "primary_results_unchanged": True,
        "official_results_unchanged": True,
    })
    _write_json(completion_path, completion)
    result = {
        "schema_version": "visual-peer-11b-sensitivity-result/v1",
        "status": "complete",
        "stage": STAGE_EVALUATION,
        "cohort_id": COHORT_ID,
        "release_receipt_sha256": release["sha256"],
        "inference_receipt_sha256": inference["file_sha256"],
        "evaluation_contract_sha256": str(completion["contract_sha256"]),
        "safe_manifest_sha256": file_sha256(safe_root / "artifact_manifest.json"),
        "safe_artifact_count": int(manifest["artifact_count"]),
        "descriptive_sensitivity_only": True,
        "no_retuning": True,
        "primary_results_unchanged": True,
        "official_results_unchanged": True,
    }
    result["result_sha256"] = __import__("hashlib").sha256(canonical_json_bytes(result)).hexdigest()
    result_path = output / "full_pool_sensitivity_result.json"
    _write_json(result_path, result)
    publish_frozen_resource(
        resource_root=output / "publish/post-freeze-sensitivity-result",
        stage=STAGE_EVALUATION,
        files={"full_pool_sensitivity_result.json": result_path},
        bindings={
            "release_receipt_sha256": release["sha256"],
            "inference_receipt_sha256": inference["file_sha256"],
            "safe_manifest_sha256": result["safe_manifest_sha256"],
        },
    )
    return result


def _track(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts/finalize_full_pool_clearml.py"),
        "--safe-root",
        str(args.result_root / "safe"),
        "--receipt",
        str(args.result_root / "tracking/clearml_receipt.json"),
        "--stage",
        STAGE_EVALUATION,
        "--task-name",
        "11B full-pool descriptive sensitivity",
        "--tag",
        "final",
        "--mode",
        args.mode,
        "--wall-time-seconds",
        str(args.wall_time_seconds),
    ]
    completed = subprocess.run(command, check=False, timeout=args.wall_time_seconds + 30)
    return {"status": "completed" if completed.returncode == 0 else "tracking_failed", "returncode": completed.returncode}


def main() -> None:
    args = parse_arguments()
    if args.command == "track":
        result = _track(args)
    else:
        release, inference = _configure(args)
        if args.command == "contract":
            result = evaluator.create_evaluation_contract(
                stage="11b", mounted_roots=args.mount_root, output_path=args.output
            )
        else:
            evaluator.run_one_shot_evaluation(
                contract_path=args.evaluation_contract,
                mounted_roots=args.mount_root,
                output_root=args.output_root,
                expected_stage="11b",
                bootstrap_repetitions=args.bootstrap_repetitions,
                panel_count=args.panel_count,
            )
            result = _finalize_result(args.output_root, release, inference)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
