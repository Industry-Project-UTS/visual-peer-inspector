#!/usr/bin/env python3
"""Run the frozen automatic route on the optional eligible full pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_frozen_development_pipeline as frozen
import run_protected_cohort_inference as protected

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.frozen_resource import publish_frozen_resource
from visual_peer_inspector.tracking_safe_artifacts import write_tracking_safe_manifest
from visual_peer_inspector.workflows.full_pool_sensitivity import (
    COHORT_ID,
    STAGE_INFERENCE,
    load_full_pool_release_receipt,
)
from visual_peer_inspector.workflows.protected_cohort_inference import (
    load_method_freeze,
    load_policy_freeze,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _tracking(args: argparse.Namespace, safe_root: Path, tag: str) -> None:
    receipt = safe_root.parent / "tracking" / "clearml_receipt.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/finalize_full_pool_clearml.py"),
        "--safe-root",
        str(safe_root),
        "--receipt",
        str(receipt),
        "--stage",
        STAGE_INFERENCE,
        "--task-name",
        f"11A full-pool inference {args.run_mode}",
        "--tag",
        tag,
        "--mode",
        args.clearml_mode,
        "--wall-time-seconds",
        str(args.clearml_wall_time),
    ]
    try:
        subprocess.run(command, check=False, timeout=args.clearml_wall_time + 30)
    except subprocess.TimeoutExpired:
        _write_json(receipt, {"schema_version": 1, "status": "timed_out", "aggregate_only": True})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--application-contract", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--selected-prompt", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mounted-root", type=Path, action="append", default=[])
    parser.add_argument("--locate-model-source", type=Path)
    parser.add_argument("--run-mode", choices=("gate", "incremental", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--maximum-new-shards", type=int, default=10)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--recovery-output-directory", type=Path)
    parser.add_argument("--prune-after-recovery", action="store_true")
    parser.add_argument("--clearml-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--clearml-wall-time", type=int, default=180)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--maximum-preflight-images", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    release, contract = load_full_pool_release_receipt(args.release_receipt)
    method = load_method_freeze(
        args.method_freeze,
        application_contract_path=args.application_contract,
        require_resource=True,
    )
    policy = load_policy_freeze(
        args.policy_freeze,
        method_sha256=method["sha256"],
        require_resource=True,
    )
    if method["sha256"] != release["method_record_sha256"] or policy["sha256"] != release["policy_record_sha256"]:
        raise ValueError("Notebook 11A must use the exact method and policy recorded after Notebook 10B")
    protected.CONTRACTS["full-pool"] = contract
    protected.RUN_STAGES["full-pool"] = STAGE_INFERENCE
    if args.preflight_only:
        if args.preflight_output is None:
            raise ValueError("--preflight-output is required with --preflight-only")
        frozen.PREFLIGHT_COHORTS["full-pool"] = contract
        result = frozen.run_preflight(
            argparse.Namespace(
                cohort="full-pool",
                model_input_root=args.model_input_root,
                selected_prompt=args.selected_prompt,
                sam_checkpoint=args.sam_checkpoint,
                output_path=args.preflight_output,
                locate_model_source=args.locate_model_source,
                device=args.device,
                maximum_image_attempts=args.maximum_preflight_images,
            )
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    run_args = argparse.Namespace(
        cohort="full-pool",
        model_input_root=args.model_input_root,
        method_freeze=args.method_freeze,
        application_contract=args.application_contract,
        policy_freeze=args.policy_freeze,
        selected_prompt=args.selected_prompt,
        sam_checkpoint=args.sam_checkpoint,
        output_root=args.output_root,
        mounted_root=args.mounted_root,
        locate_model_source=args.locate_model_source,
        run_mode=args.run_mode,
        device=args.device,
        shard_size=args.shard_size,
        maximum_new_shards=args.maximum_new_shards,
        recovery_bundle=args.recovery_bundle,
        recovery_manifest=args.recovery_manifest,
        recovery_output_directory=args.recovery_output_directory,
        prune_after_recovery=args.prune_after_recovery,
    )
    result = protected.run(run_args)
    complete = result.get("status") == "completed"
    safe_root = args.output_root / COHORT_ID / "safe" if complete else args.output_root / "tracking-safe"
    run_path = safe_root / "run_record.json"
    if run_path.is_file():
        safe_run = json.loads(run_path.read_text(encoding="utf-8"))
    else:
        safe_root.mkdir(parents=True, exist_ok=True)
        safe_run = {
            "schema_version": "visual-peer-11a-run/v1",
            "status": "checkpointed",
            "stage": STAGE_INFERENCE,
            "cohort_id": COHORT_ID,
        }
    safe_run.update({
        "descriptive_sensitivity_only": True,
        "no_retuning": True,
        "primary_results_unchanged": True,
        "official_results_unchanged": True,
        "release_receipt_sha256": release["sha256"],
    })
    _write_json(run_path, safe_run)
    write_tracking_safe_manifest(safe_root)
    if complete:
        lock_path = args.output_root / COHORT_ID / "model-safe/prediction_lock.json"
        receipt = {
            "schema_version": "visual-peer-11a-inference-receipt/v1",
            "status": "complete",
            "stage": STAGE_INFERENCE,
            "cohort_id": COHORT_ID,
            "release_receipt_sha256": release["sha256"],
            "method_record_sha256": method["sha256"],
            "policy_record_sha256": policy["sha256"],
            "prediction_lock_sha256": file_sha256(lock_path),
            "prediction_run_record_sha256": file_sha256(run_path),
            "descriptive_sensitivity_only": True,
            "no_retuning": True,
        }
        receipt["receipt_sha256"] = __import__("hashlib").sha256(canonical_json_bytes(receipt)).hexdigest()
        receipt_path = args.output_root / "full_pool_inference_receipt.json"
        _write_json(receipt_path, receipt)
        publish_frozen_resource(
            resource_root=args.output_root / "publish/post-freeze-inference",
            stage=STAGE_INFERENCE,
            files={"full_pool_inference_receipt.json": receipt_path},
            bindings={
                "release_receipt_sha256": release["sha256"],
                "method_record_sha256": method["sha256"],
                "policy_record_sha256": policy["sha256"],
                "prediction_lock_sha256": receipt["prediction_lock_sha256"],
            },
        )
    _tracking(args, safe_root, "final" if complete else "recovery")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
