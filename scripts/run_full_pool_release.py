#!/usr/bin/env python3
"""Prepare the optional post-freeze full-pool sensitivity resources."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import canonical_json_bytes
from visual_peer_inspector.data.toysad_selection import SourcePaths
from visual_peer_inspector.frozen_resource import publish_frozen_resource
from visual_peer_inspector.tracking_safe_artifacts import write_tracking_safe_manifest
from visual_peer_inspector.workflows.full_pool_sensitivity import (
    ASSET_GROUP_COUNT,
    COHORT_ID,
    IMAGE_COUNT,
    MODEL_MEMBERSHIP_SHA256,
    PRIMARY_ROLE_COUNTS,
    SCENE_COUNT,
    STAGE_RELEASE,
    build_full_pool_resource,
    estimate_full_pool_rgb_bytes,
    prepare_full_pool_release,
)
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    prune_verified_state,
    restore_recovery_bundle,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_bytes(canonical_json_bytes(dict(value)) + b"\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _render_roles(path: Path) -> None:
    image = Image.new("RGB", (960, 430), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 24), "Eligible full pool with primary roles preserved", fill=(25, 42, 62))
    maximum = max(PRIMARY_ROLE_COUNTS.values())
    for index, (role, count) in enumerate(PRIMARY_ROLE_COUNTS.items()):
        y = 90 + index * 72
        width = int(660 * count / maximum)
        draw.text((35, y + 7), role.replace("_", " "), fill=(25, 42, 62))
        draw.rectangle((205, y, 205 + width, y + 30), fill=(52, 120, 185))
        draw.text((885, y + 7), f"{count:,}", fill=(25, 42, 62))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _tracking(args: argparse.Namespace, safe_root: Path, *, tag: str) -> None:
    receipt = safe_root.parent / "tracking" / "clearml_receipt.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/finalize_full_pool_clearml.py"),
        "--safe-root",
        str(safe_root),
        "--receipt",
        str(receipt),
        "--stage",
        STAGE_RELEASE,
        "--task-name",
        "11R full-pool release",
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
    parser.add_argument("--master-json", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--readiness-audit", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--administrative-root", type=Path, required=True)
    parser.add_argument("--official-result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-new-scenes", type=int, default=100)
    parser.add_argument("--max-working-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--recovery-output-directory", type=Path)
    parser.add_argument("--prune-after-recovery", action="store_true")
    parser.add_argument("--clearml-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--clearml-wall-time", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    source_paths = SourcePaths(
        master_json=args.master_json,
        catalogue=args.catalogue,
        readiness_audit=args.readiness_audit,
        archive_index=args.archive_index,
        archive_zip=args.archive_zip,
        archive_manifest=args.archive_manifest,
        official_split=args.official_split,
    )
    plan, scenes, contract, role_map = prepare_full_pool_release(
        source_paths=source_paths,
        official_result_root=args.official_result_root,
        administrative_root=args.administrative_root,
    )
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    estimated = estimate_full_pool_rgb_bytes(args.archive_index, scenes)
    required = int(estimated * 1.15) + 512 * 1024**2
    free = shutil.disk_usage(output).free
    if required > min(args.max_working_bytes, free) and args.recovery_output_directory is None:
        raise RuntimeError("The full-pool release requires bounded recovery output under this storage budget")
    binding = {
        "stage": STAGE_RELEASE,
        "cohort_id": COHORT_ID,
        "plan_sha256": plan["plan_sha256"],
        "source_membership_sha256": contract.membership_sha256,
        "model_membership_sha256": MODEL_MEMBERSHIP_SHA256,
        "official_evaluation_completion_sha256": plan["official_authorization"]["official_evaluation_completion_sha256"],
    }
    recovery_inputs = (args.recovery_bundle, args.recovery_manifest)
    if any(recovery_inputs) and not all(recovery_inputs):
        raise ValueError("Recovery requires one ZIP and its matching sidecar manifest")
    if all(recovery_inputs):
        restore_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            destination_root=output,
            expected_binding=binding,
            allowed_top_levels=("full-pool-model-input", "trusted-release-state", "release_checkpoint.json"),
        )
    result = build_full_pool_resource(
        archive_zip=args.archive_zip,
        archive_index=args.archive_index,
        scenes=scenes,
        contract=contract,
        role_map=role_map,
        output_root=output,
        plan_sha256=str(plan["plan_sha256"]),
        maximum_new_scenes=args.maximum_new_scenes,
    )
    safe_root = output / "safe"
    role_rows = [
        {"primary_role": role, "scene_count": count, "share": count / SCENE_COUNT}
        for role, count in PRIMARY_ROLE_COUNTS.items()
    ]
    _write_csv(safe_root / "primary_role_coverage.csv", role_rows)
    _render_roles(safe_root / "primary_role_coverage.png")
    run_record = {
        "schema_version": "visual-peer-11r-run/v1",
        "status": "completed" if result["status"] == "complete" else "checkpointed",
        "stage": STAGE_RELEASE,
        "cohort_id": COHORT_ID,
        "target_scene_count": SCENE_COUNT,
        "target_image_count": IMAGE_COUNT,
        "asset_group_count": ASSET_GROUP_COUNT,
        "completed_scene_count": int(result["completed_scene_count"]),
        "remaining_scene_count": int(result["remaining_scene_count"]),
        "estimated_rgb_bytes": estimated,
        "working_bytes_required": required,
        "descriptive_sensitivity_only": True,
        "no_retuning": True,
        "primary_role_labels_preserved": True,
        "model_and_evaluator_physically_separate": True,
        "clearml_aggregate_only": True,
    }
    if result["status"] == "complete":
        receipt = {
            "schema_version": "visual-peer-11r-release-receipt/v1",
            "status": "complete",
            "stage": STAGE_RELEASE,
            "cohort_id": COHORT_ID,
            "scene_count": SCENE_COUNT,
            "image_count": IMAGE_COUNT,
            "asset_group_count": ASSET_GROUP_COUNT,
            "source_membership_sha256": contract.membership_sha256,
            "model_membership_sha256": MODEL_MEMBERSHIP_SHA256,
            "model_package_record_sha256": result["model_package_record_sha256"],
            "evaluator_manifest_sha256": result["evaluator_manifest_sha256"],
            "official_evaluation_completion_sha256": binding["official_evaluation_completion_sha256"],
            "method_record_sha256": plan["official_authorization"]["method_record_sha256"],
            "policy_record_sha256": plan["official_authorization"]["policy_record_sha256"],
            "primary_role_counts": dict(PRIMARY_ROLE_COUNTS),
            "descriptive_sensitivity_only": True,
            "no_retuning": True,
        }
        receipt["receipt_sha256"] = __import__("hashlib").sha256(canonical_json_bytes(receipt)).hexdigest()
        receipt_path = output / "full_pool_release_receipt.json"
        _write_json(receipt_path, receipt)
        publish_frozen_resource(
            resource_root=output / "publish/post-freeze-release",
            stage=STAGE_RELEASE,
            files={"full_pool_release_receipt.json": receipt_path},
            bindings={
                "official_evaluation_completion_sha256": binding["official_evaluation_completion_sha256"],
                "source_membership_sha256": contract.membership_sha256,
                "model_membership_sha256": MODEL_MEMBERSHIP_SHA256,
            },
        )
    _write_json(safe_root / "run_record.json", run_record)
    write_tracking_safe_manifest(safe_root)
    recovery = None
    if args.recovery_output_directory is not None and result["status"] != "complete":
        recovery_directory = args.recovery_output_directory.expanduser().resolve()
        recovery_directory.mkdir(parents=True, exist_ok=True)
        bundle = recovery_directory / "full-pool-release-recovery.zip"
        manifest = recovery_directory / "full-pool-release-recovery-manifest.json"
        recovery = create_recovery_bundle(
            source_root=output,
            bundle_path=bundle,
            manifest_path=manifest,
            binding=binding,
            allowed_top_levels=("full-pool-model-input", "trusted-release-state", "release_checkpoint.json"),
        )
        if args.prune_after_recovery:
            prune_verified_state(
                source_root=output,
                bundle_path=bundle,
                manifest_path=manifest,
                expected_binding=binding,
                allowed_top_levels=("full-pool-model-input", "trusted-release-state", "release_checkpoint.json"),
            )
    _tracking(args, safe_root, tag="final" if result["status"] == "complete" else "recovery")
    print(json.dumps({"release": result, "recovery": recovery}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
