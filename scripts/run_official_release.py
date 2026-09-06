#!/usr/bin/env python3
"""Build the separated official ToysAD-8K evaluation resources."""

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

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
)
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    prune_verified_state,
    restore_recovery_bundle,
    verify_recovery_bundle,
    write_recovery_tracking_root,
)
from visual_peer_inspector.workflows.official_release import (
    EVALUATOR_MANIFEST_SCHEMA,
    OFFICIAL_CONTRACTS,
    SCHEMA_VERSION,
    build_official_release,
    estimate_official_model_bytes,
    verify_internal_lock_decision,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".incomplete").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path.with_suffix(path.suffix + ".incomplete"), path)


def _render_summary(rows: list[dict[str, Any]], path: Path) -> None:
    image = Image.new("RGB", (880, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), "Official evaluation release", fill=(25, 42, 62))
    maximum = max(int(row["image_count"]) for row in rows)
    for index, row in enumerate(rows):
        y = 105 + index * 95
        width = int(650 * int(row["image_count"]) / maximum)
        draw.text((30, y + 7), str(row["cohort"]), fill=(25, 42, 62))
        draw.rectangle((160, y, 160 + width, y + 35), fill=(52, 120, 185))
        draw.text((825, y + 9), f"{int(row['image_count']):,}", fill=(25, 42, 62))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9, optimize=False)


def _safe_manifest(root: Path) -> None:
    run_record_path = root / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    _write_json(
        root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifact_count": len(files),
            "artifacts": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            ],
        },
    )


def _finish_clearml(
    args: argparse.Namespace, safe_root: Path, *, result_tag: str
) -> None:
    receipt = safe_root.parent / "tracking" / "clearml_receipt.json"
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts/finalize_ds2_clearml.py"),
        "--safe-root",
        str(safe_root),
        "--receipt",
        str(receipt),
        "--stage",
        "official_release",
        "--project",
        "Visual Peer Inspector/08 Official Release",
        "--task-name",
        "official-release-v1",
        "--tag",
        "notebook-10r",
        "--tag",
        result_tag,
        "--mode",
        args.clearml_mode,
        "--wall-time-seconds",
        str(args.clearml_wall_time),
    ]
    try:
        subprocess.run(command, check=False, timeout=args.clearml_wall_time + 30)
    except subprocess.TimeoutExpired:
        _write_json(
            receipt,
            {
                "schema_version": 1,
                "stage": "official_release_clearml_finalisation",
                "status": "timed_out",
            },
        )


def _recovery_binding(args: argparse.Namespace) -> dict[str, Any]:
    contract = OFFICIAL_CONTRACTS[args.role]
    return {
        "stage": "10r-official-release",
        "role": args.role,
        "cohort_id": contract.cohort_id,
        "source_membership_sha256": contract.membership_sha256,
        "model_membership_sha256": contract.model_membership_sha256,
        "master_sha256": file_sha256(args.master),
        "archive_index_sha256": file_sha256(
            args.source_root / "archive_index.sqlite3"
        ),
        "package_checksums_sha256": file_sha256(
            args.source_root / "package-checksums.sha256"
        ),
        "verify_full_archive_sha256": not args.skip_full_archive_hash,
        "release_schema_version": SCHEMA_VERSION,
        "evaluator_manifest_schema_version": EVALUATOR_MANIFEST_SCHEMA,
        "lock_result_receipt_sha256": file_sha256(args.lock_result_receipt),
    }


def _recovery_top_levels(role: str) -> tuple[str, ...]:
    return (
        f"{role}-model-input",
        f"{role}-evaluator-only",
        "trusted-release-state",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--role", choices=("seen", "unseen"), required=True)
    parser.add_argument("--lock-result-receipt", type=Path, required=True)
    parser.add_argument("--skip-full-archive-hash", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--run-mode", choices=("incremental", "full"), default="full"
    )
    parser.add_argument("--maximum-new-scenes", type=int, default=100)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--recovery-output-directory", type=Path)
    parser.add_argument("--prune-after-recovery", action="store_true")
    parser.add_argument("--max-working-bytes", type=int, default=20 * 1024**3)
    parser.add_argument(
        "--clearml-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--clearml-wall-time", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    lock_result = verify_internal_lock_decision(args.lock_result_receipt)
    roles = (args.role,)
    args.output_root.mkdir(parents=True, exist_ok=True)
    master = json.loads(args.master.read_text(encoding="utf-8"))
    contract = OFFICIAL_CONTRACTS[args.role]
    binding = _recovery_binding(args)
    recovery_inputs = (args.recovery_bundle, args.recovery_manifest)
    if any(recovery_inputs) and not all(recovery_inputs):
        raise ValueError("Official release recovery requires both ZIP and manifest")
    estimated_media_bytes = estimate_official_model_bytes(
        master=master,
        archive_index=args.source_root / "archive_index.sqlite3",
        contract=contract,
    )
    free_bytes = shutil.disk_usage(args.output_root.parent.resolve()).free
    recovery_bytes = 0
    recovery_record = None
    if all(recovery_inputs):
        recovery_record = verify_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            expected_binding=binding,
            allowed_top_levels=_recovery_top_levels(args.role),
        )
        recovery_bytes = sum(
            int(item["size_bytes"]) for item in recovery_record["artifacts"]
        )
    short = args.role
    state_root = args.output_root / "trusted-release-state" / short
    existing_scenes = len(list(state_root.glob("S*.json"))) if state_root.is_dir() else 0
    if recovery_record is not None:
        state_prefix = f"trusted-release-state/{short}/S"
        existing_scenes = sum(
            str(item["path"]).startswith(state_prefix)
            and str(item["path"]).endswith(".json")
            for item in recovery_record["artifacts"]
        )
    remaining_scenes = max(0, contract.scene_count - existing_scenes)
    planned_scenes = (
        min(remaining_scenes, args.maximum_new_scenes)
        if args.run_mode == "incremental"
        else remaining_scenes
    )
    estimated_new_media = int(
        estimated_media_bytes * planned_scenes / contract.scene_count
    )
    projected_state_bytes = recovery_bytes + int(estimated_new_media * 1.15)
    recovery_archive_reserve = (
        projected_state_bytes if args.run_mode == "incremental" else 0
    )
    required_working_bytes = (
        projected_state_bytes + recovery_archive_reserve + 512 * 1024**2
    )
    if required_working_bytes > min(args.max_working_bytes, free_bytes):
        raise RuntimeError(
            "The selected official split exceeds the safe working-storage budget. "
            "Run the split with a larger verified output volume or publish it through "
            "a private dataset version workflow."
        )
    if all(recovery_inputs):
        restore_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            destination_root=args.output_root,
            expected_binding=binding,
            allowed_top_levels=_recovery_top_levels(args.role),
        )
    receipt = build_official_release(
        source_root=args.source_root,
        master_path=args.master,
        output_root=args.output_root,
        lock_result_receipt=args.lock_result_receipt,
        roles=roles,
        verify_archive_hash=not args.skip_full_archive_hash,
        checkpoint_every=args.checkpoint_every,
        maximum_new_scenes=(
            args.maximum_new_scenes if args.run_mode == "incremental" else None
        ),
    )
    counts = {
        "seen": {
            "cohort": "seen-test",
            "scene_count": 715,
            "asset_group_count": 113,
            "image_count": 14_300,
        },
        "unseen": {
            "cohort": "unseen-test",
            "scene_count": 2_437,
            "asset_group_count": 316,
            "image_count": 48_740,
        },
    }
    release_status = str(receipt["status"])
    resource = receipt["resources"][0]
    completed_scene_count = int(
        resource.get("completed_scene_count", contract.scene_count)
    )
    rows = [{
        "cohort": counts[args.role]["cohort"],
        "scene_count": completed_scene_count,
        "asset_group_count": counts[args.role]["asset_group_count"],
        "image_count": completed_scene_count * contract.views_per_scene,
        "target_scene_count": contract.scene_count,
        "target_image_count": contract.image_count,
        "estimated_media_bytes": estimated_media_bytes,
        "completed_scene_count": completed_scene_count,
        "status": release_status,
    }]
    safe_root = args.output_root.resolve() / "safe"
    _write_csv(safe_root / "official_cohort_summary.csv", rows)
    _render_summary(rows, safe_root / "official_cohort_summary.png")
    run_record = {
        "schema_version": 1,
        "status": "completed" if release_status == "complete" else "checkpointed",
        "stage": "official_release",
        "cohort_count": 1,
        **counts[args.role],
        "estimated_media_bytes": estimated_media_bytes,
        "required_working_bytes": required_working_bytes,
        "available_working_bytes_at_start": free_bytes,
        "recovery_bytes_restored": recovery_bytes,
        "run_mode": args.run_mode,
        "maximum_new_scenes": args.maximum_new_scenes,
        "completed_scene_count": completed_scene_count,
        "physical_resource_count": 2,
        "model_resource_count": 1,
        "evaluator_resource_count": 1,
        "evaluator_is_model_input": False,
        "lock_result_receipt_sha256": lock_result["sha256"],
        "release_receipt_sha256": file_sha256(
            safe_root / "official_release_receipt.json"
        ),
    }
    _write_json(safe_root / "run_record.json", run_record)
    _safe_manifest(safe_root)
    tracking_root = safe_root
    if release_status != "complete":
        if args.recovery_output_directory is None:
            raise ValueError("Incremental official release requires a recovery output directory")
        recovery_directory = args.recovery_output_directory.expanduser().resolve()
        recovery_directory.mkdir(parents=True, exist_ok=True)
        bundle_path = recovery_directory / f"official-{args.role}-release-recovery.zip"
        manifest_path = recovery_directory / f"official-{args.role}-release-recovery-manifest.json"
        recovery = create_recovery_bundle(
            source_root=args.output_root,
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            binding=binding,
            allowed_top_levels=_recovery_top_levels(args.role),
        )
        run_record["recovery"] = {
            "bundle": bundle_path.as_posix(),
            "manifest": manifest_path.as_posix(),
            "bundle_sha256": recovery["bundle_sha256"],
            "artifact_count": recovery["artifact_count"],
        }
        if args.prune_after_recovery:
            run_record["pruned_file_count"] = prune_verified_state(
                source_root=args.output_root,
                bundle_path=bundle_path,
                manifest_path=manifest_path,
                expected_binding=binding,
                allowed_top_levels=_recovery_top_levels(args.role),
            )
        _write_json(safe_root / "run_record.json", run_record)
        _safe_manifest(safe_root)
        tracking_root = args.output_root / "tracking-safe"
        write_recovery_tracking_root(
            safe_root=tracking_root,
            stage="official_release",
            pipeline_status="checkpointed",
            metrics={
                "completed_scene_count": completed_scene_count,
                "target_scene_count": contract.scene_count,
                "completed_image_count": completed_scene_count
                * contract.views_per_scene,
                "target_image_count": contract.image_count,
                "recovery_artifact_count": recovery["artifact_count"],
            },
        )
    _finish_clearml(
        args,
        tracking_root,
        result_tag="final" if release_status == "complete" else "recovery",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
