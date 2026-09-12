#!/usr/bin/env python3
"""Build one bounded primary resource gate with cross-version recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    prune_verified_state,
    restore_recovery_bundle,
    write_recovery_tracking_root,
    write_safe_artifact_manifest,
)
from visual_peer_inspector.workflows.primary_resource_release import (
    REFERENCE_GEOMETRY_POLICY,
    ROLE_CONTRACTS,
    SCHEMA_VERSION,
    build_development_identity_evaluator,
    build_protected_role_resources,
    verify_release_source,
)

STAGES = {
    "development-identity": "development_identity_evaluator_release",
    "calibration": "calibration_resource_release",
    "internal-lock": "internal_lock_resource_release",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _top_levels(mode: str) -> tuple[str, ...]:
    if mode == "development-identity":
        return ("development-identity-evaluator", "trusted-release-state")
    short = "calibration" if mode == "calibration" else "internal-lock"
    return (
        f"{short}-model-input",
        f"{short}-evaluator-only",
        "trusted-release-state",
    )


def _binding(args: argparse.Namespace) -> dict[str, object]:
    common: dict[str, object] = {
        "stage": STAGES[args.mode],
        "release_schema_version": SCHEMA_VERSION,
        "administrative_checksums_sha256": file_sha256(
            args.administrative_root / "checksums.sha256"
        ),
        "source_identity": verify_release_source(args.source_root),
        "reference_geometry_policy": REFERENCE_GEOMETRY_POLICY,
    }
    if args.mode == "development-identity":
        common.update(
            {
                "cohort_id": "toysad8k-development-full-1000-v1",
                "prediction_lock_sha256s": sorted(
                    file_sha256(path / "prediction_lock.json")
                    for path in args.prediction_root
                ),
            }
        )
    else:
        contract = ROLE_CONTRACTS[args.mode]
        common.update(
            {
                "cohort_id": contract.cohort_id,
                "membership_sha256": contract.membership_sha256,
                "selection_checksums_sha256": file_sha256(
                    args.selection_root / "checksums.sha256"
                ),
                "label_vault_checksums_sha256": file_sha256(
                    args.label_root / "checksums.sha256"
                ),
                "prerequisite_sha256": file_sha256(args.prerequisite_lock),
                "method_sha256": (
                    None if args.method_lock is None else file_sha256(args.method_lock)
                ),
            }
        )
        method_path = (
            args.prerequisite_lock
            if args.mode == "calibration"
            else args.method_lock
        )
        assert method_path is not None
        common["method_application_contract_sha256"] = file_sha256(
            method_path.parent / "application_output_contract.json"
        )
        common["method_resource_manifest_sha256"] = file_sha256(
            method_path.parent / "artifact_manifest.json"
        )
        if args.mode == "internal-lock":
            common["policy_resource_manifest_sha256"] = file_sha256(
                args.prerequisite_lock.parent / "artifact_manifest.json"
            )
    return common


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("development-identity", "calibration", "internal-lock"),
        required=True,
    )
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--administrative-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prerequisite-lock", type=Path)
    parser.add_argument("--method-lock", type=Path)
    parser.add_argument("--prediction-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("incremental", "full"), default="full")
    parser.add_argument("--maximum-new-scenes", type=int, default=100)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--recovery-output-directory", type=Path)
    parser.add_argument("--prune-after-recovery", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "development-identity" and not args.prediction_root:
        raise ValueError("Development identity release requires prediction roots")
    if args.mode != "development-identity" and any(
        value is None
        for value in (args.selection_root, args.label_root, args.prerequisite_lock)
    ):
        raise ValueError("Protected release requires selection, label, and prerequisite roots")
    if args.mode == "internal-lock" and args.method_lock is None:
        raise ValueError("Internal-lock release requires the frozen method resource")
    args.output_root.mkdir(parents=True, exist_ok=True)
    binding = _binding(args)
    recovery = (args.recovery_bundle, args.recovery_manifest)
    if any(recovery) and not all(recovery):
        raise ValueError("Recovery requires both its ZIP and sidecar manifest")
    if all(recovery):
        restore_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            destination_root=args.output_root,
            expected_binding=binding,
            allowed_top_levels=_top_levels(args.mode),
        )
    try:
        return _build_and_publish(args, binding)
    except (Exception, KeyboardInterrupt) as error:
        # Keep completed scene records even if the release or reporting fails.
        failure = {"status": "interrupted", "error_type": type(error).__name__}
        try:
            _write_json(args.output_root / "resource_run_result.json", failure)
            if args.recovery_output_directory is not None:
                failure["recovery"] = _save_recovery(args, binding)
                _write_json(args.output_root / "resource_run_result.json", failure)
        except Exception as recovery_error:  # noqa: BLE001 - preserve the original release error
            print(json.dumps({
                "event": "recovery_export_failed",
                "error_type": type(recovery_error).__name__,
                "loose_scene_checkpoints_retained": True,
            }), flush=True)
        raise


def _save_recovery(args: argparse.Namespace, binding: dict[str, object]) -> dict[str, object]:
    directory = args.recovery_output_directory
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}-resource-recovery"
    bundle = directory / f"{stem}.zip"
    manifest = directory / f"{stem}-manifest.json"
    record = create_recovery_bundle(
        source_root=args.output_root,
        bundle_path=bundle,
        manifest_path=manifest,
        binding=binding,
        allowed_top_levels=_top_levels(args.mode),
    )
    return {
        "bundle": bundle.as_posix(),
        "manifest": manifest.as_posix(),
        "bundle_sha256": record["bundle_sha256"],
    }


def _build_and_publish(args: argparse.Namespace, binding: dict[str, object]) -> dict[str, object]:
    limit = args.maximum_new_scenes if args.run_mode == "incremental" else None
    if args.mode == "development-identity":
        result = build_development_identity_evaluator(
            prediction_roots=args.prediction_root,
            administrative_root=args.administrative_root,
            source_root=args.source_root,
            output_root=args.output_root,
            maximum_new_scenes=limit,
        )
    else:
        result = build_protected_role_resources(
            role=args.mode,
            selection_root=args.selection_root,
            administrative_root=args.administrative_root,
            label_root=args.label_root,
            source_root=args.source_root,
            prerequisite_lock=args.prerequisite_lock,
            output_root=args.output_root,
            method_lock=args.method_lock,
            maximum_new_scenes=limit,
        )
    if args.recovery_output_directory is not None:
        result["recovery"] = _save_recovery(args, binding)
        write_recovery_tracking_root(
            safe_root=args.output_root / "tracking-safe",
            stage=STAGES[args.mode],
            pipeline_status=str(result["status"]),
            metrics={
                "completed_scene_count": int(result["completed_scene_count"]),
                "remaining_scene_count": int(result["remaining_scene_count"]),
                "exact_resume": bool(result["exact_resume"]),
            },
        )
        if args.prune_after_recovery:
            result["pruned_file_count"] = prune_verified_state(
                source_root=args.output_root,
                bundle_path=Path(result["recovery"]["bundle"]),
                manifest_path=Path(result["recovery"]["manifest"]),
                expected_binding=binding,
                allowed_top_levels=_top_levels(args.mode),
            )
    if result["status"] == "complete":
        safe_root = args.output_root / "safe"
        _write_json(
            safe_root / "run_record.json",
            {
                "schema_version": 1,
                "stage": STAGES[args.mode],
                "status": "completed",
                "cohort_id": result["cohort_id"],
                "completed_scene_count": result["completed_scene_count"],
                "remaining_scene_count": 0,
                "processed_scene_count": result["processed_scene_count"],
                "resumed_scene_count": result["resumed_scene_count"],
                "exact_resume": True,
                "aggregate_only": True,
                "input_signature": result["input_signature"],
                "reference_geometry_policy": REFERENCE_GEOMETRY_POLICY,
                **{
                    name: result[name]
                    for name in (
                        "reference_mapped_observation_count",
                        "reference_unmapped_observation_count",
                        "reference_mapping_fraction",
                    )
                    if name in result
                },
            },
        )
        write_safe_artifact_manifest(safe_root)
    _write_json(args.output_root / "resource_run_result.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parse_arguments()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
