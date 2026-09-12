#!/usr/bin/env python3
"""Run the frozen automatic route on one protected model-facing cohort."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

import run_frozen_development_pipeline as frozen_route

from visual_peer_inspector.data.cohort_contract import (
    CALIBRATION_CONTRACT,
    INTERNAL_LOCK_CONTRACT,
    OFFICIAL_SEEN_TEST_CONTRACT,
    OFFICIAL_UNSEEN_TEST_CONTRACT,
)
from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.locateanything import (
    LocateAnythingWorker,
)
from visual_peer_inspector.models.sam2 import (
    Sam2Configuration,
    Sam2ImageWorker,
)
from visual_peer_inspector.models.visual_encoder import (
    FrozenVisualEncoder,
    specification_by_key,
)
from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
)
from visual_peer_inspector.representation.crops import CropConfiguration
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    prune_verified_state,
    restore_recovery_bundle,
    write_recovery_tracking_root,
)
from visual_peer_inspector.workflows.protected_cohort_inference import (
    STAGES,
    assert_evaluator_absent,
    build_protected_identity,
    load_method_freeze,
    load_policy_freeze,
    publish_protected_bundle,
    read_stage,
    render_prediction_audit,
    run_exact_sharded_stage,
    safe_aggregate_tables,
    verify_frozen_base_route,
    verify_protected_input,
)

CONTRACTS = {
    "calibration": CALIBRATION_CONTRACT,
    "internal-lock": INTERNAL_LOCK_CONTRACT,
    "official-seen": OFFICIAL_SEEN_TEST_CONTRACT,
    "official-unseen": OFFICIAL_UNSEEN_TEST_CONTRACT,
}
RUN_STAGES = {
    "calibration": "protected_calibration_inference",
    "internal-lock": "protected_internal_lock_inference",
    "official-seen": "official_seen_frozen_inference",
    "official-unseen": "official_unseen_frozen_inference",
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _progress(event: str, details: Mapping[str, Any]) -> None:
    print(json.dumps({"event": event, **dict(details)}, sort_keys=True), flush=True)


def _clear_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - GPU cleanup is best effort after a completed stage
        return


def _gpu_snapshot() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "cuda_available": available,
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "device_name": str(torch.cuda.get_device_name(0)) if available else None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if available else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if available else 0,
        }
    except Exception as error:  # noqa: BLE001 - runtime probing must not hide scientific output
        return {
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "runtime_probe_error": type(error).__name__,
        }


def _render_summary_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    width, height = 820, 410
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), "Prediction-only multi-view completion", fill=(25, 42, 62))
    maximum = max((int(row["scene_count"]) for row in rows), default=1)
    for index, row in enumerate(rows):
        y = 90 + index * 95
        ready = int(row["result_available_count"])
        insufficient = int(row["insufficient_evidence_count"])
        bar_width = 620
        ready_width = int(bar_width * ready / maximum)
        insufficient_width = int(bar_width * insufficient / maximum)
        draw.text((30, y + 10), f"{row['view_count']} views", fill=(25, 42, 62))
        draw.rectangle((130, y, 130 + ready_width, y + 28), fill=(52, 120, 185))
        draw.rectangle((130 + ready_width, y, 130 + ready_width + insufficient_width, y + 28), fill=(214, 90, 80))
        draw.text((765, y + 8), f"{ready}/{ready + insufficient}", fill=(25, 42, 62))
    draw.rectangle((30, 365, 50, 385), fill=(52, 120, 185))
    draw.text((58, 367), "result available", fill=(25, 42, 62))
    draw.rectangle((230, 365, 250, 385), fill=(214, 90, 80))
    draw.text((258, 367), "insufficient evidence", fill=(25, 42, 62))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=False, compress_level=9)


def _safe_manifest(root: Path) -> dict[str, Any]:
    run_record_path = root / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"]
    record = {
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
    }
    _write_json(root / "artifact_manifest.json", record)
    return record


def _recovery_binding(
    args: argparse.Namespace,
    *,
    contract: Any,
    identity_sha256: str,
    method_sha256: str,
    policy_sha256: str | None,
) -> dict[str, Any]:
    membership = contract.model_membership_sha256 or contract.membership_sha256
    return {
        "stage": RUN_STAGES[args.cohort],
        "cohort_id": contract.cohort_id,
        "pipeline_identity_sha256": identity_sha256,
        "input_membership_sha256": membership,
        "input_package_record_sha256": file_sha256(
            args.model_input_root / "package_record.json"
        ),
        "input_manifest_sha256": file_sha256(
            args.model_input_root / "neutral_corpus_manifest.jsonl"
        ),
        "method_record_sha256": method_sha256,
        "policy_record_sha256": policy_sha256,
    }


def _publish_recovery(
    args: argparse.Namespace,
    *,
    binding: Mapping[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    if args.recovery_output_directory is None:
        return result
    directory = args.recovery_output_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{args.cohort}-inference-recovery"
    bundle_path = directory / f"{stem}.zip"
    manifest_path = directory / f"{stem}-manifest.json"
    recovery = create_recovery_bundle(
        source_root=args.output_root,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        binding=binding,
        allowed_top_levels=(CONTRACTS[args.cohort].cohort_id,),
    )
    result["recovery"] = {
        "bundle": bundle_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "bundle_sha256": recovery["bundle_sha256"],
        "artifact_count": recovery["artifact_count"],
    }
    stage_rows = result.get("stages", {})
    covered_values = [int(result.get("covered_image_count", 0))]
    covered_values.extend(
        int(values["covered_input_count"])
        for values in stage_rows.values()
        if isinstance(values, Mapping) and "covered_input_count" in values
    )
    covered = max(covered_values)
    write_recovery_tracking_root(
        safe_root=args.output_root / "tracking-safe",
        stage=RUN_STAGES[args.cohort],
        pipeline_status=str(result["status"]),
        metrics={
            "covered_image_count": covered,
            "target_image_count": CONTRACTS[args.cohort].image_count,
            "recovery_artifact_count": recovery["artifact_count"],
            "exact_resume_available": bool(result["exact_resume_available"]),
        },
    )
    if args.prune_after_recovery and result["status"] != "interrupted":
        result["pruned_file_count"] = prune_verified_state(
            source_root=args.output_root,
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            expected_binding=binding,
            allowed_top_levels=(CONTRACTS[args.cohort].cohort_id,),
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = CONTRACTS[args.cohort]
    if args.cohort == "calibration" and args.policy_freeze is not None:
        raise ValueError("Notebook 09A must run before a calibration-derived review policy exists")
    if args.cohort == "internal-lock" and args.policy_freeze is None:
        raise ValueError("Notebook 09C requires the frozen Notebook 09B review policy")
    if args.cohort.startswith("official-") and args.policy_freeze is None:
        raise ValueError("Official inference requires the frozen Notebook 09B review policy")
    method = load_method_freeze(
        args.method_freeze,
        application_contract_path=args.application_contract,
        require_resource=True,
    )
    policy = None
    if args.policy_freeze is not None:
        policy = load_policy_freeze(
            args.policy_freeze,
            method_sha256=method["sha256"],
            require_resource=True,
        )
    verify_protected_input(args.model_input_root, contract)
    permitted_roots = [
        CODE_ROOT,
        args.model_input_root,
        args.method_freeze.parent,
        args.selected_prompt.parent,
        args.sam_checkpoint.parent,
    ]
    if args.policy_freeze is not None:
        permitted_roots.append(args.policy_freeze.parent)
    assert_evaluator_absent(args.mounted_root, permitted_roots=permitted_roots)
    base_identity, locate_configuration, execution_policy = frozen_route.build_identity(args.selected_prompt)
    verify_frozen_base_route(base_identity, method)
    identity = build_protected_identity(
        base_identity,
        contract=contract,
        method_sha256=method["sha256"],
        policy_sha256=None if policy is None else policy["sha256"],
    )
    input_rows = frozen_route._input_records(args.model_input_root, contract)
    recovery_binding = _recovery_binding(
        args,
        contract=contract,
        identity_sha256=identity.sha256,
        method_sha256=method["sha256"],
        policy_sha256=None if policy is None else policy["sha256"],
    )
    recovery_inputs = (args.recovery_bundle, args.recovery_manifest)
    if any(recovery_inputs) and not all(recovery_inputs):
        raise ValueError("Recovery requires both the ZIP and sidecar manifest")
    if all(recovery_inputs):
        restore_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            destination_root=args.output_root,
            expected_binding=recovery_binding,
            allowed_top_levels=(contract.cohort_id,),
        )
    cohort_root = args.output_root.expanduser().resolve() / contract.cohort_id
    identity_path = cohort_root / "pipeline_identity.json"
    identity_record = {
        "schema_version": 1,
        "cohort_id": contract.cohort_id,
        "identity_sha256": identity.sha256,
    }
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity_record:
            raise ValueError("Existing output belongs to another frozen pipeline identity")
    else:
        _write_json(identity_path, identity_record)
    try:
        return _run_prepared(
            args, contract=contract, identity=identity, input_rows=input_rows,
            locate_configuration=locate_configuration, execution_policy=execution_policy,
            method=method, policy=policy, cohort_root=cohort_root,
            recovery_binding=recovery_binding,
        )
    except (Exception, KeyboardInterrupt) as error:
        # Keep valid shards; never prune state while handling an interrupted run.
        interrupted = {
            "status": "interrupted",
            "error_type": type(error).__name__,
            "exact_resume_available": True,
            "covered_image_count": 0,
        }
        try:
            _write_json(cohort_root / "interruption_record.json", interrupted)
            _publish_recovery(args, binding=recovery_binding, result=interrupted)
        except Exception as recovery_error:  # noqa: BLE001 - preserve the original inference error
            _progress("recovery_export_failed", {
                "error_type": type(recovery_error).__name__,
                "loose_shards_retained": True,
            })
        raise


def _run_prepared(
    args: argparse.Namespace, *, contract: Any, identity: Any,
    input_rows: list[dict[str, Any]], locate_configuration: Any, execution_policy: Any,
    method: Mapping[str, Any], policy: Mapping[str, Any] | None,
    cohort_root: Path, recovery_binding: Mapping[str, Any],
) -> dict[str, Any]:
    shard_limit = 1 if args.run_mode == "gate" else None
    maximum_new_shards = args.maximum_new_shards if args.run_mode == "incremental" else None
    wall_started = time.monotonic()
    stages: dict[str, dict[str, Any]] = {}

    locate = LocateAnythingWorker(
        locate_configuration,
        device=args.device,
        model_source=args.locate_model_source,
        execution_policy=execution_policy,
        progress_callback=lambda event, details: _progress(f"locate_{event}", details),
    )
    stages["locateanything"] = run_exact_sharded_stage(
        stage="locateanything",
        input_records=input_rows,
        output_root=cohort_root,
        identity=identity,
        worker=frozen_route._locate_worker(locate, frozen_route.FROZEN_PROPOSAL_POLICY),
        shard_size=args.shard_size,
        shard_limit=shard_limit,
        maximum_new_shards=maximum_new_shards,
        progress_callback=_progress,
    )
    del locate
    _clear_gpu()

    sam = Sam2ImageWorker(Sam2Configuration(device=args.device, precision="float16"), args.sam_checkpoint)
    stages["sam21"] = run_exact_sharded_stage(
        stage="sam21",
        input_records=read_stage(cohort_root / "locateanything"),
        output_root=cohort_root,
        identity=identity,
        worker=frozen_route._sam_worker(sam, identity),
        shard_size=args.shard_size,
        shard_limit=shard_limit,
        maximum_new_shards=maximum_new_shards,
        progress_callback=_progress,
    )
    del sam
    _clear_gpu()

    crop_configuration = CropConfiguration()
    encoder = FrozenVisualEncoder(
        specification_by_key(frozen_route.DINO_MODEL_KEY),
        revision=frozen_route.DINO_REVISION,
        device=args.device,
    )
    stages["dinov2_small"] = run_exact_sharded_stage(
        stage="dinov2_small",
        input_records=read_stage(cohort_root / "sam21"),
        output_root=cohort_root,
        identity=identity,
        worker=frozen_route._dino_worker(encoder, crop_configuration),
        shard_size=args.shard_size,
        shard_limit=shard_limit,
        maximum_new_shards=maximum_new_shards,
        progress_callback=_progress,
    )
    del encoder
    _clear_gpu()

    stages["peer_scoring"] = run_exact_sharded_stage(
        stage="peer_scoring",
        input_records=read_stage(cohort_root / "dinov2_small"),
        output_root=cohort_root,
        identity=identity,
        worker=frozen_route._score_worker,
        shard_size=args.shard_size,
        shard_limit=shard_limit,
        maximum_new_shards=maximum_new_shards,
        progress_callback=_progress,
    )
    runtime = {
        "schema_version": 1,
        "cohort_id": contract.cohort_id,
        "run_mode": args.run_mode,
        "pipeline_duration_seconds": round(time.monotonic() - wall_started, 6),
        "stages": {stage: {key: value for key, value in stages[stage].items() if key in {"duration_seconds", "processed_this_run", "resumed", "completed_shards", "terminal_failures"}} for stage in STAGES},
        **_gpu_snapshot(),
    }

    if args.run_mode == "gate":
        gate = {
            "schema_version": 1,
            "status": "complete",
            "stage": RUN_STAGES[args.cohort],
            "cohort_id": contract.cohort_id,
            "identity_sha256": identity.sha256,
            "method_record_sha256": method["sha256"],
            "policy_record_sha256": None if policy is None else policy["sha256"],
            "covered_image_count": stages["peer_scoring"]["covered_input_count"],
            "shard_size": args.shard_size,
            "exact_resume_available": True,
            "evaluator_opened": False,
            "runtime": runtime,
        }
        _write_json(cohort_root / "gate_record.json", gate)
        return _publish_recovery(
            args,
            binding=recovery_binding,
            result=gate,
        )

    if args.run_mode == "incremental":
        progress = {
            "schema_version": 1,
            "status": "checkpointed",
            "stage": RUN_STAGES[args.cohort],
            "cohort_id": contract.cohort_id,
            "identity_sha256": identity.sha256,
            "method_record_sha256": method["sha256"],
            "policy_record_sha256": None if policy is None else policy["sha256"],
            "covered_image_count": stages["peer_scoring"]["covered_input_count"],
            "stages": runtime["stages"],
            "exact_resume_available": True,
            "evaluator_opened": False,
        }
        _write_json(cohort_root / "incremental_progress.json", progress)
        return _publish_recovery(
            args,
            binding=recovery_binding,
            result=progress,
        )

    if any(stage["covered_input_count"] != contract.image_count for stage in stages.values()):
        raise RuntimeError("Every protected image must reach a terminal stage outcome before publication")
    bundle = publish_protected_bundle(
        cohort_root=cohort_root,
        identity=identity,
        contract=contract,
        method=method,
        policy=policy,
    )
    score_rows = read_stage(cohort_root / "peer_scoring")
    multiview_path = cohort_root / "model-safe/multiview_prediction_manifest.jsonl"
    multiview_rows = [json.loads(line) for line in multiview_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = render_prediction_audit(
        score_rows=score_rows,
        multiview_rows=multiview_rows,
        output_root=cohort_root / "review-only/figures",
    )
    contract_rows, review_rows = safe_aggregate_tables(multiview_rows)
    safe_root = cohort_root / "safe"
    _write_csv(safe_root / "view_contract_summary.csv", contract_rows)
    _write_csv(safe_root / "review_signal_summary.csv", review_rows)
    _render_summary_chart(contract_rows, safe_root / "multiview_completion.png")
    _write_json(safe_root / "runtime_receipt.json", runtime)
    _write_json(
        cohort_root / "review-only" / "qualitative_evidence_inventory.json",
        {"panel_count": len(audit), "items": audit},
    )
    run_record = {
        "schema_version": 1,
        "status": "completed",
        "stage": RUN_STAGES[args.cohort],
        "cohort_id": contract.cohort_id,
        "scene_count": contract.scene_count,
        "image_count": contract.image_count,
        "view_contracts": {str(key): list(value) for key, value in method["view_contracts"].items()},
        "identity_sha256": identity.sha256,
        "method_record_sha256": method["sha256"],
        "policy_record_sha256": None if policy is None else policy["sha256"],
        "prediction_bundle": {key: value for key, value in bundle.items() if key != "lock"},
        "terminal_failures": {stage: stages[stage]["failure_counts"] for stage in STAGES},
        "qualitative_panel_count": len(audit),
        "prediction_only": True,
        "evaluator_opened": False,
        "accuracy_metrics_reported": False,
        "runtime": runtime,
    }
    _write_json(safe_root / "run_record.json", run_record)
    _safe_manifest(safe_root)
    return run_record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(CONTRACTS), required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--application-contract", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path)
    parser.add_argument("--selected-prompt", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mounted-root", type=Path, action="append", default=[])
    parser.add_argument("--locate-model-source", type=Path)
    parser.add_argument("--run-mode", choices=("gate", "incremental", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--maximum-new-shards", type=int)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--recovery-output-directory", type=Path)
    parser.add_argument("--prune-after-recovery", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_arguments())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
