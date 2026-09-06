#!/usr/bin/env python3
"""Run and lock the frozen route on official seen and unseen RGB resources."""

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
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

from PIL import Image, ImageDraw

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

import run_protected_cohort_inference as protected_runner

from visual_peer_inspector.data.cohort_contract import (
    OFFICIAL_SEEN_TEST_CONTRACT,
    OFFICIAL_UNSEEN_TEST_CONTRACT,
)
from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.frozen_resource import publish_frozen_resource
from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
)
from visual_peer_inspector.workflows.official_inference import (
    build_official_inference_summary,
    create_recovery_bundle,
    load_lock_result_receipt,
    restore_recovery_bundle,
)
from visual_peer_inspector.workflows.protected_cohort_inference import (
    assert_evaluator_absent,
    load_method_freeze,
    load_policy_freeze,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _render_summary(rows: list[dict[str, Any]], path: Path) -> None:
    image = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), "Official prediction locks", fill=(25, 42, 62))
    maximum = max(int(row["image_count"]) for row in rows)
    for index, row in enumerate(rows):
        y = 105 + index * 95
        width = int(650 * int(row["image_count"]) / maximum)
        draw.text((30, y + 7), str(row["cohort"]), fill=(25, 42, 62))
        draw.rectangle((170, y, 170 + width, y + 35), fill=(52, 120, 185))
        draw.text((835, y + 9), f"{int(row['image_count']):,}", fill=(25, 42, 62))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9, optimize=False)


def _safe_manifest(root: Path) -> None:
    run_record_path = root / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"]
    _write_json(root / "artifact_manifest.json", {
        "schema_version": 1,
        "artifact_count": len(files),
        "artifacts": [
            {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in files
        ],
    })


def _finish_clearml(
    args: argparse.Namespace, safe_root: Path, *, stage: str, task_name: str, tag: str
) -> None:
    receipt = safe_root.parent / "tracking" / "clearml_receipt.json"
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts/finalize_ds2_clearml.py"),
        "--safe-root", str(safe_root),
        "--receipt", str(receipt),
        "--stage", stage,
        "--project", "Visual Peer Inspector/09 Frozen Evaluation",
        "--task-name", task_name,
        "--tag", "notebook-10a",
        "--tag", tag,
        "--mode", args.clearml_mode,
        "--wall-time-seconds", str(args.clearml_wall_time),
    ]
    try:
        subprocess.run(command, check=False, timeout=args.clearml_wall_time + 30)
    except subprocess.TimeoutExpired:
        _write_json(receipt, {"schema_version": 1, "stage": "official_frozen_inference_clearml_finalisation", "status": "timed_out"})


def _verify_prerequisites(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    result = load_lock_result_receipt(
        args.lock_result_receipt,
        method_sha256=method["sha256"],
        policy_sha256=policy["sha256"],
        require_resource=True,
    )
    return method, policy, result


def _run_role(args: argparse.Namespace, role: str) -> dict[str, Any]:
    model_root = args.seen_model_input if role == "seen" else args.unseen_model_input
    if model_root is None:
        raise ValueError(f"The {role} model-input root is required")
    if args.selected_prompt is None or args.sam_checkpoint is None:
        raise ValueError("Model phases require the prompt lock and SAM checkpoint")
    assert_evaluator_absent(
        args.mounted_root,
        permitted_roots=(
            CODE_ROOT,
            model_root,
            args.method_freeze.parent,
            args.policy_freeze.parent,
            args.lock_result_receipt.parent,
            args.selected_prompt.parent,
            args.sam_checkpoint.parent,
        ),
    )
    namespace = SimpleNamespace(
        cohort=f"official-{role}",
        model_input_root=model_root,
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
        recovery_bundle=None,
        recovery_manifest=None,
        recovery_output_directory=None,
        prune_after_recovery=False,
    )
    return protected_runner.run(namespace)


def _finalize(args: argparse.Namespace, method: Mapping[str, Any], policy: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    root = args.output_root.resolve()
    seen_lock = args.seen_prediction_lock or root / OFFICIAL_SEEN_TEST_CONTRACT.cohort_id / "model-safe/prediction_lock.json"
    unseen_lock = args.unseen_prediction_lock or root / OFFICIAL_UNSEEN_TEST_CONTRACT.cohort_id / "model-safe/prediction_lock.json"
    assert_evaluator_absent(
        args.mounted_root,
        permitted_roots=(
            CODE_ROOT,
            args.method_freeze.parent,
            args.policy_freeze.parent,
            args.lock_result_receipt.parent,
            seen_lock.parent.parent,
            unseen_lock.parent.parent,
        ),
    )
    summary = build_official_inference_summary(
        seen_lock_path=seen_lock,
        unseen_lock_path=unseen_lock,
        method_sha256=str(method["sha256"]),
        policy_sha256=str(policy["sha256"]),
        lock_result_receipt_sha256=str(result["sha256"]),
    )
    safe_root = root / "official-safe"
    rows = [
        {"cohort": "seen-test", "scene_count": 715, "image_count": 14_300, "prediction_lock_sha256": summary["cohorts"][0]["prediction_lock_sha256"]},
        {"cohort": "unseen-test", "scene_count": 2_437, "image_count": 48_740, "prediction_lock_sha256": summary["cohorts"][1]["prediction_lock_sha256"]},
    ]
    _write_json(safe_root / "official_inference_summary.json", summary)
    _write_csv(safe_root / "official_prediction_lock_summary.csv", rows)
    _render_summary(rows, safe_root / "official_prediction_lock_summary.png")
    _write_json(safe_root / "run_record.json", {
        "schema_version": 1,
        "status": "completed",
        "stage": "official_frozen_inference",
        "cohort_count": 2,
        "scene_count": 3_152,
        "image_count": 63_040,
        "prediction_only": True,
        "evaluator_opened": False,
        "accuracy_metrics_reported": False,
        "official_summary_sha256": file_sha256(safe_root / "official_inference_summary.json"),
    })
    _safe_manifest(safe_root)
    summary_resource = publish_frozen_resource(
        resource_root=root / "publish" / "official-prediction-summary",
        stage="official_prediction_summary_release",
        files={
            "official_inference_summary.json": safe_root
            / "official_inference_summary.json"
        },
        bindings={
            "method_record_sha256": str(method["sha256"]),
            "policy_record_sha256": str(policy["sha256"]),
            "lock_result_receipt_sha256": str(result["sha256"]),
            "seen_prediction_lock_sha256": summary["cohorts"][0][
                "prediction_lock_sha256"
            ],
            "unseen_prediction_lock_sha256": summary["cohorts"][1][
                "prediction_lock_sha256"
            ],
        },
    )
    _finish_clearml(
        args,
        safe_root,
        stage="official_frozen_inference",
        task_name="official-frozen-inference-v1",
        tag="final",
    )
    return {"summary": summary, "published_resource": summary_resource}


def _publish_prediction_resource(
    args: argparse.Namespace,
    role: str,
    run_result: Mapping[str, Any],
    *,
    method: Mapping[str, Any],
    policy: Mapping[str, Any],
    lock_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Publish the exact model-only files required by the later evaluator."""

    contract = (
        OFFICIAL_SEEN_TEST_CONTRACT
        if role == "seen"
        else OFFICIAL_UNSEEN_TEST_CONTRACT
    )
    cohort_root = args.output_root.resolve() / contract.cohort_id
    model_safe = cohort_root / "model-safe"
    run_record = cohort_root / "safe" / "run_record.json"
    prediction_lock = model_safe / "prediction_lock.json"
    files = {
        f"model-safe/{path.relative_to(model_safe).as_posix()}": path
        for path in sorted(model_safe.rglob("*"))
        if path.is_file()
    }
    files["safe/run_record.json"] = run_record
    resource = publish_frozen_resource(
        resource_root=args.output_root.resolve()
        / "publish"
        / f"official-{role}-prediction",
        stage=f"official_{role}_prediction_release",
        files=files,
        bindings={
            "cohort_id": contract.cohort_id,
            "method_record_sha256": str(method["sha256"]),
            "policy_record_sha256": str(policy["sha256"]),
            "lock_result_receipt_sha256": str(lock_result["sha256"]),
            "prediction_lock_sha256": file_sha256(prediction_lock),
            "identity_sha256": str(run_result["identity_sha256"]),
        },
    )
    return resource


def _publish_recovery(
    args: argparse.Namespace, role: str, run_result: Mapping[str, Any]
) -> dict[str, Any]:
    contract = OFFICIAL_SEEN_TEST_CONTRACT if role == "seen" else OFFICIAL_UNSEEN_TEST_CONTRACT
    cohort_root = args.output_root.resolve() / contract.cohort_id
    bundle, manifest = create_recovery_bundle(
        cohort_root=cohort_root,
        output_directory=args.output_root.resolve() / "recovery",
        cohort_id=contract.cohort_id,
    )
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    safe_root = args.output_root.resolve() / "recovery-safe"
    _write_json(safe_root / "recovery_run_summary.json", {
        "schema_version": 1,
        "cohort_id": contract.cohort_id,
        "run_mode": args.run_mode,
        "bundle_sha256": manifest_value["bundle_sha256"],
        "bundle_size_bytes": manifest_value["bundle_size_bytes"],
        "artifact_count": manifest_value["artifact_count"],
        "identity_sha256": manifest_value["identity_sha256"],
        "exact_resume": True,
        "final": False,
    })
    _write_json(safe_root / "run_record.json", {
        "schema_version": 1,
        "status": "completed",
        "stage": "official_inference_recovery",
        "cohort_count": 1,
        "run_mode": args.run_mode,
        "exact_resume": True,
        "final": False,
        "stage_count": len(run_result.get("stages", {})),
    })
    _safe_manifest(safe_root)
    _finish_clearml(
        args,
        safe_root,
        stage="official_inference_recovery",
        task_name=f"official-{role}-recovery",
        tag="recovery",
    )
    if args.prune_after_recovery:
        with ZipFile(bundle) as archive:
            if archive.testzip() is not None:
                raise ValueError("The recovery ZIP failed its CRC check")
        shutil.rmtree(cohort_root)
    return {
        "bundle": str(bundle),
        "manifest": str(manifest),
        "bundle_sha256": manifest_value["bundle_sha256"],
        "working_state_pruned": bool(args.prune_after_recovery),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--run-mode", choices=("gate", "incremental", "full", "finalize"), required=True)
    parser.add_argument("--seen-model-input", type=Path)
    parser.add_argument("--unseen-model-input", type=Path)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--application-contract", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--lock-result-receipt", type=Path, required=True)
    parser.add_argument("--selected-prompt", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mounted-root", type=Path, action="append", default=[])
    parser.add_argument("--locate-model-source", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--maximum-new-shards", type=int, default=10)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--seen-prediction-lock", type=Path)
    parser.add_argument("--unseen-prediction-lock", type=Path)
    parser.add_argument("--export-recovery", action="store_true")
    parser.add_argument("--prune-after-recovery", action="store_true")
    parser.add_argument("--clearml-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--clearml-wall-time", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    method, policy, result = _verify_prerequisites(args)
    contract = OFFICIAL_SEEN_TEST_CONTRACT if args.cohort == "seen" else OFFICIAL_UNSEEN_TEST_CONTRACT
    if (args.recovery_bundle is None) != (args.recovery_manifest is None):
        raise ValueError("A recovery ZIP and its manifest must be supplied together")
    if args.recovery_bundle is not None:
        restore_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            output_root=args.output_root,
            cohort_id=contract.cohort_id,
        )
    if args.run_mode == "finalize":
        output = _finalize(args, method, policy, result)
    else:
        output = {args.cohort: _run_role(args, args.cohort)}
        if args.export_recovery:
            if args.run_mode not in {"gate", "incremental"}:
                raise ValueError("Recovery export is for gate or incremental runs")
            output["recovery"] = _publish_recovery(
                args, args.cohort, output[args.cohort]
            )
        elif args.run_mode == "full":
            contract = OFFICIAL_SEEN_TEST_CONTRACT if args.cohort == "seen" else OFFICIAL_UNSEEN_TEST_CONTRACT
            output["published_resource"] = _publish_prediction_resource(
                args,
                args.cohort,
                output[args.cohort],
                method=method,
                policy=policy,
                lock_result=result,
            )
            _finish_clearml(
                args,
                args.output_root.resolve() / contract.cohort_id / "safe",
                stage=f"official_{args.cohort}_frozen_inference",
                task_name=f"official-{args.cohort}-frozen-inference",
                tag="final",
            )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
