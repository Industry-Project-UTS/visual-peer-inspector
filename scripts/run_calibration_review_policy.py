#!/usr/bin/env python3
"""Evaluate calibration predictions and freeze the review policy for later cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.frozen_resource import (
    publish_frozen_resource,
    verify_frozen_resource,
)
from visual_peer_inspector.notebook_provenance import (
    PROVENANCE_FIELDS,
    runtime_code_provenance,
)
from visual_peer_inspector.tracking_safe_artifacts import (
    validate_tracking_safe_root,
)
from visual_peer_inspector.workflows.calibration_review_policy import (
    artifact_manifest,
    build_signal_records,
    choose_policy,
    cross_validated_calibration_curve,
    exact_resume_receipt_valid,
    freeze_review_policy,
    grouped_cross_validation,
    open_calibration_evaluator,
    policy_grid,
    primary_metric_table,
    reason_burden_table,
    render_evaluator_examples,
    state_summary_table,
    subgroup_metric_table,
    verify_calibration_boundary,
    write_csv,
    write_exact_resume_receipt,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")
    temporary.replace(path)


def _write_csv_allow_empty(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a fixed-schema CSV, including a header when there are no rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_risk_coverage(
    frontier_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    efficient = [row for row in frontier_rows if bool(row["pareto_efficient"])]
    selected = next(row for row in frontier_rows if bool(row["selected"]))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.scatter(
        [float(row["automatic_coverage"]) for row in frontier_rows],
        [
            1.0 if row["false_ready_rate"] is None else float(row["false_ready_rate"])
            for row in frontier_rows
        ],
        s=18,
        color="#9aa7b4",
        alpha=0.45,
        label="candidate policies",
    )
    axis.plot(
        [float(row["automatic_coverage"]) for row in efficient],
        [
            1.0 if row["false_ready_rate"] is None else float(row["false_ready_rate"])
            for row in efficient
        ],
        marker="o",
        color="#24557a",
        linewidth=1.7,
        label="risk-coverage frontier",
    )
    axis.scatter(
        [float(selected["automatic_coverage"])],
        [
            1.0
            if selected["false_ready_rate"] is None
            else float(selected["false_ready_rate"])
        ],
        marker="*",
        s=180,
        color="#d94841",
        label="frozen policy",
        zorder=5,
    )
    axis.set_xlabel("Automatic coverage")
    axis.set_ylabel("False-ready rate")
    axis.set_title("Calibration risk versus automatic coverage")
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_reliability(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.plot(
        [0, 1], [0, 1], linestyle="--", color="#7c8793", label="perfect calibration"
    )
    if rows:
        x = [float(row["mean_calibrated_error_probability"]) for row in rows]
        y = [float(row["observed_wrong_result_rate"]) for row in rows]
        low = [float(row["observed_rate_ci_low"]) for row in rows]
        high = [float(row["observed_rate_ci_high"]) for row in rows]
        axis.errorbar(
            x,
            y,
            yerr=(
                [value - lower for value, lower in zip(y, low)],
                [upper - value for value, upper in zip(y, high)],
            ),
            marker="o",
            color="#24557a",
            capsize=3,
            label="grouped held-out folds",
        )
    axis.set_xlabel("Calibrated probability of a wrong fused result")
    axis.set_ylabel("Observed wrong-result rate")
    axis.set_title("Held-out reliability curve")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, output_path)


def _plot_state_summary(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    counts = sorted({int(row["view_count"]) for row in rows})
    states = ("result_ready", "review_required", "insufficient_evidence")
    colours = ("#4c956c", "#e7a83e", "#c75146")
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    bottoms = [0.0] * len(counts)
    for state, colour in zip(states, colours, strict=True):
        values = [
            float(
                next(
                    row["share"]
                    for row in rows
                    if int(row["view_count"]) == count and row["policy_state"] == state
                )
            )
            for count in counts
        ]
        axis.bar(
            [str(value) for value in counts],
            values,
            bottom=bottoms,
            label=state.replace("_", " "),
            color=colour,
        )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]
    axis.set_xlabel("Uploaded views")
    axis.set_ylabel("Share of scene contracts")
    axis.set_title("Frozen review-policy states")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, output_path)


def _plot_reason_burden(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    ordered = sorted(
        rows, key=lambda row: float(row["share_of_all_contracts"]), reverse=True
    )
    figure, axis = plt.subplots(figsize=(8.0, max(4.0, 0.42 * len(ordered))))
    axis.barh(
        [str(row["reason"]).replace("_", " ") for row in reversed(ordered)],
        [float(row["share_of_all_contracts"]) for row in reversed(ordered)],
        color="#4b779b",
    )
    axis.set_xlabel("Share of scene contracts")
    axis.set_title("Reason-specific review burden")
    axis.set_xlim(0.0, 1.0)
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, output_path)


def _input_hashes(
    args: argparse.Namespace, boundary: Any, evaluator_sha256: str
) -> dict[str, str]:
    return {
        "prediction_lock_sha256": boundary.lock_sha256,
        "run_record_sha256": boundary.run_record_sha256,
        "method_record_sha256": boundary.method_record_sha256,
        "evaluator_sha256": evaluator_sha256,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "review_cost": f"{args.review_cost:.8f}",
        "insufficient_cost": f"{args.insufficient_cost:.8f}",
        "fold_count": str(args.fold_count),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run the protected calibration evaluator and write a frozen policy record."""

    started = time.monotonic()
    verify_frozen_resource(
        args.method_record.expanduser().resolve(strict=True).parent,
        expected_stage="cross_view_identity_method_release",
        expected_files={
            "application_output_contract.json",
            "method_freeze.json",
        },
    )
    boundary = verify_calibration_boundary(
        prediction_lock_path=args.prediction_lock,
        run_record_path=args.run_record,
        method_record_path=args.method_record,
    )
    evaluator_rows, access_receipt = open_calibration_evaluator(
        boundary, args.evaluator_correspondence
    )
    input_hashes = _input_hashes(
        args, boundary, str(access_receipt["evaluator_sha256"])
    )
    run_id = canonical_run_id(input_hashes)
    run_root = args.output_root.expanduser().resolve() / run_id
    safe_root = run_root / "safe"
    evaluator_root = run_root / "evaluator-only"
    downstream_root = run_root / "downstream"
    receipt_path = run_root / "evaluation_receipt.json"
    if exact_resume_receipt_valid(receipt_path, input_hashes):
        record = json.loads((safe_root / "run_record.json").read_text(encoding="utf-8"))
        return {**record, "resumed": True, "run_root": run_root.as_posix()}
    policy_resource_root = run_root / "publish" / "review-policy"
    if policy_resource_root.exists():
        shutil.rmtree(policy_resource_root)

    _write_json(evaluator_root / "evaluator_access_receipt.json", access_receipt)
    signal_rows = build_signal_records(boundary, evaluator_rows)
    cv_folds, cv_rows = grouped_cross_validation(
        signal_rows,
        fold_count=args.fold_count,
        review_cost=args.review_cost,
        insufficient_cost=args.insufficient_cost,
    )
    calibrated_rows, reliability_rows = cross_validated_calibration_curve(cv_rows)
    final_policy, frontier_rows = choose_policy(
        signal_rows,
        policy_grid(signal_rows),
        review_cost=args.review_cost,
        insufficient_cost=args.insufficient_cost,
    )
    final_rows = []
    by_key = {
        (str(row["opaque_scene_id"]), int(row["view_count"])): row
        for row in calibrated_rows
    }
    from visual_peer_inspector.workflows.calibration_review_policy import (
        apply_review_policy,
    )

    for row in signal_rows:
        state, reasons = apply_review_policy(row, final_policy)
        calibrated = by_key[(str(row["opaque_scene_id"]), int(row["view_count"]))]
        final_rows.append(
            {
                **dict(row),
                "cv_fold": int(calibrated["cv_fold"]),
                "review_risk_score": float(calibrated["review_risk_score"]),
                "calibrated_error_probability": float(
                    calibrated["calibrated_error_probability"]
                ),
                "policy_state": state,
                "review_reasons": list(reasons),
            }
        )

    primary_rows = primary_metric_table(final_rows)
    reason_rows = reason_burden_table(final_rows)
    subgroup_rows = subgroup_metric_table(final_rows)
    state_rows = state_summary_table(final_rows)
    policy_record = freeze_review_policy(
        boundary=boundary,
        policy=final_policy,
        cv_rows=cv_rows,
        evaluator_access_receipt=access_receipt,
        output_path=downstream_root / "review_policy_freeze.json",
        review_cost=args.review_cost,
        insufficient_cost=args.insufficient_cost,
    )

    write_csv(safe_root / "cross_validation_folds.csv", cv_folds)
    write_csv(safe_root / "policy_frontier.csv", frontier_rows)
    write_csv(safe_root / "apparent_fit_metrics.csv", primary_rows)
    write_csv(safe_root / "reason_specific_review_burden.csv", reason_rows)
    write_csv(safe_root / "subgroup_metrics.csv", subgroup_rows)
    write_csv(safe_root / "state_summary.csv", state_rows)
    write_csv(
        safe_root / "reliability_curve.csv",
        reliability_rows
        or [
            {
                "probability_bin": "none",
                "lower_bound": None,
                "upper_bound": None,
                "scene_contract_count": 0,
                "mean_calibrated_error_probability": None,
                "observed_wrong_result_rate": None,
                "observed_rate_ci_low": None,
                "observed_rate_ci_high": None,
            }
        ],
    )
    _write_jsonl(evaluator_root / "evaluated_scene_contracts.jsonl", final_rows)
    wrong_ready_rows = sorted(
        (
            row
            for row in final_rows
            if row["policy_state"] == "result_ready"
            and row.get("is_correct") is False
        ),
        key=lambda row: (str(row["opaque_scene_id"]), int(row["view_count"])),
    )
    _write_jsonl(
        evaluator_root / "wrong_result_ready_scene_contracts.jsonl",
        wrong_ready_rows,
    )
    wrong_ready_fields = (
        "opaque_scene_id",
        "view_count",
        "opaque_asset_group_id",
        "category",
        "anomaly_type",
        "fused_odd_one_out",
        "acceptable_anomalous_identity_ids",
        "policy_state",
        "is_correct",
        "margin",
        "agreement",
        "minimum_mask_quality",
        "review_risk_score",
        "calibrated_error_probability",
    )
    _write_csv_allow_empty(
        evaluator_root / "wrong_result_ready_scene_contracts.csv",
        fieldnames=wrong_ready_fields,
        rows=[
            {
                field: (
                    json.dumps(row[field], separators=(",", ":"))
                    if field == "acceptable_anomalous_identity_ids"
                    else row.get(field)
                )
                for field in wrong_ready_fields
            }
            for row in wrong_ready_rows
        ],
    )
    example_inventory = render_evaluator_examples(
        rows=final_rows,
        boundary=boundary,
        model_input_root=args.model_input_root,
        output_root=evaluator_root / "figures",
        count=args.example_count,
    )
    _write_json(
        evaluator_root / "example_inventory.json",
        {
            "schema_version": 1,
            "selection_is_evaluator_informed": True,
            "publication_location": "evaluator-only",
            "example_count_limit": args.example_count,
            "rendered_example_count": len(example_inventory),
            "total_wrong_result_ready_count": len(wrong_ready_rows),
            "rendered_wrong_result_ready_count": sum(
                bool(item["wrong_result_ready"]) for item in example_inventory
            ),
            "complete_wrong_result_ready_jsonl": (
                "wrong_result_ready_scene_contracts.jsonl"
            ),
            "complete_wrong_result_ready_table": (
                "wrong_result_ready_scene_contracts.csv"
            ),
            "items": example_inventory,
        },
    )

    _plot_risk_coverage(frontier_rows, safe_root / "risk_coverage_frontier.png")
    _plot_reliability(reliability_rows, safe_root / "held_out_reliability_curve.png")
    _plot_state_summary(state_rows, safe_root / "policy_state_by_view_count.png")
    _plot_reason_burden(reason_rows, safe_root / "reason_specific_review_burden.png")
    run_record = {
        "schema_version": 1,
        "status": "completed",
        "stage": "calibration_review_policy_lock",
        "run_id": run_id,
        "cohort_id": "toysad8k-calibration-600-v1",
        "scene_count": 600,
        "scene_contract_count": len(final_rows),
        "asset_group_count": len(
            {str(row["opaque_asset_group_id"]) for row in final_rows}
        ),
        "prediction_lock_verified_before_evaluator": True,
        "evaluator_opened": True,
        "method_record_sha256": boundary.method_record_sha256,
        "chosen_configuration_sha256": boundary.chosen_configuration_sha256,
        "prediction_lock_sha256": boundary.lock_sha256,
        "review_policy_sha256": file_sha256(
            downstream_root / "review_policy_freeze.json"
        ),
        "policy_configuration_sha256": policy_record["policy_configuration_sha256"],
        "model_or_method_setting_changed": False,
        "only_review_policy_calibrated": True,
        "raw_p1_is_probability": False,
        "all_normal_scene_claim_supported": False,
        "application_claim": "relative unusualness among the uploaded peer objects",
        "output_examples_are_evaluator_only": True,
        "evaluator_example_count_limit": args.example_count,
        "rendered_evaluator_example_count": len(example_inventory),
        "total_wrong_result_ready_count": len(wrong_ready_rows),
        "runtime_seconds": round(time.monotonic() - started, 6),
        "resumed": False,
        **runtime_code_provenance(required=False),
    }
    _write_json(safe_root / "run_record.json", run_record)
    safe_manifest = artifact_manifest(
        safe_root, excluded_names={"artifact_manifest.json"}
    )
    _write_json(safe_root / "artifact_manifest.json", safe_manifest)
    evaluator_manifest = artifact_manifest(
        evaluator_root, excluded_names={"artifact_manifest.json"}
    )
    _write_json(evaluator_root / "artifact_manifest.json", evaluator_manifest)

    publish_frozen_resource(
        resource_root=policy_resource_root,
        stage="calibration_review_policy_release",
        files={
            "review_policy_freeze.json": downstream_root
            / "review_policy_freeze.json"
        },
        bindings={
            "method_record_sha256": boundary.method_record_sha256,
            "review_policy_sha256": file_sha256(
                downstream_root / "review_policy_freeze.json"
            ),
            "prediction_lock_sha256": boundary.lock_sha256,
        },
    )
    publication_receipt = {
        "schema_version": 1,
        "status": "completed",
        "stage": "calibration_review_policy_release",
        "run_record_sha256": file_sha256(safe_root / "run_record.json"),
        "resource_manifest_sha256": file_sha256(
            policy_resource_root / "artifact_manifest.json"
        ),
        **runtime_code_provenance(required=False),
    }
    _write_json(safe_root / "publication_receipt.json", publication_receipt)
    safe_manifest = artifact_manifest(
        safe_root, excluded_names={"artifact_manifest.json"}
    )
    _write_json(safe_root / "artifact_manifest.json", safe_manifest)

    outputs = [
        *[path for path in safe_root.rglob("*") if path.is_file()],
        *[path for path in evaluator_root.rglob("*") if path.is_file()],
        downstream_root / "review_policy_freeze.json",
        policy_resource_root / "artifact_manifest.json",
    ]
    write_exact_resume_receipt(
        receipt_path, input_hashes=input_hashes, output_paths=outputs
    )
    return {**run_record, "run_root": run_root.as_posix()}


def canonical_run_id(input_hashes: Mapping[str, str]) -> str:
    digest = (
        __import__("hashlib")
        .sha256(canonical_json_bytes(dict(input_hashes)))
        .hexdigest()
    )
    return f"calibration-policy-{digest[:16]}"


def finalise_clearml(run_root: Path) -> dict[str, Any]:
    """Upload only aggregate evidence and the policy file from a bounded process."""

    root = Path(run_root).expanduser().resolve(strict=True)
    safe_root = root / "safe"
    policy_path = root / "downstream/review_policy_freeze.json"
    receipt_path = root / "clearml_receipt.json"
    try:
        if os.getenv("VPI_TRACKING_MODE", "disabled").lower() != "online":
            raise RuntimeError("tracking_disabled")
        run_record, _manifest = validate_tracking_safe_root(
            safe_root,
            expected_stage="calibration_review_policy_lock",
            allowed_relative_paths={
                "cross_validation_folds.csv",
                "held_out_reliability_curve.png",
                "policy_frontier.csv",
                "policy_state_by_view_count.png",
                "apparent_fit_metrics.csv",
                "reason_specific_review_burden.csv",
                "reason_specific_review_burden.png",
                "reliability_curve.csv",
                "risk_coverage_frontier.png",
                "state_summary.csv",
                "subgroup_metrics.csv",
            },
        )
        if file_sha256(policy_path) != run_record.get("review_policy_sha256"):
            raise ValueError("Review policy differs from the completed run record")
        from clearml import Task

        task = Task.init(
            project_name="Visual Peer Inspector/06 Protected Evaluation",
            task_name="calibration-review-policy-lock",
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
            auto_connect_streams=False,
            auto_resource_monitoring=False,
            output_uri=False,
        )
        provenance = {
            name: run_record[name]
            for name in PROVENANCE_FIELDS
            if name in run_record
        }
        if provenance:
            task.set_user_properties(**provenance)
        logger = task.get_logger()
        for table_name in (
            "apparent_fit_metrics",
            "cross_validation_folds",
            "policy_frontier",
            "reliability_curve",
            "reason_specific_review_burden",
            "subgroup_metrics",
            "state_summary",
        ):
            logger.report_table(
                title="Notebook 09B",
                series=table_name,
                iteration=0,
                table_plot=_read_csv(safe_root / f"{table_name}.csv"),
            )
        for figure_name in (
            "risk_coverage_frontier.png",
            "held_out_reliability_curve.png",
            "policy_state_by_view_count.png",
            "reason_specific_review_burden.png",
        ):
            logger.report_image(
                title="Notebook 09B",
                series=figure_name,
                iteration=0,
                local_path=str(safe_root / figure_name),
            )
        task.upload_artifact("review_policy_freeze", artifact_object=str(policy_path))
        task.close()
        record = {
            "schema_version": 1,
            "status": "completed",
            "stage": "calibration_review_policy_lock_clearml_finalisation",
            "task_id": task.id,
            "safe_manifest_sha256": file_sha256(safe_root / "artifact_manifest.json"),
            "aggregate_only": True,
            "private_panels_uploaded": False,
            **provenance,
        }
    except Exception as error:  # noqa: BLE001 - the bounded finalizer records any tracker failure
        status = "disabled" if str(error) == "tracking_disabled" else "failed"
        record = {
            "schema_version": 1,
            "status": status,
            "error_type": type(error).__name__,
        }
    _write_json(receipt_path, record)
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-lock", type=Path)
    parser.add_argument("--run-record", type=Path)
    parser.add_argument("--method-record", type=Path)
    parser.add_argument("--evaluator-correspondence", type=Path)
    parser.add_argument("--model-input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--review-cost", type=float, default=0.15)
    parser.add_argument("--insufficient-cost", type=float, default=0.25)
    parser.add_argument("--example-count", type=int, default=16)
    parser.add_argument("--clearml-finalize-only", action="store_true")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.clearml_finalize_only:
        if args.run_root is None:
            parser.error("--run-root is required with --clearml-finalize-only")
    else:
        missing = [
            name
            for name in (
                "prediction_lock",
                "run_record",
                "method_record",
                "evaluator_correspondence",
                "model_input_root",
                "output_root",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"Missing required evaluation arguments: {', '.join(missing)}")
    return args


def main() -> int:
    args = parse_arguments()
    try:
        if args.clearml_finalize_only:
            result = finalise_clearml(args.run_root)
        else:
            result = run_evaluation(args)
    except Exception as error:
        if not args.clearml_finalize_only and args.output_root is not None:
            _write_json(
                args.output_root.expanduser().resolve()
                / "evaluation_failure_receipt.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "stage": "calibration_review_policy_lock",
                    "error_type": type(error).__name__,
                    "completed_outputs_remain_reusable_only_when_exact_receipts_verify": True,
                },
            )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
