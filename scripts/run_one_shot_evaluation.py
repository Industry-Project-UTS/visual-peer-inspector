#!/usr/bin/env python3
"""Run a sealed evaluator or log its safe aggregate evidence to ClearML."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.notebook_provenance import PROVENANCE_FIELDS
from visual_peer_inspector.tracking import configure_official_clearml_hosts
from visual_peer_inspector.tracking_safe_artifacts import (
    validate_tracking_safe_root,
)
from visual_peer_inspector.workflows.one_shot_evaluation import (
    create_evaluation_contract,
    run_one_shot_evaluation,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_kaggle_secrets() -> None:
    if os.environ.get("CLEARML_API_ACCESS_KEY") and os.environ.get(
        "CLEARML_API_SECRET_KEY"
    ):
        return
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as error:
        raise RuntimeError(
            "ClearML credentials are unavailable outside Kaggle"
        ) from error
    secrets = UserSecretsClient()
    os.environ["CLEARML_API_ACCESS_KEY"] = secrets.get_secret("CLEARML_API_ACCESS_KEY")
    os.environ["CLEARML_API_SECRET_KEY"] = secrets.get_secret("CLEARML_API_SECRET_KEY")


def finalize_clearml(result_root: Path) -> Mapping[str, Any]:
    """Upload only aggregate evidence after a completed evaluator run."""

    root = Path(result_root).expanduser().resolve(strict=True)
    completion_path = root / "evaluation_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError("ClearML finalization requires a completed evaluation")
    safe_root = root / "safe"
    manifest_path = safe_root / "artifact_manifest.json"
    if file_sha256(manifest_path) != completion.get("safe_manifest_sha256"):
        raise ValueError("Safe artifact manifest differs from the completion record")
    run_record, safe_manifest = validate_tracking_safe_root(
        safe_root,
        expected_stage=str(completion["stage"]),
        allowed_relative_paths={
            "clustered_confidence_intervals.csv",
            "decision_confusion.csv",
            "error_taxonomy.csv",
            "figures/margin_reliability.png",
            "figures/pipeline_attrition.png",
            "figures/coverage_and_review.png",
            "figures/ranking_by_view.png",
            "identity_failures.csv",
            "margin_agreement_reliability.csv",
            "pipeline_attrition.csv",
            "prediction_failures.csv",
            "runtime_summary.csv",
            "subgroup_and_category_metrics.csv",
            "summary_metrics.csv",
        },
    )
    if safe_manifest.get("artifact_inventory_sha256") != completion.get(
        "safe_artifact_inventory_sha256"
    ):
        raise ValueError("Safe artifact inventory differs from the completion record")
    _load_kaggle_secrets()
    configure_official_clearml_hosts()
    from clearml import Task

    stage = str(completion["stage"])
    task = Task.init(
        project_name="Visual Peer Inspector/09 Frozen Evaluation",
        task_name=f"{stage}-one-shot-evaluation",
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
    summary_rows = _read_csv(safe_root / "summary_metrics.csv")
    for row in summary_rows:
        series = f"{row['split_name']} | {row['view_count']} views"
        for field in (
            "top1_accuracy",
            "top2_accuracy",
            "mean_reciprocal_rank",
            "mean_average_precision",
            "mean_recall_at_2",
            "exact_set_accuracy",
            "target_found_rate",
            "mean_target_rank_when_found",
            "coverage",
            "selective_accuracy",
            "selective_risk",
            "review_rate",
        ):
            if row.get(field) not in {None, "", "None"}:
                logger.report_scalar(field, series, float(row[field]), iteration=0)
    for table_path in sorted(safe_root.glob("*.csv")):
        table_rows = _read_csv(table_path)
        columns = list(table_rows[0]) if table_rows else []
        logger.report_table(
            title="Frozen evaluator tables",
            series=table_path.stem,
            table_plot=[columns]
            + [[row[column] for column in columns] for row in table_rows],
            iteration=0,
        )
        task.upload_artifact(table_path.stem, artifact_object=table_path)
    for image_path in sorted((safe_root / "figures").glob("*.png")):
        logger.report_image(
            title="Frozen evaluator figures",
            series=image_path.stem,
            local_path=str(image_path),
            iteration=0,
        )
    task.upload_artifact("safe-artifact-manifest", artifact_object=manifest_path)
    task.add_tags(["final", stage, "one-shot", "no-retuning"])
    task.flush(wait_for_uploads=True)
    task.close()
    receipt = {
        "schema_version": "visual-peer-clearml-finalizer/v1",
        "status": "completed",
        "stage": stage,
        "task_id": task.id,
        "safe_manifest_sha256": file_sha256(manifest_path),
        "aggregate_only": True,
        "private_panels_uploaded": False,
        "credentials_persisted": False,
        **provenance,
    }
    _atomic_json(root / "clearml_receipt.json", receipt)
    return receipt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--stage", choices=("09d", "10b"), required=True)
    evaluate.add_argument("--evaluation-contract", type=Path, required=True)
    evaluate.add_argument("--mount-root", type=Path, action="append", required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    evaluate.add_argument("--panel-count", type=int, default=16)
    contract = subparsers.add_parser("contract")
    contract.add_argument("--stage", choices=("09d", "10b"), required=True)
    contract.add_argument("--mount-root", type=Path, action="append", required=True)
    contract.add_argument("--output", type=Path, required=True)
    track = subparsers.add_parser("track")
    track.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "contract":
        result = create_evaluation_contract(
            stage=arguments.stage,
            mounted_roots=arguments.mount_root,
            output_path=arguments.output,
        )
    elif arguments.command == "evaluate":
        try:
            result = run_one_shot_evaluation(
                contract_path=arguments.evaluation_contract,
                mounted_roots=arguments.mount_root,
                output_root=arguments.output_root,
                expected_stage=arguments.stage,
                bootstrap_repetitions=arguments.bootstrap_repetitions,
                panel_count=arguments.panel_count,
            )
        except Exception as error:
            failure_path = (
                arguments.output_root.expanduser()
                .resolve()
                .with_suffix(".failure.json")
            )
            _atomic_json(
                failure_path,
                {
                    "schema_version": "visual-peer-one-shot-failure/v1",
                    "status": "failed",
                    "stage": arguments.stage,
                    "error_type": type(error).__name__,
                    "labels_may_have_opened": (
                        arguments.output_root.expanduser()
                        .resolve()
                        .with_name(arguments.output_root.name + ".incomplete")
                        / "evaluator_access_receipt.json"
                    ).is_file(),
                    "credentials_persisted": False,
                },
            )
            raise
    else:
        result = finalize_clearml(arguments.result_root)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
