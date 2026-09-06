#!/usr/bin/env python3
"""Run Notebook 08F and bound optional ClearML finalisation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.notebook_provenance import PROVENANCE_FIELDS
from visual_peer_inspector.tracking_safe_artifacts import (
    validate_tracking_safe_root,
)
from visual_peer_inspector.workflows.cross_view_identity_freeze import (
    run_cross_view_identity_freeze,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _clearml_child(arguments: argparse.Namespace) -> None:
    """Upload aggregate evidence in a process the parent can terminate."""

    from clearml import Task

    safe_root = arguments.safe_root.expanduser().resolve(strict=True)
    run_record, _manifest = validate_tracking_safe_root(
        safe_root,
        expected_stage="cross_view_identity_freeze",
        allowed_relative_paths={
            "acceptance_gates.csv",
            "application_output_contract.json",
            "clustered_intervals.csv",
            "failure_summary.csv",
            "figures/five_view_identity_accuracy.png",
            "figures/missing_view_link_stability.png",
            "method_freeze.json",
            "method_summary.csv",
            "paired_method_intervals.csv",
            "stability_summary.csv",
        },
    )
    task = Task.init(
        project_name=arguments.clearml_project,
        task_name=arguments.clearml_task_name,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_connect_streams=False,
        auto_resource_monitoring=False,
        output_uri=False,
    )
    task.add_tags(
        ["notebook-08f", "development-freeze", "cross-view-identity", "final"]
    )
    provenance = {
        name: run_record[name]
        for name in PROVENANCE_FIELDS
        if name in run_record
    }
    if provenance:
        task.set_user_properties(**provenance)
    task.connect(
        {
            "stage": run_record["stage"],
            "frozen_model_identity_sha256": run_record["frozen_model_identity_sha256"],
            "identity_prediction_lock_sha256": run_record[
                "identity_prediction_lock_sha256"
            ],
            "chosen_variant": run_record["chosen_variant"],
            "scene_count": run_record["scene_count"],
            "absolute_performance_threshold_used": False,
        },
        name="frozen_configuration",
    )
    logger = task.get_logger()
    logger.report_single_value("scene_count", int(run_record["scene_count"]))
    logger.report_single_value("prediction_count", int(run_record["prediction_count"]))
    logger.report_single_value("failure_count", int(run_record["failure_count"]))
    logger.report_single_value(
        "qualitative_panel_count", int(run_record["qualitative_panel_count"])
    )

    uploaded = []
    for path in sorted(safe_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(safe_root).as_posix()
        if path.suffix.lower() == ".png":
            logger.report_image(
                title="Notebook 08F evidence",
                series=relative,
                iteration=0,
                local_path=str(path),
            )
        task.upload_artifact(
            name=relative.replace("/", "__"),
            artifact_object=str(path),
            wait_on_upload=False,
        )
        uploaded.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
            }
        )
    task.flush(wait_for_uploads=True)
    task_id = task.id
    task.close()
    _write_json(
        arguments.child_receipt,
        {
            "schema_version": "visual-peer-clearml-receipt/v1",
            "status": "complete",
            "task_id": task_id,
            "uploaded_artifact_count": len(uploaded),
            "uploaded_artifacts": uploaded,
            **provenance,
        },
    )


def _finalise_clearml(
    *,
    safe_root: Path,
    output_root: Path,
    mode: str,
    project: str,
    task_name: str,
    wall_time_seconds: int,
) -> dict[str, Any]:
    receipt_path = output_root / "tracking" / "clearml_receipt.json"
    child_receipt = output_root / "tracking" / "clearml_receipt.child.json"
    if mode == "disabled":
        record = {
            "schema_version": "visual-peer-clearml-receipt/v1",
            "status": "disabled",
            "task_id": None,
            "reason": "tracking disabled by configuration",
        }
        _write_json(receipt_path, record)
        return record

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--clearml-finalize-only",
        "--safe-root",
        str(safe_root),
        "--child-receipt",
        str(child_receipt),
        "--clearml-project",
        project,
        "--clearml-task-name",
        task_name,
    ]
    try:
        subprocess.run(
            command, check=True, timeout=wall_time_seconds, env=os.environ.copy()
        )
        record = json.loads(child_receipt.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        record = {
            "schema_version": "visual-peer-clearml-receipt/v1",
            "status": "timed_out",
            "task_id": None,
            "error_type": "TimeoutExpired",
            "wall_time_seconds": wall_time_seconds,
        }
    except Exception as error:  # noqa: BLE001 - the bounded finalizer records any tracker failure
        record = {
            "schema_version": "visual-peer-clearml-receipt/v1",
            "status": "failed",
            "task_id": None,
            "error_type": type(error).__name__,
        }
    child_receipt.unlink(missing_ok=True)
    _write_json(receipt_path, record)
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, action="append", default=[])
    parser.add_argument("--evaluator-correspondence", type=Path)
    parser.add_argument("--media-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--unmatched-cost", type=float, default=0.35)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--geometry-weight", type=float, default=0.15)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=32_040)
    parser.add_argument("--qualitative-count", type=int, default=15)
    parser.add_argument(
        "--clearml-mode", choices=("online", "disabled"), default="disabled"
    )
    parser.add_argument(
        "--clearml-project",
        default="Visual Peer Inspector/05 Development Freeze",
    )
    parser.add_argument(
        "--clearml-task-name",
        default="cross-view-identity-freeze",
    )
    parser.add_argument("--clearml-wall-time-seconds", type=int, default=180)
    parser.add_argument(
        "--clearml-finalize-only", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--safe-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-receipt", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.clearml_finalize_only:
        if arguments.safe_root is None or arguments.child_receipt is None:
            raise ValueError("ClearML child mode requires safe-root and child-receipt")
        _clearml_child(arguments)
        return
    if (
        not arguments.prediction_root
        or arguments.evaluator_correspondence is None
        or not arguments.media_root
        or arguments.output_root is None
    ):
        raise ValueError(
            "prediction-root, evaluator-correspondence, media-root, and output-root are required"
        )
    if arguments.clearml_wall_time_seconds < 1:
        raise ValueError("clearml-wall-time-seconds must be positive")
    record = run_cross_view_identity_freeze(
        prediction_roots=arguments.prediction_root,
        evaluator_correspondence_path=arguments.evaluator_correspondence,
        media_roots=arguments.media_root,
        output_root=arguments.output_root,
        unmatched_cost=arguments.unmatched_cost,
        ambiguity_margin=arguments.ambiguity_margin,
        geometry_weight=arguments.geometry_weight,
        bootstrap_repetitions=arguments.bootstrap_repetitions,
        bootstrap_seed=arguments.bootstrap_seed,
        qualitative_count=arguments.qualitative_count,
    )
    tracking = _finalise_clearml(
        safe_root=arguments.output_root / "safe",
        output_root=arguments.output_root,
        mode=arguments.clearml_mode,
        project=arguments.clearml_project,
        task_name=arguments.clearml_task_name,
        wall_time_seconds=arguments.clearml_wall_time_seconds,
    )
    print(
        json.dumps({"run": record, "clearml": tracking}, indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
