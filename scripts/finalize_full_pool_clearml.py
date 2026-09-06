#!/usr/bin/env python3
"""Publish bounded aggregate evidence for the optional full-pool stages."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.notebook_provenance import PROVENANCE_FIELDS

ALLOWED = {
    "post_freeze_full_pool_release": {
        "run_record.json",
        "primary_role_coverage.csv",
        "primary_role_coverage.png",
        "artifact_manifest.json",
    },
    "post_freeze_full_pool_inference": {
        "run_record.json",
        "runtime_receipt.json",
        "view_contract_summary.csv",
        "review_signal_summary.csv",
        "multiview_completion.png",
        "artifact_manifest.json",
    },
    "post_freeze_full_pool_sensitivity_evaluation": {
        "run_record.json",
        "summary_metrics.csv",
        "clustered_confidence_intervals.csv",
        "subgroup_and_category_metrics.csv",
        "margin_agreement_reliability.csv",
        "identity_failures.csv",
        "pipeline_attrition.csv",
        "prediction_failures.csv",
        "runtime_summary.csv",
        "error_taxonomy.csv",
        "decision_confusion.csv",
        "figures/margin_reliability.png",
        "figures/pipeline_attrition.png",
        "figures/coverage_and_review.png",
        "figures/ranking_by_view.png",
        "artifact_manifest.json",
    },
}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify(root: Path, stage: str) -> tuple[dict[str, Any], list[Path]]:
    source = Path(root).expanduser().resolve(strict=True)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    relative = {path.relative_to(source).as_posix() for path in files}
    recovery_only = {"run_record.json", "artifact_manifest.json"}
    if relative != ALLOWED[stage] and not (
        stage == "post_freeze_full_pool_inference" and relative == recovery_only
    ):
        raise ValueError("Full-pool ClearML evidence differs from the stage allowlist")
    manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise ValueError("Full-pool safe manifest is invalid")
    if {str(row.get("path")) for row in rows} != relative - {"artifact_manifest.json"}:
        raise ValueError("Full-pool safe manifest coverage differs")
    for row in rows:
        path = source / str(row["path"])
        if file_sha256(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("Full-pool safe artifact differs from its manifest")
    run = json.loads((source / "run_record.json").read_text(encoding="utf-8"))
    if run.get("stage") != stage or run.get("status") not in {"completed", "checkpointed"}:
        raise ValueError("Full-pool run record is incomplete or for another stage")
    if run.get("descriptive_sensitivity_only") is not True or run.get("no_retuning") is not True:
        raise ValueError("Full-pool evidence does not preserve the frozen interpretation")
    return run, files


def _numeric(value: Any, prefix: str = "run") -> list[tuple[str, float]]:
    output = []

    def visit(item: Any, name: str) -> None:
        if len(output) >= 100:
            return
        if isinstance(item, (bool, int, float)):
            output.append((name, float(item)))
        elif isinstance(item, Mapping):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                visit(child, f"{name}.{key}")

    visit(value, prefix)
    return output


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalization exceeded its wall-time limit")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(ALLOWED), required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run, files = _verify(args.safe_root, args.stage)
    receipt: dict[str, Any] = {
        "schema_version": "visual-peer-11-clearml/v1",
        "status": "disabled" if args.mode == "disabled" else "pending",
        "stage": args.stage,
        "mode": args.mode,
        "aggregate_only": True,
        "private_panels_uploaded": False,
        "credentials_persisted": False,
        "safe_manifest_sha256": file_sha256(args.safe_root / "artifact_manifest.json"),
        "task_id": None,
        **{name: run[name] for name in PROVENANCE_FIELDS if name in run},
    }
    if args.mode == "disabled":
        _write(args.receipt, receipt)
        return
    if args.mode == "offline":
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(30, min(args.wall_time_seconds, 300)))
    try:
        from clearml import Task

        from visual_peer_inspector.tracking import configure_official_clearml_hosts

        configure_official_clearml_hosts()
        task = Task.init(
            project_name="Visual Peer Inspector/11 Post-freeze Sensitivity",
            task_name=args.task_name,
            reuse_last_task_id=False,
            output_uri=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_connect_streams=False,
            auto_resource_monitoring=False,
        )
        if args.tag:
            task.add_tags([*args.tag, "descriptive", "no-retuning"])
        provenance = {name: run[name] for name in PROVENANCE_FIELDS if name in run}
        if provenance:
            task.set_user_properties(**provenance)
        logger = task.get_logger()
        for name, value in _numeric(run):
            logger.report_scalar("Aggregate run record", name, value, iteration=0)
        for path in files:
            if path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.reader(handle))
                logger.report_table(path.stem, "aggregate", table_plot=rows, iteration=0)
            elif path.suffix == ".png":
                logger.report_image(path.stem, "aggregate", local_path=str(path), iteration=0)
            elif path.name != "artifact_manifest.json":
                task.upload_artifact(path.stem, artifact_object=path, wait_on_upload=False)
        task.flush(wait_for_uploads=True)
        task.close()
        receipt["task_id"] = str(task.id)
        receipt["status"] = "completed" if args.mode == "online" else "offline_completed"
    except Exception as error:  # noqa: BLE001 - tracking cannot invalidate local evidence
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        _write(args.receipt, receipt)


if __name__ == "__main__":
    main()
