#!/usr/bin/env python3
"""Upload only aggregate Notebook 07 evidence after computation has finished."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
from typing import Any


ALLOWED_TABLES = {
    "encoder_availability.csv",
    "runtime_summary.csv",
    "encoder_summary.csv",
    "scorer_summary.csv",
    "route_summary.csv",
    "scene_type_summary.csv",
    "geometry_summary.csv",
    "view_summary.csv",
    "clustered_intervals.csv",
    "paired_differences.csv",
    "coverage_summary.csv",
    "completion_gates.csv",
    "model_agreement.csv",
}
ALLOWED_JSON = {
    "resolved_configuration.json",
    "resolved_model_revisions.json",
    "input_audit.json",
    "model_registry.json",
    "cache_summary.json",
    "label_free_run_record.json",
    "metric_definitions.json",
    "promotion_decision.json",
    "run_record.json",
    "artifact_manifest.json",
}
ALLOWED_CHARTS = {
    "figures/encoder_b0_scene_ap.png",
    "figures/anchor_scorer_comparison.png",
    "figures/anchor_route_comparison.png",
    "figures/one_vs_two_anomaly_scene_ap.png",
    "figures/model_agreement.png",
    "figures/encoder_efficiency.png",
    "figures/rankable_coverage.png",
    "figures/geometry_vs_appearance.png",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_scene|opaque_asset|labels_by_object|mask_rle|"
    r"box_xyxy|/kaggle/input/|/Users/|HF_TOKEN|CLEARML_API_(ACCESS|SECRET)_KEY",
    re.IGNORECASE,
)
DEFAULT_PROJECT = "Visual Peer Inspector/03 Representation and Peer Scoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--task-name", default="ds0-representation-peer-scores-kaggle-candidate"
    )
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_safe_root(
    root: Path, *, require_model_evidence: bool = True
) -> dict[str, Any]:
    """Reject unexpected files and protected content before importing ClearML."""

    expanded = root.expanduser()
    if expanded.is_symlink():
        raise ValueError("ClearML safe root is missing or unsafe")
    root = expanded.resolve(strict=True)
    expected = ALLOWED_TABLES | ALLOWED_JSON | ALLOWED_CHARTS
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("ClearML safe root cannot contain symbolic links")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError(
            f"ClearML safe allowlist differs; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    for relative in sorted(ALLOWED_TABLES | (ALLOWED_JSON - {"artifact_manifest.json"})):
        text = (root / relative).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML artifact: {relative}")
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise ValueError("ClearML artifact manifest count is invalid")
    listed = [str(row.get("path")) for row in rows]
    if listed != sorted(expected - {"artifact_manifest.json"}):
        raise ValueError("ClearML artifact manifest differs from its allowlist")
    for row in rows:
        path = root / str(row["path"])
        if (
            sha256(path.read_bytes()).hexdigest() != row["sha256"]
            or path.stat().st_size != int(row["size_bytes"])
        ):
            raise ValueError("ClearML artifact checksum differs")
    run_record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if run_record.get("status") != "completed":
        raise ValueError("ClearML requires a completed representation run")
    if require_model_evidence and run_record.get("run_mode") != "model_evidence":
        raise ValueError("ClearML requires model-evidence outputs, not contract data")
    if run_record.get("decision") != "retain_dinov2_small_anchor_pending_ds1":
        raise ValueError("Notebook 07 must retain the provisional DS0 claim boundary")
    return run_record


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _log_online(root: Path, project: str, task_name: str, tags: list[str]) -> str:
    from clearml import Task

    task = Task.init(
        project_name=project,
        task_name=task_name,
        reuse_last_task_id=False,
        output_uri=False,
    )
    if tags:
        task.add_tags(tags)
    logger = task.get_logger()
    for relative in sorted(ALLOWED_TABLES):
        logger.report_table(
            title=relative.removesuffix(".csv"),
            series="aggregate",
            iteration=0,
            csv=str(root / relative),
        )
        task.upload_artifact(relative, artifact_object=root / relative, wait_on_upload=False)
    for relative in sorted(ALLOWED_CHARTS):
        logger.report_image(
            title=Path(relative).stem,
            series="aggregate",
            iteration=0,
            local_path=str(root / relative),
        )
    for relative in sorted(ALLOWED_JSON - {"artifact_manifest.json"}):
        task.upload_artifact(relative, artifact_object=root / relative, wait_on_upload=False)
    task.flush(wait_for_uploads=True)
    task.close()
    return str(task.id)


def main() -> None:
    args = parse_args()
    root = args.safe_root.expanduser().resolve()
    record = validate_safe_root(
        root,
        require_model_evidence=args.mode != "disabled",
    )
    receipt = {
        "schema_version": 1,
        "stage": "notebook_07_clearml_finalisation",
        "mode": args.mode,
        "status": "pending",
        "safe_manifest_sha256": sha256(
            (root / "artifact_manifest.json").read_bytes()
        ).hexdigest(),
        "task_id": None,
        "run_status": record["status"],
    }
    if args.mode == "disabled":
        receipt["status"] = "disabled"
        _write_receipt(args.receipt, receipt)
        return
    if args.mode == "offline":
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(args.wall_time_seconds)
    try:
        receipt["task_id"] = _log_online(
            root, args.project, args.task_name, args.tag
        )
        receipt["status"] = "completed" if args.mode == "online" else "offline_completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        receipt["error_message"] = str(error)[:300]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        _write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
