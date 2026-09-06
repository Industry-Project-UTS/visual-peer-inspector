#!/usr/bin/env python3
"""Publish only aggregate DS2 evidence after a notebook has finished."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.notebook_provenance import PROVENANCE_FIELDS
from visual_peer_inspector.tracking_safe_artifacts import (
    validate_tracking_safe_root,
)

DEFAULT_PROJECT = "Visual Peer Inspector/04 Confirmation and Multi-view"

# Every externally tracked stage has a code-owned aggregate artifact contract.
# The manifest and run record are added by the shared validator.
STAGE_ALLOWLISTS: dict[str, tuple[frozenset[str], ...]] = {
    "primary_protocol_release": (
        frozenset(
            {
                "cohort_membership.csv",
                "development_stage_membership.csv",
                "role_overlap_audit.csv",
                "resource_release_status.csv",
                "primary_role_counts.png",
                "protocol_release_boundary.png",
            }
        ),
    ),
    "frozen_development_pipeline": (frozenset(),),
    "development_identity_evaluator_release": (frozenset(),),
    "calibration_resource_release": (frozenset(),),
    "internal_lock_resource_release": (frozenset(),),
    "protected_calibration_inference": (
        frozenset(),
        frozenset(
            {
                "runtime_receipt.json",
                "view_contract_summary.csv",
                "review_signal_summary.csv",
                "multiview_completion.png",
            }
        ),
    ),
    "protected_internal_lock_inference": (
        frozenset(),
        frozenset(
            {
                "runtime_receipt.json",
                "view_contract_summary.csv",
                "review_signal_summary.csv",
                "multiview_completion.png",
            }
        ),
    ),
    "official_seen_frozen_inference": (
        frozenset(),
        frozenset(
            {
                "runtime_receipt.json",
                "view_contract_summary.csv",
                "review_signal_summary.csv",
                "multiview_completion.png",
            }
        ),
    ),
    "official_unseen_frozen_inference": (
        frozenset(),
        frozenset(
            {
                "runtime_receipt.json",
                "view_contract_summary.csv",
                "review_signal_summary.csv",
                "multiview_completion.png",
            }
        ),
    ),
    "official_release": (
        frozenset(),
        frozenset(
            {
                "official_release_receipt.json",
                "official_cohort_summary.csv",
                "official_cohort_summary.png",
            }
        ),
    ),
    "official_frozen_inference": (
        frozenset(
            {
                "official_inference_summary.json",
                "official_prediction_lock_summary.csv",
                "official_prediction_lock_summary.png",
            }
        ),
    ),
    "official_inference_recovery": (
        frozenset({"recovery_run_summary.json"}),
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_safe_root(root: Path, *, expected_stage: str) -> dict[str, Any]:
    """Verify the aggregate manifest against the exact stage contract."""

    variants = STAGE_ALLOWLISTS.get(expected_stage)
    if variants is None:
        raise ValueError("ClearML stage has no code-owned artifact allowlist")
    last_error: ValueError | None = None
    for allowed in variants:
        try:
            record, _manifest = validate_tracking_safe_root(
                root,
                expected_stage=expected_stage,
                allowed_relative_paths=allowed,
            )
            return record
        except ValueError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _numeric_scalars(value: Any, prefix: str = "run", limit: int = 100) -> list[tuple[str, float]]:
    """Return a bounded list of numeric values from an already safe record."""

    rows: list[tuple[str, float]] = []

    def visit(item: Any, name: str) -> None:
        if len(rows) >= limit:
            return
        if isinstance(item, (bool, int, float)):
            rows.append((name, float(item)))
        elif isinstance(item, dict):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                visit(child, f"{name}.{key}")

    visit(value, prefix)
    return rows


def _log_online(
    root: Path,
    project: str,
    task_name: str,
    tags: list[str],
    record: dict[str, Any],
    *,
    online: bool,
) -> tuple[str, dict[str, int]]:
    from clearml import Task

    from visual_peer_inspector.tracking import (
        _configure_clearml_request_timeouts,
        configure_official_clearml_hosts,
        initialize_clearml_task_with_retry,
    )

    configure_official_clearml_hosts()
    _configure_clearml_request_timeouts()
    task, retry_outcome = initialize_clearml_task_with_retry(
        lambda: Task.init(
            project_name=project,
            task_name=task_name,
            reuse_last_task_id=False,
            output_uri=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_connect_streams=False,
            auto_resource_monitoring=False,
        ),
        online=online,
    )
    if tags:
        task.add_tags(tags)
    provenance = {
        name: record[name]
        for name in PROVENANCE_FIELDS
        if name in record
    }
    if provenance:
        task.set_user_properties(**provenance)
    logger = task.get_logger()
    for series, value in _numeric_scalars(record):
        logger.report_scalar(
            title="DS2 aggregate run record",
            series=series,
            value=value,
            iteration=0,
        )
    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root).as_posix()
        logger.report_table(
            title=path.stem,
            series="aggregate",
            iteration=0,
            csv=str(path),
        )
        task.upload_artifact(relative, artifact_object=path, wait_on_upload=False)
    for path in sorted(root.rglob("*.png")):
        logger.report_image(
            title=path.stem,
            series="aggregate",
            iteration=0,
            local_path=str(path),
        )
    for path in sorted(root.rglob("*.json")):
        if path.name == "artifact_manifest.json":
            continue
        task.upload_artifact(path.relative_to(root).as_posix(), artifact_object=path, wait_on_upload=False)
    task.flush(wait_for_uploads=True)
    task.close()
    return str(task.id), retry_outcome.to_receipt_fields()


def main() -> None:
    args = parse_arguments()
    root = args.safe_root.expanduser().resolve()
    record = validate_safe_root(root, expected_stage=args.stage)
    receipt = {
        "schema_version": 1,
        "stage": f"{args.stage}_clearml_finalisation",
        "mode": args.mode,
        "status": "pending",
        "safe_manifest_sha256": sha256(
            (root / "artifact_manifest.json").read_bytes()
        ).hexdigest(),
        "task_id": None,
        "run_status": record["status"],
        "tracking_attempt_count": 0,
        "tracking_retry_count": 0,
        "tracking_transient_failure_count": 0,
        **{
            name: record[name]
            for name in PROVENANCE_FIELDS
            if name in record
        },
    }
    if args.mode == "disabled":
        receipt["status"] = "disabled"
        _write_receipt(args.receipt, receipt)
        return
    if args.mode == "offline":
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(30, min(args.wall_time_seconds, 300)))
    try:
        receipt["task_id"], retry_fields = _log_online(
            root,
            args.project,
            args.task_name,
            args.tag,
            record,
            online=args.mode == "online",
        )
        receipt.update(retry_fields)
        receipt["status"] = "completed" if args.mode == "online" else "offline_completed"
    except Exception as error:  # noqa: BLE001 - tracking failure must not erase local evidence
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        _write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
