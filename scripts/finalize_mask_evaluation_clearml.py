#!/usr/bin/env python3
"""Upload only aggregate Notebook 06 evidence after evaluator compute finishes."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal


ALLOWED_TABLES = {
    "completion_gates.csv": ["gate", "observed", "expected", "status"],
    "decision_gates.csv": ["gate", "observed", "threshold", "status"],
    "proposal_gate_summary.csv": ["metric", "value"],
    "route_summary.csv": [
        "route",
        "reference_count",
        "prediction_count",
        "matched_reference_count",
        "missed_reference_count",
        "unmatched_prediction_count",
        "detection_aware_recall_at_0_75",
        "mean_matched_iou",
        "detection_aware_mean_iou",
        "mean_matched_dice",
        "detection_aware_mean_dice",
        "mean_matched_boundary_precision",
        "mean_matched_boundary_recall",
        "mean_matched_boundary_f1",
        "detection_aware_mean_boundary_precision",
        "detection_aware_mean_boundary_recall",
        "detection_aware_mean_boundary_f1",
        "mean_prediction_spill_fraction",
        "prediction_spill_rate",
        "mask_iou_threshold",
        "boundary_tolerance_pixels",
    ],
    "view_route_summary.csv": [
        "view_id",
        "route",
        "reference_count",
        "prediction_count",
        "matched_reference_count",
        "missed_reference_count",
        "unmatched_prediction_count",
        "detection_aware_recall_at_0_75",
        "detection_aware_mean_iou",
        "detection_aware_mean_dice",
        "detection_aware_mean_boundary_f1",
    ],
    "route_comparison.csv": [
        "metric",
        "reference_box",
        "automatic_box",
        "automatic_minus_reference",
    ],
    "bootstrap_intervals.csv": [
        "metric",
        "direction",
        "scene_count",
        "cluster_count",
        "baseline_mean",
        "candidate_mean",
        "mean_difference",
        "confidence_level",
        "ci_lower",
        "ci_upper",
        "repetitions",
        "seed",
    ],
}
ALLOWED_JSON = {
    "metric_definitions.json",
    "run_record.json",
    "artifact_manifest.json",
}
ALLOWED_CHARTS = {
    "reference_box_mask_iou_distribution.png",
    "automatic_box_mask_iou_distribution.png",
    "reference_box_boundary_f1_distribution.png",
    "automatic_box_boundary_f1_distribution.png",
    "reference_box_recall_by_view.png",
    "automatic_box_recall_by_view.png",
    "automatic_box_prompt_iou_vs_mask_iou.png",
    "route_metric_comparison.png",
    "misses_and_extras_by_route.png",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_|mask_rle|box_xyxy|raw_response|"
    r"labels_by_object|source_member|/kaggle/input/|/Users/|\\Users\\",
    re.IGNORECASE,
)
DEFAULT_PROJECT = "Visual Peer Inspector/02 Detection and Segmentation"
TASK_NAME_PATTERN = re.compile(
    r"^ds0-sam-mask-evaluation-kaggle-(candidate|final)$"
)
ALLOWED_TAGS = {
    "candidate",
    "final",
    "notebook-06",
    "ds0",
    "sam2.1",
    "locateanything",
    "evaluation",
    "kaggle",
}
EXPECTED_RUN_KEYS = {
    "schema_version",
    "stage",
    "status",
    "tracking_status",
    "evaluation_mode",
    "evidence_eligible",
    "completion_gate_passed",
    "scientific_gates_passed",
    "quality_claim",
    "sam_producer_kind",
    "python_version",
    "evaluator_provenance",
    "settings",
    "coverage",
    "inputs",
    "proposal_gate_summary",
    "route_summary",
    "bootstrap_summary",
    "candidate_decision",
    "ranking_metrics_status",
}
FORBIDDEN_JSON_KEYS = {
    "anomaly",
    "anomaly_label",
    "asset_group_id",
    "box_xyxy",
    "category",
    "mask_rle",
    "object_id",
    "raw_response",
    "source_id",
    "source_member",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--task-name", default="ds0-sam-mask-evaluation-kaggle-candidate"
    )
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise ValueError("ClearML manifest contains an unsafe path")
    return Path(*pure.parts)


def _reject_nonstandard_json_constant(value: str):
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def _validate_safe_json(value, location: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe_json(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Non-string JSON key at {location}")
            lowered = key.lower()
            if lowered.startswith("opaque_") or lowered in FORBIDDEN_JSON_KEYS:
                raise ValueError(f"Protected JSON field found at {location}.{key}")
            _validate_safe_json(item, f"{location}.{key}")
        return
    raise ValueError(f"Unsupported JSON value at {location}")


def validate_safe_root(root: Path) -> dict:
    """Validate the complete aggregate allowlist before ClearML is imported."""

    expanded_root = root.expanduser()
    if expanded_root.is_symlink():
        raise ValueError("ClearML safe root is missing or unsafe")
    root = expanded_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("ClearML safe root is missing or unsafe")
    expected_files = set(ALLOWED_TABLES) | ALLOWED_JSON | ALLOWED_CHARTS
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("ClearML safe root cannot contain symbolic links")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if Path(relative).name != relative:
                raise ValueError("ClearML safe root can contain files only")
            actual_files.add(relative)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            "ClearML safe allowlist differs; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name, expected_columns in ALLOWED_TABLES.items():
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML table: {name}")
        rows = list(csv.reader(text.splitlines()))
        if not rows or rows[0] != expected_columns:
            raise ValueError(f"Unexpected ClearML columns in {name}")
    for name in ALLOWED_JSON:
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML JSON: {name}")
        value = json.loads(text, parse_constant=_reject_nonstandard_json_constant)
        _validate_safe_json(value, name)

    manifest = json.loads(
        (root / "artifact_manifest.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )
    rows = manifest.get("artifacts")
    if (
        not isinstance(rows, list)
        or int(manifest.get("artifact_count", -1)) != len(rows)
    ):
        raise ValueError("ClearML artifact manifest count is inconsistent")
    expected_manifest_paths = sorted(expected_files - {"artifact_manifest.json"})
    listed_paths = [str(row.get("path")) for row in rows]
    if listed_paths != expected_manifest_paths:
        raise ValueError("ClearML artifact manifest differs from the safe allowlist")
    for row in rows:
        if set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("ClearML artifact manifest row has an invalid schema")
        relative = _safe_relative(str(row["path"]))
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != str(row["sha256"])
            or path.stat().st_size != int(row["size_bytes"])
        ):
            raise ValueError(f"ClearML artifact differs for {relative.as_posix()}")
    content_digest = sha256(
        b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    ).hexdigest()
    if content_digest != manifest.get("manifest_content_sha256"):
        raise ValueError("ClearML artifact manifest content hash differs")

    run_record = json.loads(
        (root / "run_record.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )
    if set(run_record) != EXPECTED_RUN_KEYS:
        raise ValueError("ClearML run record does not follow the accepted schema")
    required = {
        "stage": "ds0_sam21_mask_evaluation",
        "status": "completed",
        "evaluation_mode": "model_evidence",
        "evidence_eligible": True,
        "completion_gate_passed": True,
        "sam_producer_kind": "sam2.1_model",
        "ranking_metrics_status": "deferred_to_notebook_07",
    }
    for key, expected in required.items():
        if run_record.get(key) != expected:
            raise ValueError(f"ClearML requires accepted model evidence: {key}")
    coverage = run_record.get("coverage", {})
    expected_coverage = {
        "scene_count": 20,
        "view_count_per_scene": 20,
        "image_count": 400,
        "reference_mask_count": 1640,
        "sam_mask_count": 3256,
        "reference_prompt_count": 1640,
        "automatic_prompt_count": 1616,
    }
    if coverage != expected_coverage:
        raise ValueError("ClearML requires complete DS0 evaluator coverage")
    decision = run_record.get("candidate_decision", {})
    if decision.get("final_route_acceptance") is not False:
        raise ValueError("Notebook 06 cannot claim final route acceptance")
    if decision.get("proceed_to_ds1") is not run_record.get(
        "scientific_gates_passed"
    ):
        raise ValueError("Notebook 06 candidate decision is inconsistent")
    proposal_dirty = decision.get("proposal_git_dirty")
    sam_dirty = decision.get("sam_git_dirty")
    evaluator_dirty = decision.get("evaluator_git_dirty")
    if (
        not isinstance(proposal_dirty, bool)
        or not isinstance(sam_dirty, bool)
        or not isinstance(evaluator_dirty, bool)
    ):
        raise ValueError("Notebook 06 provenance decision is incomplete")
    if proposal_dirty or sam_dirty or evaluator_dirty:
        expected_status = "candidate"
    elif run_record["scientific_gates_passed"]:
        expected_status = "ds0_gate_passed"
    else:
        expected_status = "ds0_gate_failed"
    if decision.get("status") != expected_status:
        raise ValueError("Notebook 06 provenance decision status is inconsistent")
    for value in run_record.get("inputs", {}).values():
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("ClearML provenance hashes are invalid")
    provenance = run_record.get("evaluator_provenance", {})
    expected_provenance_keys = {
        "git_commit",
        "git_dirty",
        "snapshot_sha256",
        "precommit_candidate",
        "notebook_sha256",
        "kaggle_notebook_slug",
        "kaggle_notebook_version",
        "code_dataset_version",
    }
    if set(provenance) != expected_provenance_keys:
        raise ValueError("Notebook 06 evaluator provenance is incomplete")
    if provenance["git_dirty"] is not evaluator_dirty:
        raise ValueError("Notebook 06 evaluator provenance status is inconsistent")
    if not isinstance(provenance["precommit_candidate"], bool):
        raise ValueError("Notebook 06 evaluator precommit status is invalid")
    for key, length in {
        "git_commit": 40,
        "snapshot_sha256": 64,
        "notebook_sha256": 64,
    }.items():
        text = provenance[key]
        if (
            not isinstance(text, str)
            or len(text) != length
            or any(character not in "0123456789abcdef" for character in text)
        ):
            raise ValueError(f"Notebook 06 evaluator {key} is invalid")
    if provenance["kaggle_notebook_slug"] != (
        "esreekarreddy/toysad-8k-ds0-proposal-and-mask-evaluation"
    ):
        raise ValueError("Notebook 06 evaluator notebook slug is invalid")
    for key in ("kaggle_notebook_version", "code_dataset_version"):
        text = provenance[key]
        if not isinstance(text, str) or not text or "/" in text or "\\" in text:
            raise ValueError(f"Notebook 06 evaluator {key} is invalid")
    return run_record


def _install_wall_timer(seconds: int) -> None:
    if seconds < 30 or seconds > 600:
        raise ValueError("ClearML wall time must be between 30 and 600 seconds")
    if not hasattr(signal, "SIGALRM"):
        return

    def expired(_signum, _frame):
        raise TimeoutError("ClearML finalizer exceeded its wall-time limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)


def _write_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.project != DEFAULT_PROJECT:
        raise ValueError("Notebook 06 ClearML project name is fixed")
    if not TASK_NAME_PATTERN.fullmatch(args.task_name):
        raise ValueError("Notebook 06 ClearML task name is invalid")
    status_tag = "final" if args.task_name.endswith("-final") else "candidate"
    required_tags = {
        status_tag,
        "notebook-06",
        "ds0",
        "sam2.1",
        "locateanything",
        "evaluation",
        "kaggle",
    }
    tags = args.tag or sorted(required_tags)
    if set(tags) - ALLOWED_TAGS or set(tags) != required_tags:
        raise ValueError("Notebook 06 ClearML tags are invalid")
    run_record = validate_safe_root(args.safe_root)
    safe_root = args.safe_root.expanduser().resolve(strict=True)
    expanded_receipt = args.receipt.expanduser()
    if expanded_receipt.is_symlink():
        raise ValueError("ClearML receipt path cannot be a symbolic link")
    receipt_path = expanded_receipt.resolve()
    if receipt_path == safe_root or safe_root in receipt_path.parents:
        raise ValueError("ClearML receipt must remain outside the immutable safe root")
    if (
        status_tag == "final"
        and run_record["candidate_decision"]["status"] != "ds0_gate_passed"
    ):
        raise ValueError("Only a clean passing DS0 run can be tagged final")
    if args.mode == "disabled":
        _write_receipt(
            args.receipt,
            {
                "schema_version": 1,
                "status": "disabled",
                "tracking_mode": "disabled",
                "clearml_task_id": None,
                "project": args.project,
                "task_name": args.task_name,
            },
        )
        print("tracking disabled")
        return

    _install_wall_timer(args.wall_time_seconds)
    os.environ.setdefault("VPI_CLEARML_REQUEST_TIMEOUT_SECONDS", "30")
    from clearml import Task
    from visual_peer_inspector.tracking import _configure_clearml_request_timeouts

    if args.mode == "offline":
        Task.set_offline(True)
    _configure_clearml_request_timeouts()
    task = Task.init(
        project_name=args.project,
        task_name=args.task_name,
        task_type=Task.TaskTypes.testing,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_resource_monitoring=False,
        auto_connect_streams=False,
        reuse_last_task_id=False,
    )
    task.add_tags(tags)
    logger = task.get_logger()
    for route in run_record["route_summary"]:
        route_name = str(route["route"])
        for metric in (
            "detection_aware_recall_at_0_75",
            "mean_matched_iou",
            "detection_aware_mean_iou",
            "mean_matched_dice",
            "mean_matched_boundary_f1",
            "mean_prediction_spill_fraction",
        ):
            logger.report_scalar(
                "DS0 mask routes",
                f"{route_name} {metric}",
                float(route[metric]),
                iteration=0,
            )
    for metric in (
        "exact_count_rate",
        "micro_recall_at_0_50",
        "micro_recall_at_0_75",
        "mean_matched_iou",
        "detection_aware_mean_iou",
    ):
        logger.report_scalar(
            "DS0 proposal gate",
            metric,
            float(run_record["proposal_gate_summary"][metric]),
            iteration=0,
        )
    root = safe_root
    for name in sorted(ALLOWED_TABLES):
        with (root / name).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        logger.report_table(
            "DS0 mask evaluation",
            name.removesuffix(".csv").replace("_", " "),
            table_plot=rows,
            iteration=0,
        )
    for name in sorted(ALLOWED_CHARTS):
        logger.report_image(
            "DS0 mask evaluation",
            name.removesuffix(".png").replace("_", " "),
            local_path=str(root / name),
            iteration=0,
        )
    task.connect(
        {
            **run_record["inputs"],
            "evaluation_mode": run_record["evaluation_mode"],
            "scientific_gates_passed": run_record["scientific_gates_passed"],
            "decision_status": run_record["candidate_decision"]["status"],
            "proceed_to_ds1": run_record["candidate_decision"]["proceed_to_ds1"],
            "evaluator_git_commit": run_record["evaluator_provenance"][
                "git_commit"
            ],
            "evaluator_git_dirty": run_record["evaluator_provenance"][
                "git_dirty"
            ],
            "evaluator_snapshot_sha256": run_record["evaluator_provenance"][
                "snapshot_sha256"
            ],
            "evaluator_notebook_sha256": run_record["evaluator_provenance"][
                "notebook_sha256"
            ],
            "code_dataset_version": run_record["evaluator_provenance"][
                "code_dataset_version"
            ],
        },
        name="verified aggregate evidence",
    )
    task_id = str(task.id)
    offline_folder = (
        str(task.get_offline_mode_folder()) if args.mode == "offline" else None
    )
    task.flush(wait_for_uploads=True)
    task.close()
    if hasattr(signal, "alarm"):
        signal.alarm(0)
    _write_receipt(
        args.receipt,
        {
            "schema_version": 1,
            "status": "completed",
            "tracking_mode": args.mode,
            "clearml_task_id": task_id,
            "offline_session_folder": offline_folder,
            "project": args.project,
            "task_name": args.task_name,
        },
    )
    print(task_id)


if __name__ == "__main__":
    main()
