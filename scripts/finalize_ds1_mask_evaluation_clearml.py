#!/usr/bin/env python3
"""Publish only verified aggregate Notebook 06.1 evidence to ClearML."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_RELATIVE = Path(
    "notebooks/detection-and-segmentation/toysad8k-ds1-mask-evaluation/"
    "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb"
)
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.tracking import (
    OFFICIAL_CLEARML_HOSTS,
    _configure_clearml_request_timeouts,
    configure_official_clearml_hosts,
    initialize_clearml_task_with_retry,
    sanitized_tracking_failure_fields,
)


ALLOWED_TABLES = {
    "proposal_summary.csv": ["metric", "value"],
    "proposal_subgroups.csv": [
        "dimension", "value", "image_count", "proposal_count", "reference_count",
        "precision_at_0_50", "recall_at_0_50", "mean_matched_iou", "miss_count",
        "extra_count", "merge_candidate_count", "split_candidate_count",
    ],
    "proposal_bootstrap_intervals.csv": [
        "metric", "point_estimate", "ci_lower_95", "ci_upper_95",
        "bootstrap_repetitions", "asset_cluster_count", "seed",
    ],
    "mask_route_summary.csv": [
        "route", "reference_count", "prediction_count", "matched_reference_count",
        "missed_reference_count", "unmatched_prediction_count",
        "detection_aware_recall_at_0_75", "mean_matched_iou",
        "detection_aware_mean_iou", "mean_matched_dice",
        "detection_aware_mean_dice", "mean_matched_boundary_f1",
        "mean_matched_boundary_precision", "mean_matched_boundary_recall",
        "detection_aware_mean_boundary_f1", "detection_aware_mean_boundary_precision",
        "detection_aware_mean_boundary_recall", "mean_prediction_spill_fraction",
        "prediction_spill_rate", "mask_iou_threshold", "boundary_tolerance_pixels",
    ],
    "mask_subgroups.csv": [
        "dimension", "value", "route", "reference_count", "prediction_count",
        "matched_reference_count", "missed_reference_count", "unmatched_prediction_count",
        "detection_aware_recall_at_0_75", "mean_matched_iou",
        "detection_aware_mean_iou", "mean_matched_dice", "detection_aware_mean_dice",
        "mean_matched_boundary_f1", "mean_matched_boundary_precision",
        "mean_matched_boundary_recall", "detection_aware_mean_boundary_f1",
        "detection_aware_mean_boundary_precision", "detection_aware_mean_boundary_recall",
        "mean_prediction_spill_fraction", "prediction_spill_rate", "mask_iou_threshold",
        "boundary_tolerance_pixels",
    ],
    "mask_bootstrap_intervals.csv": [
        "metric", "direction", "scene_count", "cluster_count", "baseline_mean",
        "candidate_mean", "mean_difference", "confidence_level", "ci_lower",
        "ci_upper", "repetitions", "seed",
    ],
    "failure_summary.csv": ["failure_type", "count"],
    "decision_gates.csv": ["gate", "observed", "threshold", "status"],
    "subgroup_bootstrap_intervals.csv": [
        "component", "metric", "dimension", "value", "point_estimate",
        "ci_lower_95", "ci_upper_95", "bootstrap_repetitions",
        "cluster_count", "scene_count", "support_count", "support_unit", "minimum_clusters",
        "minimum_scenes", "minimum_support", "eligible",
        "declared_overall_threshold", "worst_subgroup_threshold",
    ],
    "worst_subgroup_summary.csv": [
        "component", "metric", "dimension", "value", "point_estimate",
        "ci_lower_95", "cluster_count", "scene_count", "support_count",
        "support_unit", "gate_statistic", "gate_observed", "threshold", "status",
    ],
    "diagnostic_summary.csv": [
        "diagnostic_kind", "route", "reference_count", "prediction_count",
        "miss_count", "extra_count", "detection_aware_mean_iou",
    ],
}
ALLOWED_JSON = {"runtime_summary.json", "run_record.json"}
ALLOWED_CHARTS = {
    "proposal_matched_iou_distribution.png",
    "proposal_precision_recall.png",
    "reference_box_mask_iou_distribution.png",
    "automatic_box_mask_iou_distribution.png",
    "mask_recall_by_route.png",
    "automatic_spill_vs_mask_iou.png",
}
DEFAULT_PROJECT = "Visual Peer Inspector/02 Detection and Segmentation"
TASK_NAME_PATTERN = re.compile(r"^ds1-proposal-mask-evaluation-kaggle-(candidate|final)$")
ALLOWED_TAGS = {
    "candidate", "final", "notebook-06.1", "ds1", "locateanything",
    "sam2.1", "evaluation", "kaggle", "full",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_|labels_by_object|mask_rle|box_xyxy|source_member|"
    r"evaluator_crosswalk|/kaggle/input/|/Users/|CLEARML_API_(ACCESS|SECRET)_KEY|HF_TOKEN",
    re.IGNORECASE,
)
DECISIONS = {
    "proceed_with_locateanything_and_sam21",
    "refine_or_replace_locateanything",
    "refine_or_replace_sam21",
    "retain_sam21_and_refine_automatic_proposals",
}
PINNED_CLEARML_VERSION = "2.1.10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", default="ds1-proposal-mask-evaluation-kaggle-candidate")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--secret-source", choices=("environment", "kaggle"), default="environment")
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_digest(rows: list[dict[str, Any]]) -> str:
    return sha256(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
    ).hexdigest()


def _is_digest(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def validate_safe_root(root: Path, *, require_clean_provenance: bool = False) -> dict[str, Any]:
    """Fail closed on private content, incomplete evidence and weak provenance."""

    expanded = root.expanduser()
    if expanded.is_symlink():
        raise ValueError("ClearML safe root cannot be a symbolic link")
    root = expanded.resolve(strict=True)
    expected = set(ALLOWED_TABLES) | ALLOWED_JSON | ALLOWED_CHARTS | {"artifact_manifest.json"}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("ClearML safe root cannot contain symbolic links")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if Path(relative).name != relative:
                raise ValueError("ClearML safe evidence must use a flat allowlist")
            actual.add(relative)
    if actual != expected:
        raise ValueError(f"ClearML safe allowlist differs; missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    for name, columns in ALLOWED_TABLES.items():
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML table: {name}")
        with (root / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise ValueError(f"Unexpected ClearML columns in {name}")
            if not list(reader):
                raise ValueError(f"ClearML aggregate table is empty: {name}")
    for name in ALLOWED_JSON:
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML JSON: {name}")
        json.loads(text)
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise ValueError("ClearML artifact manifest count is inconsistent")
    if [str(row.get("path")) for row in rows] != sorted(expected - {"artifact_manifest.json"}):
        raise ValueError("ClearML artifact manifest differs from the allowlist")
    if manifest.get("manifest_content_sha256") != _manifest_digest(rows):
        raise ValueError("ClearML artifact manifest content hash differs")
    for row in rows:
        path = root / str(row["path"])
        if sha256(path.read_bytes()).hexdigest() != row.get("sha256") or path.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError("ClearML artifact evidence differs from its manifest")

    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if record.get("stage") != "ds1_proposal_and_mask_evaluation" or record.get("status") != "completed":
        raise ValueError("ClearML requires a completed Notebook 06.1 evaluation")
    cohort = record.get("cohort", {})
    for key in ("cohort_id", "release_id", "role", "pilot_stage", "membership_sha256", "scene_count", "views_per_scene", "image_count", "asset_group_count"):
        if cohort.get(key) != getattr(DS1_SCREEN_CONTRACT, key):
            raise ValueError(f"ClearML DS1 cohort identity differs: {key}")
    coverage = record.get("coverage", {})
    if coverage.get("proposal_images") != DS1_SCREEN_CONTRACT.image_count or coverage.get("sam_images") != DS1_SCREEN_CONTRACT.image_count or int(coverage.get("decoded_sam_masks", 0)) <= 0:
        raise ValueError("ClearML requires complete DS1 proposal and SAM coverage")
    decision = record.get("component_decision", {})
    if decision.get("status") not in DECISIONS or decision.get("final_acceptance") is not False:
        raise ValueError("ClearML DS1 component decision is missing its claim boundary")
    provenance = record.get("evaluator_provenance", {})
    required_provenance = {
        "notebook_name", "execution_environment", "git_commit", "git_dirty",
        "precommit_candidate", "snapshot_sha256", "notebook_sha256",
        "code_dataset_version", "kaggle_notebook_version_06_1",
    }
    if set(provenance) != required_provenance:
        raise ValueError("ClearML evaluator provenance schema differs")
    if (
        not _is_digest(provenance["git_commit"], 40)
        or not _is_digest(provenance["snapshot_sha256"])
        or not _is_digest(provenance["notebook_sha256"])
    ):
        raise ValueError("ClearML evaluator provenance hashes are invalid")
    notebook_path = CODE_ROOT / NOTEBOOK_RELATIVE
    if sha256(notebook_path.read_bytes()).hexdigest() != provenance["notebook_sha256"]:
        raise ValueError("ClearML evaluator notebook identity differs")
    if not isinstance(provenance["git_dirty"], bool) or not isinstance(provenance["precommit_candidate"], bool):
        raise ValueError("ClearML evaluator source state is invalid")
    placeholders = {"", "pending", "unknown", "unavailable"}
    if any(
        str(provenance[key]).strip().lower() in placeholders
        for key in ("code_dataset_version", "kaggle_notebook_version_06_1")
    ):
        raise ValueError("ClearML evaluator version provenance is incomplete")
    if require_clean_provenance and (provenance["git_dirty"] or provenance["precommit_candidate"]):
        raise ValueError("Final ClearML evidence requires a clean code snapshot")
    inputs = record.get("inputs", {})
    if not inputs or any(not _is_digest(value) for value in inputs.values()):
        raise ValueError("ClearML evaluator input provenance is incomplete")
    return record


def _load_credentials(secret_source: str) -> None:
    """Load credentials only inside the isolated finaliser process."""

    configure_official_clearml_hosts()
    names = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    if secret_source == "kaggle":
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for name in names:
            value = client.get_secret(name)
            if not value:
                raise RuntimeError(f"Kaggle Secret {name} is missing or disabled")
            os.environ[name] = value
    elif any(not os.environ.get(name) for name in names):
        raise RuntimeError("ClearML credentials are missing from the finaliser environment")


def _pinned_clearml_is_importable() -> bool:
    try:
        installed_version = importlib.metadata.version("clearml")
    except importlib.metadata.PackageNotFoundError:
        return False
    if installed_version != PINNED_CLEARML_VERSION:
        return False
    try:
        importlib.import_module("clearml")
    except Exception:
        return False
    return True


def _install_tracking_requirement() -> bool:
    """Install the locked tracker only when the exact version is unavailable."""

    if _pinned_clearml_is_importable():
        return False
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(CODE_ROOT / "requirements/tracking-lock.txt"),
        ],
        timeout=120,
        check=True,
    )
    if not _pinned_clearml_is_importable():
        raise RuntimeError("The pinned ClearML version is unavailable after installation")
    return True


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _log(
    root: Path,
    project: str,
    task_name: str,
    tags: list[str],
    mode: str,
) -> tuple[str, str | None, dict[str, int]]:
    configure_official_clearml_hosts()
    from clearml import Task
    import pandas as pd

    if mode == "offline":
        Task.set_offline(True)
    _configure_clearml_request_timeouts()
    task, retry_outcome = initialize_clearml_task_with_retry(
        lambda: Task.init(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.testing,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
            auto_resource_monitoring=False,
            auto_connect_streams=False,
            reuse_last_task_id=False,
            output_uri=False,
        ),
        online=mode == "online",
    )
    task.add_tags(tags)
    logger = task.get_logger()
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    for metric, value in record["proposal_summary"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            logger.report_scalar("DS1 proposals", metric, float(value), iteration=0)
    for route in record["mask_route_summary"]:
        for metric in (
            "detection_aware_recall_at_0_75",
            "detection_aware_mean_iou",
            "detection_aware_mean_dice",
            "detection_aware_mean_boundary_f1",
        ):
            logger.report_scalar(f"DS1 masks {route['route']}", metric, float(route[metric]), iteration=0)
    for name in ALLOWED_TABLES:
        logger.report_table("DS1 proposal and mask evaluation", name.removesuffix(".csv").replace("_", " "), table_plot=pd.read_csv(root / name), iteration=0)
    for name in sorted(ALLOWED_CHARTS):
        logger.report_image("DS1 proposal and mask evaluation", name.removesuffix(".png").replace("_", " "), local_path=str(root / name), iteration=0)
    task.connect({**record["evaluator_provenance"], **record["inputs"], "component_decision": record["component_decision"]["status"]}, name="verified provenance")
    for name in sorted(ALLOWED_JSON):
        task.upload_artifact(name.removesuffix(".json").replace("_", " "), artifact_object=json.loads((root / name).read_text(encoding="utf-8")), wait_on_upload=False)
    task_id = str(task.id)
    offline = str(task.get_offline_mode_folder()) if mode == "offline" else None
    task.flush(wait_for_uploads=True)
    task.close()
    return task_id, offline, retry_outcome.to_receipt_fields()


def main() -> None:
    args = parse_args()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "notebook_06_1_clearml_finalisation",
        "tracking_mode": args.mode,
        "status": "pending",
        "clearml_task_id": None,
        "offline_session_folder": None,
        "project": args.project,
        "task_name": args.task_name,
        "clearml_hosts": dict(OFFICIAL_CLEARML_HOSTS),
        "tracking_attempt_count": 0,
        "tracking_retry_count": 0,
        "tracking_transient_failure_count": 0,
    }
    alarm_enabled = False
    previous_handler = None
    try:
        if args.project != DEFAULT_PROJECT or not TASK_NAME_PATTERN.fullmatch(args.task_name):
            raise ValueError("Notebook 06.1 ClearML project or task name is invalid")
        tags = args.tag or ["candidate", "notebook-06.1", "ds1", "locateanything", "sam2.1", "evaluation", "kaggle", "full"]
        if set(tags) - ALLOWED_TAGS:
            raise ValueError("Notebook 06.1 ClearML tags are invalid")
        final_task = args.task_name.endswith("-final") or "final" in tags
        root = args.safe_root.expanduser().resolve()
        record = validate_safe_root(root, require_clean_provenance=final_task)
        receipt.update({
            "safe_manifest_sha256": sha256((root / "artifact_manifest.json").read_bytes()).hexdigest(),
            "run_status": record["status"],
            "snapshot_sha256": record["evaluator_provenance"]["snapshot_sha256"],
            "component_decision": record["component_decision"]["status"],
        })
        if args.mode == "disabled":
            receipt["status"] = "disabled"
            return
        if not 30 <= args.wall_time_seconds <= 600:
            raise ValueError("ClearML wall time must be between 30 and 600 seconds")
        if hasattr(signal, "SIGALRM"):
            previous_handler = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(args.wall_time_seconds)
            alarm_enabled = True
        _install_tracking_requirement()
        _load_credentials(args.secret_source)
        task_id, offline, retry_fields = _log(
            root, args.project, args.task_name, tags, args.mode
        )
        receipt["clearml_task_id"] = task_id
        receipt["offline_session_folder"] = offline
        receipt.update(retry_fields)
        receipt["status"] = "completed" if args.mode == "online" else "offline_completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt.update(sanitized_tracking_failure_fields(error))
    finally:
        if alarm_enabled:
            signal.alarm(0)
            assert previous_handler is not None
            signal.signal(signal.SIGALRM, previous_handler)
        _write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
