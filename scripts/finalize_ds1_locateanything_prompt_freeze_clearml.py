#!/usr/bin/env python3
"""Upload only aggregate Notebook 04.3 evidence to ClearML."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any


ALLOWED_TABLES = {
    "completion_checks.csv": ["check", "passed"],
    "prompt_operational_summary.csv": [
        "prompt_key",
        "prompt",
        "source",
        "image_count",
        "successful_image_count",
        "failed_image_count",
        "mean_proposals_per_image",
        "median_runtime_seconds",
        "p95_runtime_seconds",
        "total_inference_seconds",
    ],
    "prompt_policy_summary.csv": [
        "candidate_key",
        "prompt_key",
        "prompt",
        "postprocessing_policy",
        "image_count",
        "reference_count",
        "proposal_count",
        "matched_count_at_0_50",
        "precision_at_0_50",
        "recall_at_0_50",
        "f1_at_0_50",
        "f2_at_0_50",
        "recall_at_0_75",
        "mean_matched_iou",
        "median_matched_iou",
        "p10_matched_iou",
        "detection_aware_mean_iou",
        "complete_detection_rate",
        "exact_count_rate",
        "full_image_success_rate",
        "count_mean_absolute_error",
        "count_signed_bias",
        "miss_count",
        "extra_count",
        "images_with_misses",
        "images_with_extras",
        "merge_candidate_count",
        "split_or_duplicate_candidate_count",
        "mean_proposals_per_image",
    ],
    "prompt_repeatability.csv": [
        "prompt_key",
        "prompt",
        "pilot_image_count",
        "exact_box_agreement_rate",
    ],
}
ALLOWED_JSON = {
    "artifact_manifest.json",
    "model_load_summary.json",
    "run_record.json",
    "selected_prompt.json",
    "study_protocol.json",
}
ALLOWED_CHARTS = {
    "complete_detection_rate.png",
    "detection_aware_mean_iou.png",
    "mean_proposals_by_prompt.png",
    "median_runtime_by_prompt.png",
    "missed_objects.png",
    "prompt_policy_f2.png",
    "split_or_duplicate_candidates.png",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_|labels_by_object|mask_rle|boxes_xyxy|box_xyxy|"
    r"raw_response|source_(path|scene|object|filename)|/kaggle/input/|/(?:Users|home)/|"
    r"CLEARML_API_(ACCESS|SECRET)_KEY|HF_TOKEN",
    re.IGNORECASE,
)
DEFAULT_PROJECT = "Visual Peer Inspector/02 Detection and Segmentation"
EXPECTED_STAGE = "ds1_locateanything_prompt_freeze"
PINNED_CLEARML_VERSION = "2.1.10"
CODE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", default="ds1-locateanything-prompt-freeze")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_safe_root(root: Path) -> dict[str, Any]:
    """Reject incomplete, private or unexpected tracking artifacts."""

    expanded = root.expanduser()
    if expanded.is_symlink():
        raise ValueError("ClearML safe root cannot be a symbolic link")
    root = expanded.resolve(strict=True)
    expected = set(ALLOWED_TABLES) | ALLOWED_JSON | ALLOWED_CHARTS
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("ClearML safe root cannot contain symbolic links")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError(
            f"ClearML allowlist differs; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    for relative, columns in ALLOWED_TABLES.items():
        observed_columns, rows = _read_csv(root / relative)
        if observed_columns != columns or not rows:
            raise ValueError(f"ClearML table schema or content differs: {relative}")
    for relative in sorted(set(ALLOWED_TABLES) | (ALLOWED_JSON - {"artifact_manifest.json"})):
        text = (root / relative).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML artifact: {relative}")
        if relative.endswith(".json"):
            json.loads(text)

    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise ValueError("ClearML artifact manifest count is invalid")
    if [str(row.get("path")) for row in rows] != sorted(expected - {"artifact_manifest.json"}):
        raise ValueError("ClearML artifact manifest differs from its allowlist")
    for row in rows:
        path = root / str(row["path"])
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("ClearML artifact checksum differs")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("ClearML artifact size differs")

    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if (
        record.get("stage") != EXPECTED_STAGE
        or record.get("status") != "completed"
        or record.get("all_completion_checks_passed") is not True
        or record.get("baseline_reused") is not True
        or record.get("model_load_count") not in {0, 1}
        or record.get("new_model_prompt_count") != 1
        or record.get("image_count") != 2_000
    ):
        raise ValueError("ClearML requires a complete DS1 prompt-freeze run")
    checks = _read_csv(root / "completion_checks.csv")[1]
    if any(row["passed"].strip().lower() not in {"true", "1"} for row in checks):
        raise ValueError("ClearML requires every prompt-freeze completion check")
    summary = _read_csv(root / "prompt_policy_summary.csv")[1]
    if len(summary) != 6 or {row["prompt_key"] for row in summary} != {
        "object",
        "individual_object",
    }:
        raise ValueError("ClearML prompt-policy summary is incomplete")
    selected = json.loads((root / "selected_prompt.json").read_text(encoding="utf-8"))
    selected_value = selected.get("selected")
    if (
        not isinstance(selected_value, dict)
        or selected_value.get("prompt_key") not in {
            "object",
            "individual_object",
        }
        or selected_value.get("postprocessing_policy") not in {
            "raw",
            "area_nms_075",
            "area_nms_075_cap8",
        }
        or selected.get("requires_unchanged_ds2_confirmation") is not True
    ):
        raise ValueError("The selected prompt must require unchanged DS2 confirmation")
    return record


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _prepare_tracking(mode: str) -> None:
    if mode not in {"online", "offline"}:
        return
    try:
        installed = importlib.metadata.version("clearml")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != PINNED_CLEARML_VERSION:
        subprocess.check_call(
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
        )
    os.environ.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    os.environ.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    os.environ.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")


def _log(root: Path, project: str, task_name: str, tags: list[str]) -> str:
    from clearml import Task

    task = Task.init(
        project_name=project,
        task_name=task_name,
        task_type=Task.TaskTypes.inference,
        reuse_last_task_id=False,
        output_uri=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks={"detect_repository": False},
        auto_connect_streams=False,
        auto_resource_monitoring=False,
    )
    if tags:
        task.add_tags(tags)
    selected = json.loads((root / "selected_prompt.json").read_text(encoding="utf-8"))
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    task.connect(
        {
            "stage": record["stage"],
            "candidate_lock_sha256": selected["candidate_lock_sha256"],
            "selected_candidate": record["selected_candidate"],
            "model_revision": selected["model_revision"],
        },
        name="verified selection",
    )
    logger = task.get_logger()
    for relative in sorted(ALLOWED_TABLES):
        logger.report_table(
            title=Path(relative).stem,
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
    receipt = {
        "schema_version": 1,
        "stage": "notebook_04_3_clearml_finalisation",
        "mode": args.mode,
        "status": "pending",
        "safe_manifest_sha256": None,
        "task_id": None,
    }
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, args.wall_time_seconds))
    try:
        validate_safe_root(root)
        receipt["safe_manifest_sha256"] = sha256(
            (root / "artifact_manifest.json").read_bytes()
        ).hexdigest()
        if args.mode == "disabled":
            receipt["status"] = "disabled"
            return
        _prepare_tracking(args.mode)
        if args.mode == "offline":
            os.environ["CLEARML_OFFLINE_MODE"] = "1"
        receipt["task_id"] = _log(root, args.project, args.task_name, args.tag)
        receipt["status"] = "completed" if args.mode == "online" else "offline_completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        receipt["error_message"] = str(error)[:300]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        _write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
