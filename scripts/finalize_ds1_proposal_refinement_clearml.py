#!/usr/bin/env python3
"""Upload only aggregate Notebook 04.2 evidence after computation finishes."""

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
    "policy_summary.csv": [
        "policy", "proposal_count", "precision_at_0_50", "recall_at_0_50",
        "f1_at_0_50", "recall_at_0_75", "exact_count_rate",
        "full_image_success_rate", "miss_count", "extra_count",
    ],
    "paired_f1_intervals.csv": [
        "policy", "f1_difference", "ci_lower_95", "ci_upper_95",
        "asset_cluster_count", "repetitions", "seed",
    ],
    "policy_gates.csv": [
        "policy", "precision_gate_passed", "recall_gate_passed",
        "recall_loss_vs_baseline", "paired_f1_ci_lower",
        "diagnostic_selection_eligible",
    ],
    "removal_summary.csv": ["policy", "reason", "removed_count"],
    "subgroup_summary.csv": [
        "policy", "dimension", "value", "image_count", "reference_count",
        "precision_at_0_50", "recall_at_0_50", "f1_at_0_50",
    ],
    "completion_gates.csv": ["gate", "passed"],
}
ALLOWED_JSON = {
    "resolved_configuration.json", "decision.json", "input_audit.json",
    "run_record.json", "artifact_manifest.json",
}
ALLOWED_CHARTS = {
    "figures/policy_metrics.png", "figures/misses_and_extras.png",
    "figures/proposal_count_vs_f1.png",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_|labels_by_object|mask_rle|box_xyxy|"
    r"/kaggle/input/|/Users/|HF_TOKEN|CLEARML_API_(ACCESS|SECRET)_KEY",
    re.IGNORECASE,
)
DEFAULT_PROJECT = "Visual Peer Inspector/02 Detection and Segmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", default="ds1-proposal-refinement-kaggle-candidate")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_safe_root(root: Path) -> dict[str, Any]:
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
            f"ClearML safe allowlist differs; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    for name, columns in ALLOWED_TABLES.items():
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML table: {name}")
        with (root / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns or not list(reader):
                raise ValueError(f"ClearML table schema or content differs: {name}")
    for name in ALLOWED_JSON - {"artifact_manifest.json"}:
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML JSON: {name}")
        json.loads(text)
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise ValueError("ClearML artifact manifest count is invalid")
    if [str(row.get("path")) for row in rows] != sorted(expected - {"artifact_manifest.json"}):
        raise ValueError("ClearML artifact manifest differs from its allowlist")
    for row in rows:
        path = root / str(row["path"])
        if sha256(path.read_bytes()).hexdigest() != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("ClearML artifact checksum differs")
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if record.get("stage") != "ds1_proposal_refinement" or record.get("status") != "completed":
        raise ValueError("ClearML requires a completed Notebook 04.2 run")
    if not record.get("all_completion_gates_passed"):
        raise ValueError("ClearML requires all Notebook 04.2 completion gates")
    return record


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _log_online(root: Path, project: str, task_name: str, tags: list[str]) -> str:
    from clearml import Task

    task = Task.init(project_name=project, task_name=task_name, reuse_last_task_id=False, output_uri=False)
    if tags:
        task.add_tags(tags)
    logger = task.get_logger()
    for relative in sorted(ALLOWED_TABLES):
        logger.report_table(title=relative.removesuffix(".csv"), series="aggregate", iteration=0, csv=str(root / relative))
        task.upload_artifact(relative, artifact_object=root / relative, wait_on_upload=False)
    for relative in sorted(ALLOWED_CHARTS):
        logger.report_image(title=Path(relative).stem, series="aggregate", iteration=0, local_path=str(root / relative))
    for relative in sorted(ALLOWED_JSON - {"artifact_manifest.json"}):
        task.upload_artifact(relative, artifact_object=root / relative, wait_on_upload=False)
    task.flush(wait_for_uploads=True)
    task.close()
    return str(task.id)


def main() -> None:
    args = parse_args()
    root = args.safe_root.expanduser().resolve()
    record = validate_safe_root(root)
    receipt = {
        "schema_version": 1,
        "stage": "notebook_04_2_clearml_finalisation",
        "mode": args.mode,
        "status": "pending",
        "safe_manifest_sha256": sha256((root / "artifact_manifest.json").read_bytes()).hexdigest(),
        "task_id": None,
        "run_status": record["status"],
    }
    if args.mode == "disabled":
        receipt["status"] = "disabled"
        _write_receipt(args.receipt, receipt)
        return
    if args.mode == "offline":
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(args.wall_time_seconds)
    try:
        receipt["task_id"] = _log_online(root, args.project, args.task_name, args.tag)
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
