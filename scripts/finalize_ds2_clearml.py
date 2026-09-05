#!/usr/bin/env python3
"""Publish only aggregate DS2 evidence after a notebook has finished."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
from typing import Any


ALLOWED_SUFFIXES = {".csv", ".json", ".png"}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_scene_id|labels_by_object|mask_rle|box_xyxy|"
    r"/kaggle/input/|/(?:Users|home)/|CLEARML_API_(ACCESS|SECRET)_KEY|HF_TOKEN|source_scene_id",
    re.IGNORECASE,
)
DEFAULT_PROJECT = "Visual Peer Inspector/04 Confirmation and Multi-view"


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
    """Verify the aggregate manifest and reject protected identifiers or paths."""

    root = root.expanduser().resolve(strict=True)
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("ClearML safe root cannot contain symbolic links")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    relative_files = {path.relative_to(root).as_posix() for path in files}
    if "artifact_manifest.json" not in relative_files or "run_record.json" not in relative_files:
        raise ValueError("ClearML safe root is missing its manifest or run record")
    if any(path.suffix.lower() not in ALLOWED_SUFFIXES for path in files):
        raise ValueError("ClearML safe root contains an unsupported file type")
    for path in files:
        relative = path.relative_to(root).as_posix().lower()
        if any(part in relative for part in ("private", "cache", "crosswalk", "credential", "secret")):
            raise ValueError("ClearML safe root contains a protected path")
        if path.suffix.lower() in {".csv", ".json"} and path.name != "artifact_manifest.json":
            if FORBIDDEN_TEXT.search(path.read_text(encoding="utf-8")):
                raise ValueError(f"Protected detail found in ClearML artifact: {relative}")

    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise ValueError("ClearML artifact manifest is invalid")
    declared = {str(row.get("path")) for row in rows}
    if declared != relative_files - {"artifact_manifest.json"}:
        raise ValueError("ClearML artifact manifest coverage differs")
    for row in rows:
        path = root / str(row["path"])
        if sha256(path.read_bytes()).hexdigest() != str(row["sha256"]) or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("ClearML artifact digest differs")
    run_record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if run_record.get("status") != "completed" or run_record.get("stage") != expected_stage:
        raise ValueError("ClearML run record is not a completed expected stage")
    return run_record


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _numeric_scalars(value: Any, prefix: str = "run", limit: int = 100) -> list[tuple[str, float]]:
    """Return a bounded list of numeric values from an already safe record."""

    rows: list[tuple[str, float]] = []

    def visit(item: Any, name: str) -> None:
        if len(rows) >= limit:
            return
        if isinstance(item, bool):
            rows.append((name, float(item)))
        elif isinstance(item, (int, float)):
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
        "safe_manifest_sha256": sha256((root / "artifact_manifest.json").read_bytes()).hexdigest(),
        "task_id": None,
        "run_status": record["status"],
        "tracking_attempt_count": 0,
        "tracking_retry_count": 0,
        "tracking_transient_failure_count": 0,
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
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        _write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
