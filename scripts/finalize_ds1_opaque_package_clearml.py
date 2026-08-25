#!/usr/bin/env python3
"""Validate and upload only allowlisted Notebook 03.1 evidence."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any


ALLOWED_TABLES = {
    "validation_gates.csv": ["gate", "status", "observed", "expected"],
    "view_coverage.csv": [
        "view_id",
        "scene_count",
        "expected_scene_count",
        "complete",
    ],
}
ALLOWED_JSON = {
    "artifact_manifest.json",
    "media_summary.json",
    "run_record.json",
}
ALLOWED_CHARTS = {
    "image_size_distribution.png",
    "view_coverage.png",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_scene|opaque_object|category|anomaly|"
    r"source_(path|scene|object|filename)|raw_path|/kaggle/input/|/Users/|"
    r"CLEARML_API_(ACCESS|SECRET)_KEY|HF_TOKEN",
    re.IGNORECASE,
)
DEFAULT_PROJECT = "Visual Peer Inspector/01 Data Preparation"
DEFAULT_TASK_NAME = "ds1-opaque-package-kaggle-candidate"
EXPECTED_COHORT_ID = "toysad8k-development-screen-100-v1"
EXPECTED_RELEASE_ID = "toysad8k-train-2000-v2"
EXPECTED_MEMBERSHIP_SHA256 = (
    "308e2e104448612dc60b5392a5258d97f6d93f52ca553aa68fa1b447500f4f2f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def validate_safe_root(
    root: Path,
    *,
    require_clean_snapshot: bool = True,
) -> dict[str, Any]:
    """Reject unexpected, private, corrupted, or incomplete evidence."""

    expanded = root.expanduser()
    if expanded.is_symlink():
        raise ValueError("ClearML safe root is missing or unsafe")
    root = expanded.resolve(strict=True)
    expected = set(ALLOWED_TABLES) | ALLOWED_JSON | ALLOWED_CHARTS
    actual: set[str] = set()
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
    for name, columns in ALLOWED_TABLES.items():
        if _csv_columns(root / name) != columns:
            raise ValueError(f"ClearML table columns differ: {name}")
    for relative in sorted(set(ALLOWED_TABLES) | (ALLOWED_JSON - {"artifact_manifest.json"})):
        text = (root / relative).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML artifact: {relative}")

    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise ValueError("ClearML artifact manifest count is invalid")
    expected_paths = sorted(expected - {"artifact_manifest.json"})
    if [str(row.get("path")) for row in rows] != expected_paths:
        raise ValueError("ClearML artifact manifest differs from its allowlist")
    for row in rows:
        path = root / str(row["path"])
        if (
            sha256(path.read_bytes()).hexdigest() != row.get("sha256")
            or path.stat().st_size != int(row.get("size_bytes", -1))
        ):
            raise ValueError("ClearML artifact checksum differs")

    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if record.get("stage") != "ds1_opaque_package" or record.get("status") != "completed":
        raise ValueError("ClearML requires a completed DS1 opaque-package run")
    cohort = record.get("cohort_summary", {})
    package = record.get("package", {})
    if (
        cohort.get("cohort_id") != EXPECTED_COHORT_ID
        or cohort.get("membership_sha256") != EXPECTED_MEMBERSHIP_SHA256
        or cohort.get("scene_count") != 100
        or cohort.get("asset_group_count") != 27
        or package.get("release_id") != EXPECTED_RELEASE_ID
        or package.get("image_count") != 2_000
        or package.get("view_count") != 20
        or package.get("verified_image_count") != 2_000
        or package.get("all_checks_passed") is not True
    ):
        raise ValueError("ClearML run record differs from the frozen DS1 package")
    provenance = record.get("provenance", {})
    if require_clean_snapshot and (
        provenance.get("git_dirty") is not False
        or provenance.get("precommit_candidate") is True
        or provenance.get("snapshot_sha256") in {None, "", "local-working-tree"}
    ):
        raise ValueError("ClearML requires a clean committed code snapshot")
    return record


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _load_kaggle_credentials() -> None:
    required = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    if all(os.getenv(name) for name in required):
        return
    if not Path("/kaggle").is_dir():
        raise RuntimeError("ClearML credentials are unavailable")
    from kaggle_secrets import UserSecretsClient

    secret_client = UserSecretsClient()
    for name in required:
        value = secret_client.get_secret(name)
        if not value:
            raise RuntimeError(f"Kaggle Secret {name} is empty or disabled")
        os.environ[name] = value
    os.environ.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    os.environ.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    os.environ.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")


def _ensure_clearml() -> None:
    if importlib.util.find_spec("clearml") is not None:
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "clearml==2.1.10"],
        check=True,
        timeout=120,
    )


def _log_online(root: Path, project: str, task_name: str, tags: list[str]) -> str:
    from clearml import Task

    task = Task.init(
        project_name=project,
        task_name=task_name,
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,
        output_uri=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks={"detect_repository": False},
        auto_connect_streams=False,
        auto_resource_monitoring=False,
    )
    if tags:
        task.add_tags(tags)
    logger = task.get_logger()
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    media = json.loads((root / "media_summary.json").read_text(encoding="utf-8"))
    for name, value in {
        "scenes": record["package"]["scene_count"],
        "images": record["package"]["image_count"],
        "decoded failures": record["package"]["decoded_image_failure_count"],
        "total media MiB": media["total_media_mib"],
        "fresh pass seconds": record["resume_summary"]["first_elapsed_seconds"],
        "resume pass seconds": record["resume_summary"]["second_elapsed_seconds"],
    }.items():
        logger.report_scalar(
            title="DS1 opaque package",
            series=name,
            value=float(value),
            iteration=0,
        )
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


def main() -> int:
    args = parse_args()
    root = args.safe_root.expanduser().resolve()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "notebook_03_1_clearml_finalisation",
        "mode": args.mode,
        "status": "pending",
        "safe_manifest_sha256": None,
        "task_id": None,
    }
    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, args.wall_time_seconds))
    try:
        record = validate_safe_root(
            root,
            require_clean_snapshot=args.mode != "disabled",
        )
        receipt["safe_manifest_sha256"] = sha256(
            (root / "artifact_manifest.json").read_bytes()
        ).hexdigest()
        receipt["run_status"] = record["status"]
        if args.mode == "disabled":
            receipt["status"] = "disabled"
        else:
            if args.mode == "offline":
                os.environ["CLEARML_OFFLINE_MODE"] = "1"
            else:
                _load_kaggle_credentials()
            _ensure_clearml()
            receipt["task_id"] = _log_online(
                root,
                args.project,
                args.task_name,
                args.tag,
            )
            receipt["status"] = (
                "completed" if args.mode == "online" else "offline_completed"
            )
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        receipt["error_message"] = str(error)[:300]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        _write_receipt(args.receipt, receipt)
    return 0 if receipt["status"] in {"completed", "offline_completed", "disabled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
