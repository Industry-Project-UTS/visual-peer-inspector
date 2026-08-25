#!/usr/bin/env python3
"""Upload only allowlisted aggregate Notebook 04.1 evidence to ClearML."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any


ALLOWED_TABLES = {
    "attempt_summary.csv": ["attempt_count", "image_count"],
    "completion_gates.csv": ["gate", "status", "observed", "expected"],
    "failure_summary.csv": ["error_type", "count"],
    "progress_checkpoints.csv": [
        "completed",
        "total",
        "successful",
        "failed",
        "elapsed_seconds",
    ],
    "proposal_summary_by_view.csv": [
        "view_id",
        "image_count",
        "successful_count",
        "failed_count",
        "mean_proposal_count",
        "mean_runtime_seconds",
        "truncated_count",
    ],
}
ALLOWED_JSON = {
    "artifact_manifest.json",
    "box_geometry_summary.json",
    "environment_summary.json",
    "input_audit.json",
    "model_load_summary.json",
    "one_image_preflight.json",
    "pilot_summary.json",
    "proposal_summary.json",
    "repeat_agreement.json",
    "resolved_configuration.json",
    "run_record.json",
    "runtime_summary.json",
}
ALLOWED_CHARTS = {
    "box_area_fraction_distribution.png",
    "box_aspect_ratio_distribution.png",
    "mean_proposals_by_view.png",
    "proposal_count_distribution.png",
    "runtime_by_proposal_count.png",
    "runtime_distribution.png",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_scene|opaque_object|category|anomaly|"
    r"raw_response|boxes_xyxy|box_xyxy|source_(path|scene|object|filename)|"
    r"raw_path|/kaggle/input/|/Users/|"
    r"CLEARML_API_(ACCESS|SECRET)_KEY|HF_TOKEN",
    re.IGNORECASE,
)
DEFAULT_PROJECT = "Visual Peer Inspector/02 Detection and Segmentation"
DEFAULT_TASK_NAME = "ds1-locateanything-screen-kaggle-candidate"
EXPECTED_COHORT_ID = "toysad8k-development-screen-100-v1"
EXPECTED_MEMBERSHIP_SHA256 = (
    "308e2e104448612dc60b5392a5258d97f6d93f52ca553aa68fa1b447500f4f2f"
)
EXPECTED_MODEL_ID = "nvidia/LocateAnything-3B"
EXPECTED_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.tracking import (
    OFFICIAL_CLEARML_HOSTS,
    _configure_clearml_request_timeouts,
    configure_official_clearml_hosts,
    initialize_clearml_task_with_retry,
    sanitized_tracking_failure_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_safe_root(
    root: Path, *, require_full_model: bool = True
) -> dict[str, Any]:
    """Reject unexpected files, raw outputs and incomplete model evidence."""

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
    if [str(row.get("path")) for row in rows] != sorted(expected - {"artifact_manifest.json"}):
        raise ValueError("ClearML artifact manifest differs from its allowlist")
    for row in rows:
        path = root / str(row["path"])
        if (
            sha256(path.read_bytes()).hexdigest() != row["sha256"]
            or path.stat().st_size != int(row["size_bytes"])
        ):
            raise ValueError("ClearML artifact checksum differs")
    expected_manifest_content = sha256(
        b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            + b"\n"
            for row in rows
        )
    ).hexdigest()
    if manifest.get("manifest_content_sha256") != expected_manifest_content:
        raise ValueError("ClearML artifact manifest content hash differs")

    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if record.get("stage") != "ds1_locateanything_screen":
        raise ValueError("ClearML requires DS1 LocateAnything evidence")
    if require_full_model and (
        record.get("status") != "completed"
        or record.get("run_mode") != "model_inference"
        or record.get("inference_stage") != "full"
        or record.get("completion_gate_passed") is not True
        or record.get("producer_kind") != "locateanything_model"
    ):
        raise ValueError("ClearML requires a completed full DS1 model run")
    cohort = record.get("cohort", {})
    if (
        cohort.get("cohort_id") != EXPECTED_COHORT_ID
        or cohort.get("release_id") != "toysad8k-train-2000-v2"
        or cohort.get("role") != "development"
        or cohort.get("pilot_stage") != "screen_100"
        or cohort.get("membership_sha256") != EXPECTED_MEMBERSHIP_SHA256
        or cohort.get("scene_count") != 100
        or cohort.get("views_per_scene") != 20
        or cohort.get("image_count") != 2_000
        or cohort.get("asset_group_count") != 27
    ):
        raise ValueError("ClearML run record differs from the frozen DS1 cohort")
    configuration = record.get("configuration", {})
    if (
        configuration.get("model_id") != EXPECTED_MODEL_ID
        or configuration.get("model_revision") != EXPECTED_MODEL_REVISION
        or configuration.get("prompt") != "object"
        or configuration.get("seed") != 32_040
    ):
        raise ValueError("ClearML run record differs from the frozen DS1 settings")
    if require_full_model:
        summary = record.get("summary", {})
        if (
            summary.get("job_count") != 2_000
            or summary.get("successful_job_count") != 2_000
            or summary.get("failed_job_count") != 0
            or summary.get("possibly_truncated_count") != 0
            or summary.get("zero_proposal_image_count") != 0
        ):
            raise ValueError("ClearML requires complete successful DS1 proposal coverage")
        output_identity = record.get("output_identity", {})
        proposal_results_sha256 = str(
            output_identity.get("proposal_results_sha256", "")
        )
        if (
            output_identity.get("result_filename") != "proposal_results.jsonl"
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(output_identity.get("result_sha256", ""))
            )
            or not re.fullmatch(r"[0-9a-f]{64}", proposal_results_sha256)
            or proposal_results_sha256 != output_identity.get("result_sha256")
            or output_identity.get("result_count") != 2_000
        ):
            raise ValueError("ClearML requires committed DS1 proposal-result identity")
        preflight = record.get("one_image_preflight", {})
        if (
            preflight.get("job_count") != 1
            or preflight.get("successful_job_count") != 1
            or preflight.get("failed_job_count") != 0
        ):
            raise ValueError("ClearML requires a passed one-image preflight")
        pilot = record.get("pilot_summary", {})
        if (
            pilot.get("job_count") != 20
            or pilot.get("successful_job_count") != 20
            or pilot.get("failed_job_count") != 0
            or pilot.get("scene_count") != 20
        ):
            raise ValueError("ClearML requires a passed 20-image pilot")
        repeat = record.get("repeat_agreement", {})
        if (
            repeat.get("job_count") != 20
            or repeat.get("matching_job_count") != 20
            or repeat.get("box_agreement_rate") != 1.0
        ):
            raise ValueError("ClearML requires exact pilot repeatability")
        gate_rows = _csv_rows(root / "completion_gates.csv")
        if not gate_rows or any(row.get("status") != "pass" for row in gate_rows):
            raise ValueError("ClearML completion gates are not all passed")
        attempt_rows = _csv_rows(root / "attempt_summary.csv")
        if not attempt_rows or sum(int(row["image_count"]) for row in attempt_rows) != 2_000:
            raise ValueError("ClearML retry evidence does not cover all DS1 images")
        provenance = record.get("provenance", {})
        if (
            provenance.get("git_dirty") is not False
            or provenance.get("precommit_candidate") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("git_commit", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("snapshot_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("notebook_sha256", "")))
        ):
            raise ValueError("ClearML requires clean canonical provenance")
        model_load = json.loads((root / "model_load_summary.json").read_text(encoding="utf-8"))
        checkpoints = model_load.get("checkpoints", [])
        if not any(row.get("stage") == "model_ready" for row in checkpoints):
            raise ValueError("ClearML requires a completed LocateAnything model load")
        disposable = model_load.get("disposable_preflight", {})
        if disposable.get("status") != "passed":
            raise ValueError("ClearML requires the disposable one-image GPU gate")
        environment = json.loads((root / "environment_summary.json").read_text(encoding="utf-8"))
        versions = environment.get("platform_versions", {})
        required_versions = {
            "python",
            "pillow",
            "numpy",
            "transformers",
            "torch",
            "torchvision",
            "cuda_runtime",
            "cudnn",
            "kaggle_container_image",
            "kaggle_kernel_run_type",
        }
        if not required_versions.issubset(versions):
            raise ValueError("ClearML environment evidence is incomplete")
    return record


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _prepare_tracking_environment(mode: str) -> None:
    """Install tracking support and resolve Kaggle credentials in this child."""

    configure_official_clearml_hosts()
    if mode in {"online", "offline"}:
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
    if mode == "online" and Path("/kaggle/input").is_dir():
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for name in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"):
            value = client.get_secret(name)
            if not value:
                raise RuntimeError(f"Kaggle Secret {name} is missing or disabled")
            os.environ[name] = value
def _log_online(
    root: Path,
    project: str,
    task_name: str,
    tags: list[str],
    *,
    online: bool,
) -> tuple[str, dict[str, int]]:
    configure_official_clearml_hosts()
    from clearml import Task

    _configure_clearml_request_timeouts()
    task, retry_outcome = initialize_clearml_task_with_retry(
        lambda: Task.init(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.inference,
            reuse_last_task_id=False,
            output_uri=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks={"detect_repository": False},
            auto_connect_streams=False,
            auto_resource_monitoring=False,
        ),
        online=online,
    )
    if tags:
        task.add_tags(tags)
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    task.connect(
        {
            "cohort_id": record["cohort"]["cohort_id"],
            "membership_sha256": record["cohort"]["membership_sha256"],
            "configuration_sha256": record["configuration_sha256"],
            "proposal_results_sha256": record["output_identity"][
                "proposal_results_sha256"
            ],
            "model_revision": record["configuration"]["model_revision"],
        },
        name="verified provenance",
    )
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
    return str(task.id), retry_outcome.to_receipt_fields()


def main() -> None:
    args = parse_args()
    root = args.safe_root.expanduser().resolve()
    receipt = {
        "schema_version": 1,
        "stage": "notebook_04_1_clearml_finalisation",
        "mode": args.mode,
        "status": "pending",
        "safe_manifest_sha256": None,
        "task_id": None,
        "run_status": None,
        "clearml_hosts": dict(OFFICIAL_CLEARML_HOSTS),
        "tracking_attempt_count": 0,
        "tracking_retry_count": 0,
        "tracking_transient_failure_count": 0,
    }
    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, args.wall_time_seconds))
    try:
        record = validate_safe_root(root, require_full_model=args.mode != "disabled")
        receipt["safe_manifest_sha256"] = sha256(
            (root / "artifact_manifest.json").read_bytes()
        ).hexdigest()
        receipt["run_status"] = record["status"]
        if args.mode == "disabled":
            receipt["status"] = "disabled"
            return
        _prepare_tracking_environment(args.mode)
        if args.mode == "offline":
            os.environ["CLEARML_OFFLINE_MODE"] = "1"
        task_id, retry_fields = _log_online(
            root,
            args.project,
            args.task_name,
            args.tag,
            online=args.mode == "online",
        )
        receipt["task_id"] = task_id
        receipt.update(retry_fields)
        receipt["status"] = "completed" if args.mode == "online" else "offline_completed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt.update(sanitized_tracking_failure_fields(error))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        _write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
