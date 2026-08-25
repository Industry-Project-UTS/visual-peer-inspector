#!/usr/bin/env python3
"""Publish only allowlisted aggregate DS1 SAM evidence to ClearML."""

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

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT


ALLOWED_TABLES = {
    "completion_gates.csv": ["gate", "observed", "expected", "status"],
    "route_summary.csv": [
        "route",
        "mask_count",
        "empty_mask_count",
        "empty_mask_rate",
        "mean_area_fraction",
        "median_area_fraction",
        "p95_area_fraction",
        "mean_predicted_quality_score",
        "p10_predicted_quality_score",
    ],
    "view_summary.csv": [
        "view_id",
        "route",
        "mask_count",
        "empty_mask_count",
        "mean_area_fraction",
        "mean_predicted_quality_score",
    ],
    "attempt_summary.csv": ["status", "attempt", "count"],
    "failure_summary.csv": ["error_type", "count"],
}
ALLOWED_JSON = {
    "preflight_gates.json",
    "repeat_agreement.json",
    "runtime_summary.json",
    "mask_geometry_summary.json",
    "run_record.json",
}
ALLOWED_CHARTS = {
    "runtime_distribution.png",
    "mask_count_distribution.png",
    "runtime_by_prompt_count.png",
    "reference_box_mask_area_distribution.png",
    "automatic_box_mask_area_distribution.png",
    "reference_box_predicted_score_distribution.png",
    "automatic_box_predicted_score_distribution.png",
    "mask_count_by_route.png",
}
DEFAULT_PROJECT = "Visual Peer Inspector/02 Detection and Segmentation"
TASK_NAME_PATTERN = re.compile(r"^ds1-sam21-[a-z0-9-]+$")
ALLOWED_TAGS = {
    "candidate",
    "final",
    "notebook-05.1",
    "ds1",
    "sam2.1",
    "kaggle",
    "full",
}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|opaque_scene|opaque_asset|labels_by_object|mask_rle|box_xyxy|"
    r"/kaggle/input/|/Users/|CLEARML_API_(ACCESS|SECRET)_KEY|HF_TOKEN",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--task-name", default="ds1-sam21-kaggle-candidate")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument(
        "--credentials-source",
        choices=("environment", "kaggle", "none"),
        default="environment",
    )
    parser.add_argument("--wall-time-seconds", type=int, default=180)
    return parser.parse_args()


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _manifest_digest(rows: list[dict[str, Any]]) -> str:
    return sha256(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for row in rows
        )
    ).hexdigest()


def validate_safe_root(root: Path) -> dict[str, Any]:
    """Reject private content, unexpected files and incomplete DS1 evidence."""

    expanded = root.expanduser()
    if expanded.is_symlink():
        raise ValueError("ClearML safe root cannot be a symbolic link")
    root = expanded.resolve(strict=True)
    expected = set(ALLOWED_TABLES) | ALLOWED_JSON | ALLOWED_CHARTS | {
        "artifact_manifest.json"
    }
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
    for name, expected_columns in ALLOWED_TABLES.items():
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML table: {name}")
        with (root / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise ValueError(f"Unexpected ClearML columns in {name}")
            list(reader)
    for name in ALLOWED_JSON:
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML JSON: {name}")
        json.loads(text)
    manifest = json.loads(
        (root / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise ValueError("ClearML artifact manifest count is inconsistent")
    listed = [str(row.get("path")) for row in rows]
    if listed != sorted(expected - {"artifact_manifest.json"}):
        raise ValueError("ClearML artifact manifest differs from the allowlist")
    if manifest.get("manifest_content_sha256") != _manifest_digest(rows):
        raise ValueError("ClearML artifact manifest content hash differs")
    for row in rows:
        path = root / str(row["path"])
        if (
            sha256(path.read_bytes()).hexdigest() != row.get("sha256")
            or path.stat().st_size != int(row.get("size_bytes", -1))
        ):
            raise ValueError("ClearML artifact evidence differs")
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    required = {
        "stage": "ds1_sam21_prompted_masks",
        "status": "completed",
        "run_mode": "model_inference",
        "inference_stage": "full",
        "producer_kind": "sam2.1_model",
        "completion_gate_passed": True,
    }
    for key, expected_value in required.items():
        if record.get(key) != expected_value:
            raise ValueError(f"ClearML requires a completed DS1 model run: {key}")
    provenance = record.get("provenance", {})
    if (
        provenance.get("git_dirty") is not False
        or provenance.get("precommit_candidate") is not False
        or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("git_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("snapshot_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("notebook_sha256", "")))
    ):
        raise ValueError("ClearML requires clean committed Notebook 05.1 provenance")
    cohort = record.get("cohort", {})
    for key in (
        "cohort_id",
        "release_id",
        "role",
        "pilot_stage",
        "membership_sha256",
        "scene_count",
        "views_per_scene",
        "image_count",
        "asset_group_count",
    ):
        if cohort.get(key) != getattr(DS1_SCREEN_CONTRACT, key):
            raise ValueError(f"ClearML DS1 cohort identity differs: {key}")
    summary = record.get("summary", {})
    if (
        summary.get("image_count") != DS1_SCREEN_CONTRACT.image_count
        or summary.get("scene_count") != DS1_SCREEN_CONTRACT.scene_count
        or summary.get("mask_count")
        != int(record.get("reference_prompt_count", -1))
        + int(record.get("automatic_prompt_count", -1))
    ):
        raise ValueError("ClearML requires complete DS1 image and mask coverage")
    configuration_sha256 = str(record.get("configuration_sha256", ""))
    prompt_package_sha256 = str(record.get("prompt_package_sha256", ""))
    checkpoint_sha256 = str(record.get("checkpoint_sha256", ""))
    input_identity = record.get("input_identity", {})
    if (
        not re.fullmatch(r"[0-9a-f]{64}", configuration_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", prompt_package_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256)
        or checkpoint_sha256
        != str(record.get("configuration", {}).get("checkpoint_sha256", ""))
        or input_identity.get("prompt_package_record_sha256")
        != prompt_package_sha256
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(input_identity.get("prompt_manifest_sha256", "")),
        )
    ):
        raise ValueError("ClearML requires committed DS1 SAM input identity")
    output_identity = record.get("output_identity", {})
    if (
        output_identity.get("mask_results_filename") != "mask_results.jsonl"
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(output_identity.get("mask_results_sha256", ""))
        )
        or output_identity.get("mask_result_count") != DS1_SCREEN_CONTRACT.image_count
        or output_identity.get("repeat_mask_results_filename")
        != "repeat_mask_results.jsonl"
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(output_identity.get("repeat_mask_results_sha256", "")),
        )
        or output_identity.get("repeat_mask_result_count") != 20
    ):
        raise ValueError("ClearML requires committed DS1 SAM result identity")
    return record


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("ClearML finalisation exceeded its wall-time limit")


def _load_credentials(source: str, mode: str) -> None:
    """Load credentials only inside the bounded finaliser process."""

    if mode != "online" or source == "none":
        return
    if source == "kaggle":
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for name in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"):
            value = client.get_secret(name)
            if not value:
                raise RuntimeError(f"Kaggle Secret {name} is missing or disabled")
            os.environ[name] = value
    missing = [
        name
        for name in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(f"ClearML credentials are missing: {missing}")
    os.environ.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    os.environ.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    os.environ.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    os.environ.setdefault("CLEARML_LOG_ENVIRONMENT", "")


def _log_online(
    root: Path, project: str, task_name: str, tags: list[str], mode: str
) -> tuple[str, str | None]:
    from clearml import Task
    import pandas as pd

    from visual_peer_inspector.tracking import _configure_clearml_request_timeouts

    if mode == "offline":
        Task.set_offline(True)
    _configure_clearml_request_timeouts()
    task = Task.init(
        project_name=project,
        task_name=task_name,
        task_type=Task.TaskTypes.inference,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_resource_monitoring=False,
        auto_connect_streams=False,
        reuse_last_task_id=False,
        output_uri=False,
    )
    task.add_tags(tags)
    logger = task.get_logger()
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    summary = record["summary"]
    for series, value in {
        "completed images": summary["image_count"],
        "completed masks": summary["mask_count"],
        "empty masks": summary["empty_mask_count"],
        "median seconds per image": summary["median_image_runtime_seconds"],
        "p95 seconds per image": summary["p95_image_runtime_seconds"],
        "reference prompt count": record["reference_prompt_count"],
        "automatic prompt count": record["automatic_prompt_count"],
        "mask repeat agreement": record["repeat_agreement"][
            "mask_digest_agreement_rate"
        ],
        "score repeat agreement": record["repeat_agreement"][
            "score_agreement_rate"
        ],
    }.items():
        logger.report_scalar("DS1 SAM 2.1", series, float(value), iteration=0)
    for name in ALLOWED_TABLES:
        logger.report_table(
            "DS1 SAM 2.1",
            name.removesuffix(".csv").replace("_", " "),
            table_plot=pd.read_csv(root / name),
            iteration=0,
        )
    for name in sorted(ALLOWED_CHARTS):
        logger.report_image(
            "DS1 SAM 2.1",
            name.removesuffix(".png").replace("_", " "),
            local_path=str(root / name),
            iteration=0,
        )
    task.connect(
        {
            "cohort_id": record["cohort"]["cohort_id"],
            "membership_sha256": record["cohort"]["membership_sha256"],
            "configuration_sha256": record["configuration_sha256"],
            "prompt_package_sha256": record["prompt_package_sha256"],
            "prompt_manifest_sha256": record["input_identity"][
                "prompt_manifest_sha256"
            ],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "mask_results_sha256": record["output_identity"][
                "mask_results_sha256"
            ],
            "repeat_mask_results_sha256": record["output_identity"][
                "repeat_mask_results_sha256"
            ],
            "code_revision": record["configuration"]["code_revision"],
            "model_revision": record["configuration"]["model_revision"],
        },
        name="verified provenance",
    )
    for name in sorted(ALLOWED_JSON):
        task.upload_artifact(
            name.removesuffix(".json").replace("_", " "),
            artifact_object=json.loads((root / name).read_text(encoding="utf-8")),
            wait_on_upload=False,
        )
    task_id = str(task.id)
    offline = str(task.get_offline_mode_folder()) if mode == "offline" else None
    task.flush(wait_for_uploads=True)
    task.close()
    return task_id, offline


def main() -> None:
    args = parse_args()
    if args.project != DEFAULT_PROJECT:
        raise ValueError("Notebook 05.1 ClearML project name is fixed")
    if not TASK_NAME_PATTERN.fullmatch(args.task_name):
        raise ValueError("Notebook 05.1 ClearML task name is invalid")
    tags = args.tag or [
        "candidate",
        "notebook-05.1",
        "ds1",
        "sam2.1",
        "kaggle",
        "full",
    ]
    if set(tags) - ALLOWED_TAGS:
        raise ValueError("Notebook 05.1 ClearML tags are invalid")
    root = args.safe_root.expanduser().resolve()
    record = validate_safe_root(root) if args.mode != "disabled" else None
    receipt = {
        "schema_version": 1,
        "stage": "notebook_05_1_clearml_finalisation",
        "tracking_mode": args.mode,
        "credentials_source": args.credentials_source,
        "status": "pending",
        "clearml_task_id": None,
        "offline_session_folder": None,
        "project": args.project,
        "task_name": args.task_name,
        "safe_manifest_sha256": sha256(
            (root / "artifact_manifest.json").read_bytes()
        ).hexdigest(),
        "run_status": None if record is None else record["status"],
    }
    if args.mode == "disabled":
        receipt["status"] = "disabled"
        _write_receipt(args.receipt, receipt)
        return
    if not 30 <= args.wall_time_seconds <= 600:
        raise ValueError("ClearML wall time must be between 30 and 600 seconds")
    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(args.wall_time_seconds)
    try:
        _load_credentials(args.credentials_source, args.mode)
        task_id, offline = _log_online(
            root, args.project, args.task_name, tags, args.mode
        )
        receipt["clearml_task_id"] = task_id
        receipt["offline_session_folder"] = offline
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
