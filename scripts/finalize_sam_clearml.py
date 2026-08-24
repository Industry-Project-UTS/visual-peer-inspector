#!/usr/bin/env python3
"""Upload only allowlisted aggregate Notebook 05 evidence to ClearML."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re


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
    "artifact_manifest.json",
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
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_scene|opaque_object|category|anomaly|"
    r"raw_response|box_xyxy|mask_rle|labels_by_object",
    re.IGNORECASE,
)
ALLOWED_PROVENANCE_KEYS = {
    "git_commit",
    "git_dirty",
    "snapshot_sha256",
    "precommit_candidate",
    "notebook_sha256",
    "kaggle_notebook_slug",
    "kaggle_notebook_version",
    "code_dataset_version",
    "prompt_dataset_version",
    "model_input_dataset_version",
}
ALLOWED_PLATFORM_KEYS = {
    "python",
    "numpy",
    "pillow",
    "torch",
    "torchvision",
    "kaggle_notebook_version",
    "code_dataset_version",
    "prompt_dataset_version",
    "model_input_dataset_version",
}
ALLOWED_GPU_KEYS = {
    "cuda_available",
    "device_count",
    "device_name",
    "compute_capability",
    "total_memory_gib",
    "torch_architecture",
    "torch_architecture_supported",
    "peak_allocated_gib",
    "peak_reserved_gib",
}
ALLOWED_LOAD_KEYS = {
    "stage",
    "elapsed_seconds",
    "working_free_gib",
    "gpu_allocated_gib",
    "gpu_reserved_gib",
    "download_seconds",
    "checkpoint_size_bytes",
    "checkpoint_sha256",
}
ALLOWED_RUN_KEYS = {
    "schema_version",
    "stage",
    "status",
    "tracking_status",
    "run_mode",
    "inference_stage",
    "producer_kind",
    "python_version",
    "configuration",
    "configuration_sha256",
    "prompt_package_sha256",
    "checkpoint_sha256",
    "provenance",
    "platform_versions",
    "gpu",
    "model_load_checkpoints",
    "summary",
    "reference_prompt_count",
    "automatic_prompt_count",
    "repeat_agreement",
    "completion_gate_passed",
    "error_type",
}
ALLOWED_CONFIGURATION_KEYS = {
    "code_repository",
    "code_revision",
    "model_repository",
    "model_revision",
    "checkpoint_filename",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "model_config",
    "device",
    "precision",
    "multimask_output",
    "mask_threshold",
    "max_hole_area",
    "max_sprinkle_area",
    "seed",
    "image_height",
    "image_width",
}
ALLOWED_LOAD_STAGES = {
    "dependency_install_started",
    "dependency_install_completed",
    "sam_installation_verified",
    "checkpoint_download_started",
    "checkpoint_download_completed",
    "sam_import_started",
    "sam_model_build_started",
    "sam_worker_ready",
    "waiting",
}
DEFAULT_PROJECT = "Visual Peer Inspector/02 Detection and Segmentation"
TASK_NAME_PATTERN = re.compile(r"^ds0-sam21-kaggle-(candidate|final)$")
ALLOWED_TAGS = {
    "candidate",
    "final",
    "notebook-05",
    "ds0",
    "sam2.1",
    "kaggle",
    "full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT
    )
    parser.add_argument("--task-name", default="ds0-sam21-kaggle-candidate")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def validate_safe_root(root: Path) -> None:
    root = root.expanduser().resolve(strict=True)
    allowed = set(ALLOWED_TABLES) | ALLOWED_JSON | ALLOWED_CHARTS
    entries = list(root.iterdir())
    if any(path.is_symlink() for path in entries):
        raise ValueError("ClearML safe root cannot contain symbolic links")
    if any(not path.is_file() for path in entries):
        raise ValueError("ClearML safe root can contain files only")
    if any(path.resolve(strict=True).parent != root for path in entries):
        raise ValueError("A ClearML file resolves outside the safe root")
    actual = {path.name for path in entries}
    unexpected = actual - allowed
    if unexpected:
        raise ValueError(f"ClearML safe root contains unexpected files: {sorted(unexpected)}")
    missing = (set(ALLOWED_TABLES) | ALLOWED_JSON | ALLOWED_CHARTS) - actual
    if missing:
        raise ValueError(f"ClearML safe root is missing required files: {sorted(missing)}")
    for name, expected_columns in ALLOWED_TABLES.items():
        with (root / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        columns = rows[0]
        if columns != expected_columns:
            raise ValueError(f"Unexpected ClearML columns in {name}: {columns}")
        if FORBIDDEN_TEXT.search("\n".join(",".join(row) for row in rows)):
            raise ValueError(f"Protected detail found in ClearML file: {name}")
    for name in ALLOWED_JSON:
        text = (root / name).read_text(encoding="utf-8")
        if FORBIDDEN_TEXT.search(text):
            raise ValueError(f"Protected detail found in ClearML file: {name}")
        json.loads(text)
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise ValueError("ClearML artifact manifest has no artifact rows")
    if int(manifest.get("artifact_count", -1)) != len(artifact_rows):
        raise ValueError("ClearML artifact count is inconsistent")
    expected_names = actual - {"artifact_manifest.json"}
    manifest_names = {str(row["path"]) for row in artifact_rows}
    if manifest_names != expected_names:
        raise ValueError("ClearML artifact manifest does not match the safe files")
    for row in artifact_rows:
        name = str(row["path"])
        if Path(name).name != name or "private" in name.lower():
            raise ValueError("ClearML artifact manifest contains an unsafe path")
        path = root / name
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != str(row["sha256"]) or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"ClearML artifact evidence differs for {name}")
    manifest_content_sha256 = sha256(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for row in artifact_rows
        )
    ).hexdigest()
    if manifest_content_sha256 != manifest.get("manifest_content_sha256"):
        raise ValueError("ClearML artifact manifest content hash differs")

    run_record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    if set(run_record) != ALLOWED_RUN_KEYS:
        raise ValueError("ClearML run record does not follow the accepted schema")
    if set(run_record.get("configuration", {})) != ALLOWED_CONFIGURATION_KEYS:
        raise ValueError("ClearML SAM configuration does not follow the accepted schema")
    required = {
        "run_mode": "model_inference",
        "inference_stage": "full",
        "producer_kind": "sam2.1_model",
        "status": "completed",
        "completion_gate_passed": True,
        "reference_prompt_count": 1640,
        "automatic_prompt_count": 1616,
    }
    for key, expected in required.items():
        if run_record.get(key) != expected:
            raise ValueError(f"ClearML requires a completed full model run: {key}")
    summary = run_record.get("summary", {})
    if summary.get("image_count") != 400 or summary.get("mask_count") != 3256:
        raise ValueError("ClearML requires complete DS0 image and mask coverage")
    if set(run_record.get("provenance", {})) - ALLOWED_PROVENANCE_KEYS:
        raise ValueError("ClearML run record contains unexpected provenance fields")
    if set(run_record.get("platform_versions", {})) - ALLOWED_PLATFORM_KEYS:
        raise ValueError("ClearML run record contains unexpected platform fields")
    if set(run_record.get("gpu", {})) - ALLOWED_GPU_KEYS:
        raise ValueError("ClearML run record contains unexpected GPU fields")
    for row in run_record.get("model_load_checkpoints", []):
        if not isinstance(row, dict) or set(row) - ALLOWED_LOAD_KEYS:
            raise ValueError("ClearML run record contains unexpected model-load fields")
        if row.get("stage") not in ALLOWED_LOAD_STAGES:
            raise ValueError("ClearML run record contains an unexpected model-load stage")
        if any(
            not isinstance(value, (str, int, float, bool)) and value is not None
            for value in row.values()
        ):
            raise ValueError("ClearML model-load evidence contains an invalid value")
    protected_sections = {
        "provenance": run_record.get("provenance", {}),
        "platform_versions": run_record.get("platform_versions", {}),
        "gpu": run_record.get("gpu", {}),
        "model_load_checkpoints": run_record.get("model_load_checkpoints", []),
    }
    protected_text = json.dumps(protected_sections, sort_keys=True)
    if re.search(r"(^|[\"'])/(kaggle|home|Users|tmp)/", protected_text):
        raise ValueError("ClearML run record contains an absolute local path")


def main() -> None:
    args = parse_args()
    if args.project != DEFAULT_PROJECT:
        raise ValueError("Notebook 05 ClearML project name is fixed")
    if not TASK_NAME_PATTERN.fullmatch(args.task_name):
        raise ValueError("Notebook 05 ClearML task name is invalid")
    tags = args.tag or ["candidate", "notebook-05", "ds0", "sam2.1", "kaggle", "full"]
    if set(tags) - ALLOWED_TAGS:
        raise ValueError("Notebook 05 ClearML tags are invalid")
    root = args.safe_root.expanduser().resolve()
    validate_safe_root(root)
    if args.mode == "disabled":
        receipt = {
            "schema_version": 1,
            "status": "disabled",
            "tracking_mode": "disabled",
            "clearml_task_id": None,
            "project": args.project,
            "task_name": args.task_name,
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("tracking disabled")
        return
    from clearml import Task
    from visual_peer_inspector.tracking import _configure_clearml_request_timeouts
    import pandas as pd

    if args.mode == "offline":
        Task.set_offline(True)
    _configure_clearml_request_timeouts()
    task = Task.init(
        project_name=args.project,
        task_name=args.task_name,
        task_type=Task.TaskTypes.inference,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_resource_monitoring=False,
        auto_connect_streams=False,
        reuse_last_task_id=False,
    )
    task.add_tags(tags)
    logger = task.get_logger()
    run_record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    summary = run_record["summary"]
    for series, value in {
        "completed images": summary["image_count"],
        "completed masks": summary["mask_count"],
        "empty masks": summary["empty_mask_count"],
        "median seconds per image": summary["median_image_runtime_seconds"],
        "p95 seconds per image": summary["p95_image_runtime_seconds"],
        "reference prompt count": run_record["reference_prompt_count"],
        "automatic prompt count": run_record["automatic_prompt_count"],
        "mask repeat agreement": run_record["repeat_agreement"][
            "mask_digest_agreement_rate"
        ],
        "score repeat agreement": run_record["repeat_agreement"][
            "score_agreement_rate"
        ],
    }.items():
        logger.report_scalar("DS0 SAM 2.1", series, float(value), iteration=0)
    for name in ALLOWED_TABLES:
        logger.report_table(
            "DS0 SAM 2.1",
            name.removesuffix(".csv").replace("_", " "),
            table_plot=pd.read_csv(root / name),
            iteration=0,
        )
    for name in sorted(ALLOWED_CHARTS):
        path = root / name
        if path.is_file():
            logger.report_image(
                "DS0 SAM 2.1",
                name.removesuffix(".png").replace("_", " "),
                local_path=str(path),
                iteration=0,
            )
    task.connect(
        {
            "configuration_sha256": run_record["configuration_sha256"],
            "prompt_package_sha256": run_record["prompt_package_sha256"],
            "checkpoint_sha256": run_record["checkpoint_sha256"],
            "code_revision": run_record["configuration"]["code_revision"],
            "model_revision": run_record["configuration"]["model_revision"],
            "completion_gate_passed": run_record["completion_gate_passed"],
        },
        name="verified provenance",
    )
    for name in sorted(ALLOWED_JSON):
        task.upload_artifact(
            name.removesuffix(".json").replace("_", " "),
            artifact_object=json.loads((root / name).read_text(encoding="utf-8")),
            wait_on_upload=False,
        )
    task_id = task.id
    offline_folder = (
        str(task.get_offline_mode_folder()) if args.mode == "offline" else None
    )
    task.flush(wait_for_uploads=True)
    task.close()
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "tracking_mode": args.mode,
        "clearml_task_id": task_id,
        "offline_session_folder": offline_folder,
        "project": args.project,
        "task_name": args.task_name,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(task_id)


if __name__ == "__main__":
    main()
