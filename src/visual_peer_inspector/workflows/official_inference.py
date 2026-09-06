"""Release gates and safe summaries for official prediction-only inference."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from visual_peer_inspector.data.cohort_contract import (
    OFFICIAL_SEEN_TEST_CONTRACT,
    OFFICIAL_UNSEEN_TEST_CONTRACT,
)
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.frozen_resource import verify_frozen_resource
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    verify_prediction_lock,
)

LOCK_RESULT_SCHEMA = "visual-peer-09d-lock-result/v1"
OFFICIAL_INFERENCE_SCHEMA = "visual-peer-10a-official-inference/v1"
RECOVERY_SCHEMA = "visual-peer-10a-recovery/v1"


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def load_lock_result_receipt(
    path: Path,
    *,
    method_sha256: str,
    policy_sha256: str,
    require_resource: bool = False,
) -> dict[str, Any]:
    """Verify the completed 09D protocol against the exact method and policy."""

    source = Path(path).expanduser().resolve(strict=True)
    if require_resource:
        verify_frozen_resource(
            source.parent,
            expected_stage="internal_lock_decision_release",
            expected_files={"internal_lock_result_receipt.json"},
        )
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != LOCK_RESULT_SCHEMA:
        raise ValueError("The internal-lock result receipt has an unexpected schema")
    if value.get("status") != "protocol_complete":
        raise ValueError("Official inference requires a completed internal-lock protocol")
    if value.get("decision_scope") != "evaluation_completed_once_without_method_or_policy_change":
        raise ValueError("The internal-lock receipt has an unexpected decision scope")
    if value.get("scientific_performance_accepted") is not False:
        raise ValueError("The internal-lock receipt must not claim performance acceptance")
    if value.get("method_record_sha256") != method_sha256:
        raise ValueError("The internal-lock result used another method record")
    if value.get("policy_record_sha256") != policy_sha256:
        raise ValueError("The internal-lock result used another review policy")
    if value.get("method_or_policy_changed") is not False:
        raise ValueError(
            "The method and policy must remain unchanged after internal lock"
        )
    for field in (
        "internal_lock_prediction_lock_sha256",
        "internal_lock_evaluation_manifest_sha256",
    ):
        if not _is_sha256(value.get(field)):
            raise ValueError(f"The internal-lock receipt is missing {field}")
    return {"path": source, "sha256": file_sha256(source), **value}


def verify_official_prediction_lock(path: Path, *, cohort: str) -> dict[str, Any]:
    """Verify one role-specific prediction lock and its frozen membership."""

    contract = {
        "seen": OFFICIAL_SEEN_TEST_CONTRACT,
        "unseen": OFFICIAL_UNSEEN_TEST_CONTRACT,
    }[cohort]
    record = verify_prediction_lock(path)
    expected_membership = contract.model_membership_sha256 or contract.membership_sha256
    if (
        record.get("cohort_id") != contract.cohort_id
        or record.get("cohort_membership_sha256") != expected_membership
        or int(record.get("scene_count", -1)) != contract.scene_count
        or int(record.get("image_count", -1)) != contract.image_count
    ):
        raise ValueError("Official prediction lock does not match its cohort contract")
    return record


def build_official_inference_summary(
    *,
    seen_lock_path: Path,
    unseen_lock_path: Path,
    method_sha256: str,
    policy_sha256: str,
    lock_result_receipt_sha256: str,
) -> dict[str, Any]:
    """Combine only safe receipts after separate official prediction locks verify."""

    if not all(
        _is_sha256(value)
        for value in (method_sha256, policy_sha256, lock_result_receipt_sha256)
    ):
        raise ValueError("Official inference prerequisite hashes are invalid")
    seen = verify_official_prediction_lock(seen_lock_path, cohort="seen")
    unseen = verify_official_prediction_lock(unseen_lock_path, cohort="unseen")
    return {
        "schema_version": OFFICIAL_INFERENCE_SCHEMA,
        "status": "completed",
        "method_record_sha256": method_sha256,
        "policy_record_sha256": policy_sha256,
        "lock_result_receipt_sha256": lock_result_receipt_sha256,
        "cohorts": [
            {
                "cohort_id": seen["cohort_id"],
                "scene_count": seen["scene_count"],
                "image_count": seen["image_count"],
                "prediction_lock_sha256": file_sha256(Path(seen_lock_path)),
                "identity_sha256": seen["identity_sha256"],
            },
            {
                "cohort_id": unseen["cohort_id"],
                "scene_count": unseen["scene_count"],
                "image_count": unseen["image_count"],
                "prediction_lock_sha256": file_sha256(Path(unseen_lock_path)),
                "identity_sha256": unseen["identity_sha256"],
            },
        ],
        "prediction_resources_combined": False,
        "evaluator_opened": False,
        "prediction_only": True,
        "accuracy_metrics_reported": False,
    }


def create_recovery_bundle(
    *, cohort_root: Path, output_directory: Path, cohort_id: str
) -> tuple[Path, Path]:
    """Write one deterministic, hash-locked recovery bundle for a split."""

    root = Path(cohort_root).resolve(strict=True)
    allowed_top_levels = {
        "pipeline_identity.json",
        "gate_record.json",
        "incremental_progress.json",
        "locateanything",
        "sam21",
        "dinov2_small",
        "peer_scoring",
    }
    files = [
        path
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix())
        if path.is_file()
        and ".incomplete" not in path.name
        and path.relative_to(root).parts[0] in allowed_top_levels
    ]
    if not files or not (root / "pipeline_identity.json").is_file():
        raise ValueError("Recovery state is empty or has no pipeline identity")
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    identity = json.loads((root / "pipeline_identity.json").read_text(encoding="utf-8"))
    internal = {
        "schema_version": RECOVERY_SCHEMA,
        "cohort_id": cohort_id,
        "identity_sha256": identity["identity_sha256"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / f"{cohort_id}-recovery.zip"
    temporary = bundle_path.with_suffix(".zip.incomplete")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        manifest_info = ZipInfo(
            "recovery_manifest.json", date_time=(1980, 1, 1, 0, 0, 0)
        )
        manifest_info.compress_type = ZIP_DEFLATED
        manifest_info.external_attr = 0o100644 << 16
        archive.writestr(manifest_info, canonical_json_bytes(internal) + b"\n")
        for item in artifacts:
            name = f"{cohort_id}/{item['path']}"
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (
                root.joinpath(item["path"]).open("rb") as source,
                archive.open(info, "w") as sink,
            ):
                shutil.copyfileobj(source, sink, length=1024 * 1024)
    os.replace(temporary, bundle_path)
    sidecar = {
        **internal,
        "bundle_sha256": file_sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    manifest_path = output / f"{cohort_id}-recovery-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(sidecar) + b"\n")
    return bundle_path, manifest_path


def restore_recovery_bundle(
    *, bundle_path: Path, manifest_path: Path, output_root: Path, cohort_id: str
) -> dict[str, Any]:
    """Verify and restore exactly one previous recovery bundle."""

    bundle = Path(bundle_path).resolve(strict=True)
    manifest = json.loads(
        Path(manifest_path).resolve(strict=True).read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != RECOVERY_SCHEMA
        or manifest.get("cohort_id") != cohort_id
        or manifest.get("bundle_sha256") != file_sha256(bundle)
    ):
        raise ValueError("Official recovery bundle identity differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or int(
        manifest.get("artifact_count", -1)
    ) != len(artifacts):
        raise ValueError("Official recovery artifact inventory is invalid")
    declared = {f"{cohort_id}/{item['path']}" for item in artifacts}
    destination = Path(output_root).resolve() / cohort_id
    with ZipFile(bundle) as archive:
        names = set(archive.namelist())
        if names != declared | {"recovery_manifest.json"}:
            raise ValueError(
                "Official recovery archive has an unexpected file allowlist"
            )
        internal = json.loads(archive.read("recovery_manifest.json"))
        expected_internal = {
            key: value
            for key, value in manifest.items()
            if key not in {"bundle_sha256", "bundle_size_bytes"}
        }
        if internal != expected_internal:
            raise ValueError("Official recovery manifests differ")
        for item in artifacts:
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Official recovery archive contains an unsafe path")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                archive.open(f"{cohort_id}/{relative.as_posix()}") as source,
                target.with_suffix(target.suffix + ".incomplete").open("wb") as sink,
            ):
                shutil.copyfileobj(source, sink)
            temporary = target.with_suffix(target.suffix + ".incomplete")
            if file_sha256(temporary) != item[
                "sha256"
            ] or temporary.stat().st_size != int(item["size_bytes"]):
                raise ValueError("Official recovery artifact differs from its manifest")
            if target.is_file() and file_sha256(target) != item["sha256"]:
                raise ValueError("Existing output differs from the recovery bundle")
            os.replace(temporary, target)
    return manifest


__all__ = [
    "LOCK_RESULT_SCHEMA",
    "OFFICIAL_INFERENCE_SCHEMA",
    "RECOVERY_SCHEMA",
    "build_official_inference_summary",
    "create_recovery_bundle",
    "load_lock_result_receipt",
    "restore_recovery_bundle",
    "verify_official_prediction_lock",
]
