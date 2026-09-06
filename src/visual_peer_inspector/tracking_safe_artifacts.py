"""Validate aggregate-only artifact roots before external tracking."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
)

ALLOWED_SUFFIXES = {".csv", ".json", ".png"}
FORBIDDEN_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_scene_id|labels_by_object|mask_rle|box_xyxy|"
    r"/kaggle/input/|/(?:Users|home)/|CLEARML_API_(?:ACCESS|SECRET)_KEY|"
    r"HF_TOKEN|source_scene_id",
    re.IGNORECASE,
)
PROTECTED_PATH_PARTS = (
    "private",
    "review-only",
    "cache",
    "crosswalk",
    "credential",
    "secret",
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tracking_safe_root(
    root: Path,
    *,
    expected_stage: str,
    allowed_relative_paths: Collection[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a completed aggregate root, its inventory, and every file digest."""

    resolved = Path(root).expanduser().resolve(strict=True)
    if resolved.is_symlink() or any(path.is_symlink() for path in resolved.rglob("*")):
        raise ValueError("ClearML safe root cannot contain symbolic links")
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    relative_files = {path.relative_to(resolved).as_posix() for path in files}
    required = {"artifact_manifest.json", "run_record.json"}
    if not required.issubset(relative_files):
        raise ValueError("ClearML safe root is missing its manifest or run record")
    if allowed_relative_paths is not None:
        allowed = set(allowed_relative_paths) | required
        if relative_files != allowed:
            raise ValueError("ClearML safe root differs from the stage allowlist")
    if any(path.suffix.lower() not in ALLOWED_SUFFIXES for path in files):
        raise ValueError("ClearML safe root contains an unsupported file type")

    for path in files:
        relative = path.relative_to(resolved).as_posix().lower()
        if any(part in relative for part in PROTECTED_PATH_PARTS):
            raise ValueError("ClearML safe root contains a protected path")
        if (
            path.suffix.lower() in {".csv", ".json"}
            and path.name != "artifact_manifest.json"
            and FORBIDDEN_TEXT.search(path.read_text(encoding="utf-8"))
        ):
            raise ValueError(f"Protected detail found in ClearML artifact: {relative}")

    manifest = json.loads(
        (resolved / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(
        rows
    ):
        raise ValueError("ClearML artifact manifest is invalid")
    declared = {str(row.get("path")) for row in rows}
    if declared != relative_files - {"artifact_manifest.json"}:
        raise ValueError("ClearML artifact manifest coverage differs")
    for row in rows:
        path = resolved / str(row["path"])
        if file_sha256(path) != str(row["sha256"]) or path.stat().st_size != int(
            row["size_bytes"]
        ):
            raise ValueError("ClearML artifact digest differs")

    run_record = json.loads((resolved / "run_record.json").read_text(encoding="utf-8"))
    if run_record.get("status") not in {"complete", "completed"}:
        raise ValueError("ClearML run record is not complete")
    if run_record.get("stage") != expected_stage:
        raise ValueError("ClearML run record is for a different stage")
    return run_record, manifest


def write_tracking_safe_manifest(root: Path) -> Mapping[str, Any]:
    """Write the manifest format accepted by the tracking validator."""

    resolved = Path(root)
    run_record_path = resolved / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    artifacts = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError("ClearML safe root cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(resolved).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    path = resolved / "artifact_manifest.json"
    temporary = path.with_suffix(".json.incomplete")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return manifest


__all__ = [
    "ALLOWED_SUFFIXES",
    "FORBIDDEN_TEXT",
    "PROTECTED_PATH_PARTS",
    "file_sha256",
    "validate_tracking_safe_root",
    "write_tracking_safe_manifest",
]
