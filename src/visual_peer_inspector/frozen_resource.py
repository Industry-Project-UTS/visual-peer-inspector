"""Build minimal hash-locked resources between protected notebook stages."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.notebook_provenance import runtime_code_provenance

SCHEMA_VERSION = "visual-peer-frozen-resource/v1"


def _safe_name(value: str) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("Frozen resource paths must be safe relative paths")
    return relative


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_bytes(canonical_json_bytes(dict(value)) + b"\n")
    os.replace(temporary, path)


def _manifest(root: Path) -> dict[str, Any]:
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def verify_frozen_resource(
    resource_root: Path,
    *,
    expected_stage: str,
    expected_files: set[str],
) -> dict[str, Any]:
    """Verify exact resource contents, manifest coverage, and every digest."""

    root = Path(resource_root).expanduser().resolve(strict=True)
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Frozen resource cannot contain symbolic links")
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    required = set(expected_files) | {"resource_record.json", "artifact_manifest.json"}
    if files != required:
        raise ValueError("Frozen resource file allowlist differs")
    record = json.loads((root / "resource_record.json").read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("status") != "complete"
        or record.get("stage") != expected_stage
    ):
        raise ValueError("Frozen resource record differs")
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise ValueError("Frozen resource manifest is invalid")
    if {str(row.get("path")) for row in rows} != files - {"artifact_manifest.json"}:
        raise ValueError("Frozen resource manifest coverage differs")
    for row in rows:
        path = root.joinpath(*_safe_name(str(row["path"])).parts)
        if (
            file_sha256(path) != str(row["sha256"])
            or path.stat().st_size != int(row["size_bytes"])
        ):
            raise ValueError("Frozen resource artifact differs")
    return record


def publish_frozen_resource(
    *,
    resource_root: Path,
    stage: str,
    files: Mapping[str, Path],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one minimal resource atomically or reuse an identical resource."""

    root = Path(resource_root).expanduser().resolve()
    bound_bindings = {
        **dict(bindings),
        **runtime_code_provenance(required=False),
    }
    expected = {_safe_name(name).as_posix() for name in files}
    if len(expected) != len(files):
        raise ValueError("Frozen resource contains duplicate target paths")
    sources = {
        _safe_name(name).as_posix(): Path(path).expanduser().resolve(strict=True)
        for name, path in files.items()
    }
    if root.exists():
        existing = verify_frozen_resource(
            root, expected_stage=stage, expected_files=expected
        )
        expected_hashes = {name: file_sha256(path) for name, path in sources.items()}
        if existing.get("file_sha256s") != expected_hashes or existing.get(
            "bindings"
        ) != bound_bindings:
            raise ValueError("Existing frozen resource differs from current inputs")
        return existing

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(root.name + ".incomplete")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for name, source in sorted(sources.items()):
            destination = staging.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "stage": stage,
            "bindings": bound_bindings,
            "file_sha256s": {
                name: file_sha256(staging.joinpath(*PurePosixPath(name).parts))
                for name in sorted(sources)
            },
        }
        _write_json(staging / "resource_record.json", record)
        _write_json(staging / "artifact_manifest.json", _manifest(staging))
        verify_frozen_resource(
            staging, expected_stage=stage, expected_files=expected
        )
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_frozen_resource(root, expected_stage=stage, expected_files=expected)


__all__ = [
    "SCHEMA_VERSION",
    "publish_frozen_resource",
    "verify_frozen_resource",
]
