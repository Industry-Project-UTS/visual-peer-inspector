"""Verified recovery bundles for long Kaggle notebook stages."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
    runtime_code_provenance,
)

RECOVERY_SCHEMA = "visual-peer-kaggle-recovery/v1"
_ARCHIVE_MANIFEST = "recovery_manifest.json"
_ARCHIVE_PREFIX = "state"


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _binding_record(binding: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    record = {
        **dict(binding),
        **runtime_code_provenance(required=False),
    }
    if not record:
        raise ValueError("A recovery bundle requires an exact identity binding")
    digest = sha256(canonical_json_bytes(record)).hexdigest()
    return record, digest


def write_safe_artifact_manifest(root: Path) -> dict[str, Any]:
    """Write the exact artifact inventory required by the ClearML finalizer."""

    safe_root = Path(root).expanduser().resolve(strict=True)
    run_record_path = safe_root / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    files = [
        path
        for path in sorted(safe_root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    record = {
        "schema_version": 1,
        "artifact_count": len(files),
        "artifacts": [
            {
                "path": path.relative_to(safe_root).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    _atomic_bytes(
        safe_root / "artifact_manifest.json",
        canonical_json_bytes(record) + b"\n",
    )
    return record


def write_recovery_tracking_root(
    *,
    safe_root: Path,
    stage: str,
    pipeline_status: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a completed, aggregate record for one verified recovery export."""

    root = Path(safe_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "completed",
        "stage": str(stage),
        "record_kind": "recovery",
        "pipeline_status": str(pipeline_status),
        "metrics": dict(metrics),
        **runtime_code_provenance(required=False),
    }
    _atomic_bytes(root / "run_record.json", canonical_json_bytes(record) + b"\n")
    write_safe_artifact_manifest(root)
    return record


def _safe_relative(value: object, allowed_top_levels: set[str]) -> PurePosixPath:
    text = str(value)
    if not text or "\\" in text or "\x00" in text:
        raise ValueError("Recovery archive contains an unsafe path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Recovery archive contains an unsafe path")
    if relative.parts[0] not in allowed_top_levels:
        raise ValueError("Recovery archive contains a path outside its allowlist")
    return relative


def _stream_sha256(handle: Any) -> tuple[str, int]:
    digest = sha256()
    size = 0
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _file_entries(source_root: Path, allowed_top_levels: set[str]) -> list[Path]:
    entries: list[Path] = []
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("Recovery source contains a symbolic link")
        if not path.is_file() or path.name.endswith(".incomplete"):
            continue
        relative = PurePosixPath(path.relative_to(source_root).as_posix())
        if relative.parts[0] in allowed_top_levels:
            entries.append(path)
    return entries


def create_recovery_bundle(
    *,
    source_root: Path,
    bundle_path: Path,
    manifest_path: Path,
    binding: Mapping[str, Any],
    allowed_top_levels: Sequence[str],
) -> dict[str, Any]:
    """Create a deterministic ZIP after hashing every allowlisted state file."""

    root = Path(source_root).expanduser().resolve(strict=True)
    allowed = {str(value) for value in allowed_top_levels}
    if not allowed or any(
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        for value in allowed
    ):
        raise ValueError("Recovery top-level allowlist is invalid")
    identity, identity_sha256 = _binding_record(binding)
    files = _file_entries(root, allowed)
    if not files:
        raise ValueError("Recovery state is empty")
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    internal = {
        "schema_version": RECOVERY_SCHEMA,
        "binding": identity,
        "binding_sha256": identity_sha256,
        "allowed_top_levels": sorted(allowed),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    destination = Path(bundle_path).expanduser().resolve()
    sidecar_path = Path(manifest_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        manifest_info = ZipInfo(_ARCHIVE_MANIFEST, date_time=(1980, 1, 1, 0, 0, 0))
        manifest_info.compress_type = ZIP_DEFLATED
        manifest_info.external_attr = 0o100644 << 16
        archive.writestr(manifest_info, canonical_json_bytes(internal) + b"\n")
        for item, source in zip(artifacts, files, strict=True):
            name = f"{_ARCHIVE_PREFIX}/{item['path']}"
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with source.open("rb") as reader, archive.open(info, "w") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Recovery ZIP failed its CRC check")
    os.replace(temporary, destination)
    sidecar = {
        **internal,
        "bundle_sha256": file_sha256(destination),
        "bundle_size_bytes": destination.stat().st_size,
    }
    _atomic_bytes(sidecar_path, canonical_json_bytes(sidecar) + b"\n")
    verify_recovery_bundle(
        bundle_path=destination,
        manifest_path=sidecar_path,
        expected_binding=identity,
        allowed_top_levels=sorted(allowed),
    )
    return sidecar


def verify_recovery_bundle(
    *,
    bundle_path: Path,
    manifest_path: Path,
    expected_binding: Mapping[str, Any],
    allowed_top_levels: Sequence[str],
) -> dict[str, Any]:
    """Verify the sidecar, archive allowlist, CRC, identity and every file hash."""

    bundle = Path(bundle_path).expanduser().resolve(strict=True)
    sidecar = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    allowed = {str(value) for value in allowed_top_levels}
    binding, binding_sha256 = _binding_record(expected_binding)
    if (
        manifest.get("schema_version") != RECOVERY_SCHEMA
        or manifest.get("binding") != binding
        or manifest.get("binding_sha256") != binding_sha256
        or manifest.get("allowed_top_levels") != sorted(allowed)
        or manifest.get("bundle_sha256") != file_sha256(bundle)
        or int(manifest.get("bundle_size_bytes", -1)) != bundle.stat().st_size
    ):
        raise ValueError("Recovery bundle identity differs")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or int(manifest.get("artifact_count", -1)) != len(artifacts)
    ):
        raise ValueError("Recovery artifact inventory is invalid")
    declared: dict[str, dict[str, Any]] = {}
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise TypeError("Recovery artifact inventory is invalid")
        relative = _safe_relative(raw.get("path"), allowed)
        name = f"{_ARCHIVE_PREFIX}/{relative.as_posix()}"
        if name in declared:
            raise ValueError("Recovery artifact inventory contains duplicate paths")
        digest = str(raw.get("sha256"))
        size = int(raw.get("size_bytes", -1))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest) or size < 0:
            raise ValueError("Recovery artifact inventory contains an invalid hash or size")
        declared[name] = raw
    expected_internal = {
        key: value
        for key, value in manifest.items()
        if key not in {"bundle_sha256", "bundle_size_bytes"}
    }
    try:
        with ZipFile(bundle) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or set(names) != set(declared) | {_ARCHIVE_MANIFEST}:
                raise ValueError("Recovery archive has an unexpected file allowlist")
            for entry in entries:
                mode = entry.external_attr >> 16
                if entry.is_dir() or stat.S_ISLNK(mode):
                    raise ValueError("Recovery archive contains an unsafe member")
                if entry.filename != _ARCHIVE_MANIFEST:
                    relative = entry.filename.removeprefix(f"{_ARCHIVE_PREFIX}/")
                    _safe_relative(relative, allowed)
            if archive.testzip() is not None:
                raise ValueError("Recovery ZIP failed its CRC check")
            internal = json.loads(archive.read(_ARCHIVE_MANIFEST))
            if internal != expected_internal:
                raise ValueError("Recovery archive and sidecar manifests differ")
            for name, item in declared.items():
                with archive.open(name) as handle:
                    digest, size = _stream_sha256(handle)
                if digest != item["sha256"] or size != int(item["size_bytes"]):
                    raise ValueError("Recovery artifact differs from its manifest")
    except BadZipFile as error:
        raise ValueError("Recovery ZIP is corrupt") from error
    return manifest


def restore_recovery_bundle(
    *,
    bundle_path: Path,
    manifest_path: Path,
    destination_root: Path,
    expected_binding: Mapping[str, Any],
    allowed_top_levels: Sequence[str],
) -> dict[str, Any]:
    """Stage a verified archive and atomically install only missing state files."""

    manifest = verify_recovery_bundle(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        expected_binding=expected_binding,
        allowed_top_levels=allowed_top_levels,
    )
    destination = Path(destination_root).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="vpi-recovery-", dir=destination.parent))
    try:
        with ZipFile(Path(bundle_path).expanduser().resolve(strict=True)) as archive:
            for item in manifest["artifacts"]:
                relative = PurePosixPath(str(item["path"]))
                target = stage.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(f"{_ARCHIVE_PREFIX}/{relative.as_posix()}") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                if file_sha256(target) != item["sha256"] or target.stat().st_size != int(item["size_bytes"]):
                    raise ValueError("Staged recovery artifact differs from its manifest")
        for item in manifest["artifacts"]:
            relative = PurePosixPath(str(item["path"]))
            target = destination.joinpath(*relative.parts)
            if target.exists() and (
                not target.is_file()
                or target.is_symlink()
                or file_sha256(target) != item["sha256"]
                or target.stat().st_size != int(item["size_bytes"])
            ):
                raise ValueError("Existing output differs from the recovery bundle")
        for item in manifest["artifacts"]:
            relative = PurePosixPath(str(item["path"]))
            target = destination.joinpath(*relative.parts)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage.joinpath(*relative.parts), target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return manifest


def prune_verified_state(
    *,
    source_root: Path,
    bundle_path: Path,
    manifest_path: Path,
    expected_binding: Mapping[str, Any],
    allowed_top_levels: Sequence[str],
) -> int:
    """Remove only files already verified in the completed recovery bundle."""

    root = Path(source_root).expanduser().resolve(strict=True)
    manifest = verify_recovery_bundle(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        expected_binding=expected_binding,
        allowed_top_levels=allowed_top_levels,
    )
    paths = [root.joinpath(*PurePosixPath(str(item["path"])).parts) for item in manifest["artifacts"]]
    for path, item in zip(paths, manifest["artifacts"], strict=True):
        if (
            not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != item["sha256"]
            or path.stat().st_size != int(item["size_bytes"])
        ):
            raise ValueError("Recovery source changed before pruning")
    for path in paths:
        path.unlink()
    for directory in sorted(
        {parent for path in paths for parent in path.parents if parent != root and root in parent.parents},
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return len(paths)


__all__ = [
    "RECOVERY_SCHEMA",
    "create_recovery_bundle",
    "prune_verified_state",
    "restore_recovery_bundle",
    "verify_recovery_bundle",
    "write_recovery_tracking_root",
    "write_safe_artifact_manifest",
]
