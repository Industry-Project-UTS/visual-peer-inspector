"""Fail-closed verification for code snapshots and downloaded run bundles."""

from __future__ import annotations

from csv import DictReader
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject other top-level values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def safe_relative_path(root: Path, name: str) -> Path:
    """Resolve one relative path while preventing traversal and symlinks."""

    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {name}")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"Symlinks are not accepted: {name}")
    path = unresolved.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes its root: {name}") from error
    return path


def verify_code_snapshot_record(code_root: Path, snapshot_path: Path) -> dict[str, Any]:
    """Verify every file and the aggregate digest in a code snapshot record."""

    code_root = code_root.expanduser().resolve()
    snapshot_path = snapshot_path.expanduser().resolve(strict=True)
    if snapshot_path != code_root / "snapshot_record.json":
        raise ValueError("Code snapshot record must be the root snapshot_record.json")
    snapshot = read_json_object(snapshot_path)
    recorded_files = snapshot.get("files")
    if not isinstance(recorded_files, dict) or not recorded_files:
        raise ValueError("Code snapshot record has no file hashes")
    if snapshot.get("file_count") != len(recorded_files):
        raise ValueError("Code snapshot file count mismatch")
    verified_lines: list[str] = []
    for name, expected_hash in sorted(recorded_files.items()):
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise ValueError("Code snapshot file records must be string pairs")
        source = safe_relative_path(code_root, name)
        if not source.is_file():
            raise FileNotFoundError(f"Code snapshot file is missing: {name}")
        actual_hash = file_sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(f"Code snapshot file hash mismatch: {name}")
        verified_lines.append(f"{expected_hash}  {name}\n")
    aggregate = sha256("".join(verified_lines).encode("utf-8")).hexdigest()
    if aggregate != snapshot.get("snapshot_sha256"):
        raise ValueError("Code snapshot aggregate hash mismatch")
    expected_inventory = set(recorded_files) | {"snapshot_record.json"}
    actual_inventory: set[str] = set()
    for path in code_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Code snapshot cannot contain symbolic links")
        if path.is_file():
            actual_inventory.add(path.relative_to(code_root).as_posix())
    if actual_inventory != expected_inventory:
        raise ValueError(
            "Code snapshot inventory mismatch: "
            f"missing={sorted(expected_inventory-actual_inventory)}, "
            f"undeclared={sorted(actual_inventory-expected_inventory)}"
        )
    return snapshot


def verify_artifact_manifest(run_root: Path) -> dict[str, Any]:
    """Verify declared artifacts and require every report file to be covered."""

    run_root = run_root.expanduser().resolve()
    manifest_path = run_root / "report" / "artifact_manifest.json"
    manifest = read_json_object(manifest_path)
    if not manifest:
        raise ValueError("Artifact manifest is empty")
    for name, expected_hash in sorted(manifest.items()):
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise ValueError("Artifact manifest entries must be string pairs")
        artifact = safe_relative_path(run_root, name)
        if not artifact.is_file():
            raise FileNotFoundError(f"Run artifact is missing: {name}")
        if file_sha256(artifact) != expected_hash:
            raise ValueError(f"Run artifact hash mismatch: {name}")
    report_files = {
        path.relative_to(run_root).as_posix()
        for path in (run_root / "report").rglob("*")
        if path.is_file() and path != manifest_path
    }
    unlisted = sorted(report_files.difference(manifest))
    if unlisted:
        raise ValueError(f"Report files are absent from the artifact manifest: {unlisted}")
    return {
        "artifact_count": len(manifest),
        "report_file_count": len(report_files),
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "all_checks_passed": True,
    }


def verify_pass_gate_csv(path: Path) -> int:
    """Require a non-empty gate table whose status column is entirely pass."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(DictReader(handle))
    if not rows:
        raise ValueError(f"Gate table is empty: {path.name}")
    failures = [row for row in rows if row.get("status") != "pass"]
    if failures:
        raise ValueError(f"Gate table contains failures: {path.name}")
    return len(rows)


__all__ = [
    "file_sha256",
    "read_json_object",
    "safe_relative_path",
    "verify_artifact_manifest",
    "verify_code_snapshot_record",
    "verify_pass_gate_csv",
]
