"""Read the same verified source from a ZIP or Kaggle's extracted folders."""

from __future__ import annotations

import json
from binascii import crc32
from pathlib import Path, PurePosixPath
from typing import Any

from visual_peer_inspector.data.archive_members import IndexedMember
from visual_peer_inspector.data.model_safe_package import file_sha256


def extracted_source_root(archive_path: Path) -> Path | None:
    """Locate the extraction directory without searching unrelated mounts."""
    for candidate in (archive_path.with_suffix(""), archive_path.parent):
        if candidate.is_dir() and (
            candidate == archive_path.with_suffix("") or (candidate / "toysad8k").is_dir()
        ):
            if candidate.is_symlink():
                raise ValueError("The extracted source root cannot be a symlink")
            return candidate
    return None


def read_extracted_source_member(
    root: Path, member: IndexedMember, *, maximum_bytes: int
) -> bytes:
    """Check indexed metadata, path, byte length and CRC before using a file."""
    parts = PurePosixPath(member.name).parts
    if (
        not parts or member.name.startswith("/") or ".." in parts
        or "\\" in member.name or member.is_directory or member.is_symlink
        or member.unsafe_reason or member.flags & 1
    ):
        raise ValueError(f"Unsafe extracted member: {member.name}")
    if not 0 <= member.uncompressed_size <= maximum_bytes:
        raise ValueError(f"Member exceeds read limit: {member.name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink in extracted member path: {member.name}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing or unsafe extracted member: {member.name}")
    if path.stat().st_size != member.uncompressed_size:
        raise ValueError(f"Uncompressed-size mismatch: {member.name}")
    with path.open("rb") as stream:
        content = stream.read(maximum_bytes + 1)
    if len(content) != member.uncompressed_size:
        raise ValueError(f"Uncompressed-size mismatch: {member.name}")
    if crc32(content) & 0xFFFFFFFF != member.crc32:
        raise ValueError(f"CRC mismatch: {member.name}")
    return content


def verify_archive_storage(
    root: Path,
    declared: dict[str, str],
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    verify_archive_hash: bool = False,
) -> dict[str, Any]:
    """Record transport separately from the original archive identity."""
    archive = root / "toysAD8K.zip"
    digest = declared.get(archive.name, "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Source package does not declare the complete archive SHA-256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("The source package declares another archive")
    if archive.is_file():
        size = archive.stat().st_size
        if expected_size is not None and size != expected_size:
            raise ValueError("The source archive has an unexpected size")
        if verify_archive_hash and file_sha256(archive) != digest:
            raise ValueError("The source archive SHA-256 does not match")
        mode = "indexed_zip"
        bytes_verified = verify_archive_hash
    else:
        if extracted_source_root(archive) is None:
            raise FileNotFoundError("No source ZIP or extracted source directory was found")
        # Kaggle removes the ZIP after extraction. Bind its recorded identity and
        # validate every requested member against the unchanged SQLite index.
        for name in (
            "archive-manifest.json", "archive_summary.json", "source-chunks-manifest.json"
        ):
            if declared.get(name) != file_sha256(root / name):
                raise ValueError(f"Extracted source identity checksum differs: {name}")
        reconstruction = json.loads((root / "archive-manifest.json").read_text())["reconstruction"]
        size = reconstruction["expected_size_bytes"]
        if type(size) is not int or size <= 0 or reconstruction["sha256"] != digest:
            raise ValueError("The extracted source declares another archive identity")
        if expected_size is not None and size != expected_size:
            raise ValueError("The extracted source declares another archive size")
        mode = "kaggle_extracted_members"
        bytes_verified = False
    return {
        "archive_sha256": digest,
        "archive_size_bytes": size,
        "archive_bytes_verified": bytes_verified,
        "source_access_mode": mode,
        "member_bytes_verification": "indexed_size_and_crc32",
    }
