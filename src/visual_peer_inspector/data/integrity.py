"""Read-only verification for split dataset archives."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileVerification:
    """Verification outcome for one expected segment."""

    name: str
    expected_size: int
    actual_size: int | None
    expected_sha256: str
    actual_sha256: str | None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class VerificationReport:
    """Complete read-only verification report for a chunk directory."""

    dataset: str
    directory: Path
    files: tuple[FileVerification, ...]
    expected_total_bytes: int
    actual_total_bytes: int
    unexpected_segments: tuple[str, ...] = ()
    incomplete_files: tuple[str, ...] = ()
    manifest_errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return (
            not self.manifest_errors
            and not self.unexpected_segments
            and not self.incomplete_files
            and self.actual_total_bytes == self.expected_total_bytes
            and all(item.ok for item in self.files)
        )


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate a JSON chunk manifest."""

    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    required = {
        "dataset",
        "segment_prefix",
        "expected_total_bytes",
        "files",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Manifest is missing required keys: {', '.join(missing)}")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise ValueError("Manifest 'files' must be a non-empty list")
    return manifest


def hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a SHA-256 digest without modifying the file."""

    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def verify_chunk_directory(
    directory: Path,
    manifest: dict[str, Any],
    *,
    calculate_hashes: bool = True,
) -> VerificationReport:
    """Verify expected files, sizes and hashes in a directory."""

    directory = directory.expanduser().resolve()
    expected_entries = manifest["files"]
    expected_names = {entry["name"] for entry in expected_entries}
    file_results: list[FileVerification] = []
    actual_total = 0
    manifest_errors: list[str] = []

    declared_total = sum(int(entry["size_bytes"]) for entry in expected_entries)
    expected_total = int(manifest["expected_total_bytes"])
    if declared_total != expected_total:
        manifest_errors.append(
            "Manifest file sizes do not add up to expected_total_bytes"
        )

    for entry in expected_entries:
        name = str(entry["name"])
        expected_size = int(entry["size_bytes"])
        expected_hash = str(entry["sha256"]).lower()
        path = directory / name
        errors: list[str] = []
        actual_size: int | None = None
        actual_hash: str | None = None

        if not path.is_file():
            errors.append("missing")
        else:
            actual_size = path.stat().st_size
            actual_total += actual_size
            if actual_size != expected_size:
                errors.append(
                    f"size mismatch: expected {expected_size}, found {actual_size}"
                )
            if calculate_hashes:
                actual_hash = hash_file(path)
                if actual_hash.lower() != expected_hash:
                    errors.append("SHA-256 mismatch")

        file_results.append(
            FileVerification(
                name=name,
                expected_size=expected_size,
                actual_size=actual_size,
                expected_sha256=expected_hash,
                actual_sha256=actual_hash,
                errors=tuple(errors),
            )
        )

    prefix = str(manifest["segment_prefix"])
    unexpected = tuple(
        sorted(
            path.name
            for path in directory.glob(f"{prefix}*")
            if path.is_file() and path.name not in expected_names
        )
    )
    incomplete = tuple(
        sorted(str(path.relative_to(directory)) for path in directory.rglob("*.incomplete"))
    )

    return VerificationReport(
        dataset=str(manifest["dataset"]),
        directory=directory,
        files=tuple(file_results),
        expected_total_bytes=expected_total,
        actual_total_bytes=actual_total,
        unexpected_segments=unexpected,
        incomplete_files=incomplete,
        manifest_errors=tuple(manifest_errors),
    )
