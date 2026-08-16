from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkResult:
    name: str
    expected_size: int
    actual_size: int | None
    expected_hash: str
    actual_hash: str | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ChunkReport:
    dataset: str
    expected_bytes: int
    actual_bytes: int
    files: tuple[ChunkResult, ...]
    unexpected_files: tuple[str, ...]
    incomplete_files: tuple[str, ...]
    manifest_errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.expected_bytes == self.actual_bytes
            and all(item.ok for item in self.files)
            and not self.unexpected_files
            and not self.incomplete_files
            and not self.manifest_errors
        )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def hash_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def verify_chunks(
    directory: Path,
    manifest: dict[str, Any],
    *,
    check_hashes: bool = True,
) -> ChunkReport:
    required = {"dataset", "segment_prefix", "expected_total_bytes", "files"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Manifest is missing: {', '.join(missing)}")

    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest files must be a non-empty list")

    directory = directory.expanduser().resolve()
    expected_names = {str(item["name"]) for item in entries}
    expected_bytes = int(manifest["expected_total_bytes"])
    declared_bytes = sum(int(item["size_bytes"]) for item in entries)
    manifest_errors = []
    if declared_bytes != expected_bytes:
        manifest_errors.append("File sizes do not add up to expected_total_bytes")

    results = []
    actual_bytes = 0
    for entry in entries:
        name = str(entry["name"])
        expected_size = int(entry["size_bytes"])
        expected_hash = str(entry["sha256"]).lower()
        path = directory / name
        errors = []
        actual_size = None
        actual_hash = None

        if not path.is_file():
            errors.append("missing")
        else:
            actual_size = path.stat().st_size
            actual_bytes += actual_size
            if actual_size != expected_size:
                errors.append(f"expected {expected_size} bytes, found {actual_size}")
            if check_hashes:
                actual_hash = hash_file(path)
                if actual_hash != expected_hash:
                    errors.append("SHA-256 mismatch")

        results.append(
            ChunkResult(
                name=name,
                expected_size=expected_size,
                actual_size=actual_size,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
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
    return ChunkReport(
        dataset=str(manifest["dataset"]),
        expected_bytes=expected_bytes,
        actual_bytes=actual_bytes,
        files=tuple(results),
        unexpected_files=unexpected,
        incomplete_files=incomplete,
        manifest_errors=tuple(manifest_errors),
    )
