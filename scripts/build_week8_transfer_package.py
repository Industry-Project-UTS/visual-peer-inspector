#!/usr/bin/env python3
"""Build the final Week 8 Git handoff from explicit, verified inputs.

The scientific notebooks create compact per-stage bundles. This script performs
the separate handoff step: it verifies every required stage bundle, copies only
files named in a manifest, checks the repository and review boundaries, writes
one checksum inventory, and creates a deterministic ZIP.

The script deliberately has no discovery mode. A transfer manifest must name
every input root, source file, destination and stage bundle directory.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Iterable, Mapping, Sequence
import zipfile


PACKAGE_NAME = "week8-confirmation-package"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
REQUIRED_STAGES = frozenset(
    {
        "ds1-locateanything-prompt-freeze",
        "ds2-confirmation-release",
        "ds2-locateanything",
        "ds2-reference-sam21",
        "ds2-automatic-sam21",
        "ds2-dinov2-p1",
        "ds2-confirmation-and-multiview",
    }
)
REQUIRED_NOTEBOOKS = frozenset(
    {
        "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb",
        "08a-toysad8k-ds2-confirmation-release.ipynb",
        "08b-toysad8k-ds2-frozen-pipeline.ipynb",
        "08c-toysad8k-ds2-confirmation-and-multiview.ipynb",
    }
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "package_name",
        "source_roots",
        "instructions",
        "required_stages",
        "repository_files",
        "review_files",
    }
)
FILE_ENTRY_KEYS = frozenset({"root", "source", "destination"})
STAGE_ENTRY_KEYS = frozenset({"stage", "root", "bundle_root"})
INSTRUCTION_KEYS = frozenset({"root", "source"})
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "clearml.conf",
        "credentials.json",
        "kaggle.json",
        "secrets.json",
    }
)
FORBIDDEN_PATH_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "cache",
        "caches",
        "downstream",
        "model-input",
        "private",
        "transfers",
    }
)
REPOSITORY_ALLOWED_SUFFIXES = frozenset(
    {".csv", ".ipynb", ".json", ".md", ".png", ".py", ".txt"}
)
REVIEW_ALLOWED_SUFFIXES = frozenset(
    {".csv", ".ipynb", ".json", ".log", ".md", ".png", ".sha256", ".txt"}
)
SECRET_TEXT = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\bhf_[A-Za-z0-9]{20,}\b|"
    r"\bKGAT_[A-Za-z0-9_-]{20,}\b|"
    r"(?i:(?:api[_-]?(?:secret|access)?[_-]?key|password|token)"
    r"\s*[:=]\s*[\"'][^\"'${}\s]{8,}[\"'])"
)
LOCAL_PATH_TEXT = re.compile(r"/(?:Users|home)/[^\s\"']+")
PROTECTED_REPORT_TEXT = re.compile(
    r"\bS\d{6}\b|\bO\d{2}\b|opaque_scene_id|source_scene_id|"
    r"labels_by_object|mask_rle|box_xyxy|/kaggle/input/",
    re.IGNORECASE,
)


class TransferPackageError(ValueError):
    """Raised when a transfer input or boundary check fails."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256(
        b"".join(
            (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            for row in rows
        )
    ).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise TransferPackageError(
            f"{label} keys differ: expected {sorted(expected)}, found {sorted(actual)}"
        )


def _safe_relative(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise TransferPackageError(f"{label} must be a safe relative path: {value}")
    return path


def _path_has_symlink(root: Path, relative: Path) -> bool:
    current = root
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _resolve_root(value: str, manifest_parent: Path, *, name: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else manifest_parent / raw
    if candidate.is_symlink() or not candidate.is_dir():
        raise TransferPackageError(f"Source root is missing or unsafe: {name}")
    return candidate.resolve(strict=True)


def _resolve_source(
    roots: Mapping[str, Path], entry: Mapping[str, Any], *, label: str
) -> tuple[Path, Path]:
    _require_exact_keys(entry, FILE_ENTRY_KEYS, label)
    root_name = str(entry["root"])
    if root_name not in roots:
        raise TransferPackageError(f"{label} uses an unknown source root: {root_name}")
    relative = _safe_relative(str(entry["source"]), label=f"{label} source")
    root = roots[root_name]
    if _path_has_symlink(root, relative):
        raise TransferPackageError(f"{label} source contains a symbolic link: {relative}")
    source = root / relative
    if not source.is_file():
        raise TransferPackageError(f"{label} source is missing or not a file: {relative}")
    return source, _safe_relative(str(entry["destination"]), label=f"{label} destination")


def _read_text_if_applicable(path: Path) -> str | None:
    if path.name != "SHA256SUMS" and path.suffix.lower() not in {
        ".csv",
        ".ipynb",
        ".json",
        ".log",
        ".md",
        ".py",
        ".sha256",
        ".txt",
    }:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise TransferPackageError(f"Text input is not valid UTF-8: {path.name}") from error
    if path.suffix.lower() != ".ipynb":
        return text
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as error:
        raise TransferPackageError(f"Notebook is not valid JSON: {path.name}") from error
    strings: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(notebook)
    return "\n".join(strings)


def _validate_no_secret_or_local_path(path: Path) -> None:
    if path.name.lower() in FORBIDDEN_NAMES:
        raise TransferPackageError(f"Credential file is forbidden: {path.name}")
    text = _read_text_if_applicable(path)
    if text is None:
        return
    if SECRET_TEXT.search(text):
        raise TransferPackageError(f"Possible credential found in transfer input: {path.name}")
    if LOCAL_PATH_TEXT.search(text):
        raise TransferPackageError(f"Local absolute path found in transfer input: {path.name}")


def _validate_repository_destination(relative: Path, source: Path) -> None:
    lowered = tuple(part.lower() for part in relative.parts)
    if any(part in FORBIDDEN_PATH_PARTS for part in lowered):
        raise TransferPackageError(f"Repository boundary violation: {relative}")
    if source.suffix.lower() not in REPOSITORY_ALLOWED_SUFFIXES:
        raise TransferPackageError(f"Unsupported repository file type: {relative}")
    if source.name.lower() in {"local_upload_steps.md", "checksums.sha256"}:
        raise TransferPackageError(f"Transfer control file cannot enter repository-files: {relative}")
    if source.suffix.lower() in {".log", ".zip"} or "clearml_receipt" in source.name.lower():
        raise TransferPackageError(f"Review-only file cannot enter repository-files: {relative}")
    _validate_no_secret_or_local_path(source)
    if relative.parts[0].lower() == "reports":
        text = _read_text_if_applicable(source)
        if text is not None and PROTECTED_REPORT_TEXT.search(text):
            raise TransferPackageError(f"Protected detail found in repository report: {relative}")


def _validate_review_destination(relative: Path, source: Path) -> None:
    if source.name != "SHA256SUMS" and source.suffix.lower() not in REVIEW_ALLOWED_SUFFIXES:
        raise TransferPackageError(f"Unsupported review-only file type: {relative}")
    _validate_no_secret_or_local_path(source)


def _validate_clean_notebook(path: Path) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransferPackageError(f"Notebook is not valid JSON: {path.name}") from error
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise TransferPackageError(f"Notebook has no cell list: {path.name}")
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs") not in ([], None):
            raise TransferPackageError(
                f"Repository notebook contains saved output at cell {index}: {path.name}"
            )


def _parse_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise TransferPackageError(f"Invalid checksum row at {path.name}:{line_number}")
        name = parts[1].strip()
        relative = _safe_relative(name, label="checksum member")
        canonical = relative.as_posix()
        if canonical in rows:
            raise TransferPackageError(f"Duplicate checksum member: {canonical}")
        rows[canonical] = parts[0]
    return rows


def _verify_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise TransferPackageError(f"Archive integrity failed: {path.name}")
            for info in archive.infolist():
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise TransferPackageError(
                        f"Archive contains an unsafe member: {path.name}:{info.filename}"
                    )
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise TransferPackageError(
                        f"Archive contains a symbolic link: {path.name}:{info.filename}"
                    )
    except zipfile.BadZipFile as error:
        raise TransferPackageError(f"Archive cannot be opened: {path.name}") from error


def _verify_stage_bundle(
    *, stage: str, root: Path, bundle_relative: Path
) -> dict[str, Any]:
    if _path_has_symlink(root, bundle_relative):
        raise TransferPackageError(f"Stage bundle path contains a symbolic link: {stage}")
    bundle_root = root / bundle_relative
    if not bundle_root.is_dir():
        raise TransferPackageError(f"Required stage bundle is missing: {stage}")
    verification_path = bundle_root / f"{stage}-verification.json"
    checksums_path = bundle_root / "SHA256SUMS"
    if not verification_path.is_file() or not checksums_path.is_file():
        raise TransferPackageError(f"Stage verification inputs are incomplete: {stage}")
    _validate_no_secret_or_local_path(verification_path)
    _validate_no_secret_or_local_path(checksums_path)
    try:
        record = json.loads(verification_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TransferPackageError(f"Stage verification JSON is invalid: {stage}") from error
    if (
        record.get("schema_version") != 1
        or record.get("stage") != stage
        or record.get("status") != "completed"
    ):
        raise TransferPackageError(f"Stage is not recorded as completed: {stage}")
    safe_manifest_sha256 = str(record.get("safe_manifest_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", safe_manifest_sha256) is None:
        raise TransferPackageError(f"Stage safe manifest hash is invalid: {stage}")
    try:
        expected_counts = {
            "repository-evidence": int(record["safe_file_count"]),
            "review-evidence": int(record["review_file_count"]),
            "downstream-input": int(record["downstream_file_count"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise TransferPackageError(f"Stage bundle counts are invalid: {stage}") from error
    if expected_counts["repository-evidence"] < 2 or any(
        value < 0 for value in expected_counts.values()
    ):
        raise TransferPackageError(f"Stage bundle counts are invalid: {stage}")
    archives = record.get("archives")
    if not isinstance(archives, list) or not archives:
        raise TransferPackageError(f"Stage has no verified archives: {stage}")
    archive_names: set[str] = set()
    archive_rows: dict[str, Mapping[str, Any]] = {}
    for row in archives:
        if not isinstance(row, dict):
            raise TransferPackageError(f"Stage archive record is invalid: {stage}")
        name = _safe_relative(str(row.get("path", "")), label="stage archive").as_posix()
        if "/" in name or not name.endswith(".zip") or name in archive_names:
            raise TransferPackageError(f"Stage archive name is invalid or duplicated: {stage}:{name}")
        archive_names.add(name)
        archive_rows[name] = row
        archive_path = bundle_root / name
        if archive_path.is_symlink() or not archive_path.is_file():
            raise TransferPackageError(f"Stage archive is missing or unsafe: {stage}:{name}")
        if archive_path.stat().st_size != int(row.get("size_bytes", -1)):
            raise TransferPackageError(f"Stage archive size differs: {stage}:{name}")
        if file_sha256(archive_path) != str(row.get("sha256", "")):
            raise TransferPackageError(f"Stage archive hash differs: {stage}:{name}")
        if int(row.get("source_file_count", -1)) < 1:
            raise TransferPackageError(f"Stage archive source count is invalid: {stage}:{name}")
        _verify_zip(archive_path)
    expected_archive_names = {
        f"{stage}-{kind}.zip"
        for kind, count in expected_counts.items()
        if count > 0
    }
    if archive_names != expected_archive_names:
        raise TransferPackageError(f"Stage archive roles differ: {stage}")
    for kind, count in expected_counts.items():
        if count == 0:
            continue
        row = archive_rows[f"{stage}-{kind}.zip"]
        if int(row.get("source_file_count", -1)) != count:
            raise TransferPackageError(f"Stage archive source count differs: {stage}:{kind}")

    repository_archive = bundle_root / f"{stage}-repository-evidence.zip"
    with zipfile.ZipFile(repository_archive) as archive:
        manifest_member = "safe/artifact_manifest.json"
        if archive.namelist().count(manifest_member) != 1:
            raise TransferPackageError(
                f"Stage repository archive has no unique safe artifact manifest: {stage}"
            )
        archived_manifest_sha256 = sha256(archive.read(manifest_member)).hexdigest()
    if archived_manifest_sha256 != safe_manifest_sha256:
        raise TransferPackageError(
            f"Stage safe manifest does not match its repository archive: {stage}"
        )
    checksums = _parse_checksums(checksums_path)
    expected_checksums = archive_names | {verification_path.name}
    if set(checksums) != expected_checksums:
        raise TransferPackageError(f"Stage checksum inventory differs: {stage}")
    for name, digest in checksums.items():
        if file_sha256(bundle_root / name) != digest:
            raise TransferPackageError(f"Stage checksum validation failed: {stage}:{name}")
    expected_files = archive_names | {verification_path.name, checksums_path.name}
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise TransferPackageError(f"Stage bundle contains unexpected files: {stage}")
    if any(path.is_symlink() for path in bundle_root.rglob("*")):
        raise TransferPackageError(f"Stage bundle contains a symbolic link: {stage}")
    return record


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = path.expanduser().resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TransferPackageError("Transfer manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise TransferPackageError("Transfer manifest must contain one JSON object")
    _require_exact_keys(manifest, TOP_LEVEL_KEYS, "transfer manifest")
    if manifest.get("schema_version") != 1:
        raise TransferPackageError("Transfer manifest schema version must be 1")
    if manifest.get("package_name") != PACKAGE_NAME:
        raise TransferPackageError(f"Package name must be {PACKAGE_NAME}")
    root_values = manifest.get("source_roots")
    if not isinstance(root_values, dict) or not root_values:
        raise TransferPackageError("Transfer manifest must declare source roots")
    roots: dict[str, Path] = {}
    for name, value in root_values.items():
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", str(name)) is None:
            raise TransferPackageError(f"Invalid source root name: {name}")
        roots[str(name)] = _resolve_root(
            str(value), manifest_path.parent, name=str(name)
        )
    return manifest, roots


def _validate_artifact_manifests(
    repository_root: Path,
    expected_manifest_sha256s: Mapping[str, str],
) -> None:
    manifests = sorted((repository_root / "reports").rglob("artifact_manifest.json"))
    if len(manifests) != len(expected_manifest_sha256s):
        raise TransferPackageError(
            f"Expected {len(expected_manifest_sha256s)} repository artifact manifests, found {len(manifests)}"
        )
    actual_manifest_sha256s: dict[str, Path] = {}
    for manifest_path in manifests:
        manifest_sha256 = file_sha256(manifest_path)
        if manifest_sha256 in actual_manifest_sha256s:
            raise TransferPackageError("Repository artifact manifests are duplicated")
        actual_manifest_sha256s[manifest_sha256] = manifest_path
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = record.get("artifacts")
        if (
            record.get("schema_version") != 1
            or not isinstance(rows, list)
            or int(record.get("artifact_count", -1)) != len(rows)
        ):
            raise TransferPackageError(f"Artifact manifest is invalid: {manifest_path}")
        if record.get("manifest_content_sha256") != _artifact_rows_sha256(rows):
            raise TransferPackageError(
                f"Artifact manifest content hash differs: {manifest_path}"
            )
        declared: set[str] = set()
        for row in rows:
            relative = _safe_relative(str(row.get("path", "")), label="artifact path")
            name = relative.as_posix()
            if name in declared:
                raise TransferPackageError(f"Artifact manifest repeats a path: {manifest_path}")
            declared.add(name)
            artifact = manifest_path.parent / relative
            if artifact.is_symlink() or not artifact.is_file():
                raise TransferPackageError(f"Declared report artifact is missing: {artifact}")
            if artifact.stat().st_size != int(row.get("size_bytes", -1)):
                raise TransferPackageError(f"Report artifact size differs: {artifact}")
            if file_sha256(artifact) != str(row.get("sha256", "")):
                raise TransferPackageError(f"Report artifact hash differs: {artifact}")
        actual = {
            path.relative_to(manifest_path.parent).as_posix()
            for path in manifest_path.parent.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if actual != declared:
            raise TransferPackageError(f"Report artifact inventory differs: {manifest_path}")
    expected_hashes = set(expected_manifest_sha256s.values())
    if set(actual_manifest_sha256s) != expected_hashes:
        raise TransferPackageError(
            "Repository evidence does not match the verified stage safe manifests"
        )


def _validate_executed_notebook(path: Path) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransferPackageError(f"Executed notebook is not valid JSON: {path.name}") from error
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if not code_cells or not any(
        cell.get("execution_count") is not None or bool(cell.get("outputs"))
        for cell in code_cells
    ):
        raise TransferPackageError(
            f"Review-only notebook has no saved execution evidence: {path.name}"
        )
    if any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
        if isinstance(output, dict)
    ):
        raise TransferPackageError(
            f"Review-only notebook contains a saved execution error: {path.name}"
        )


def _write_checksums(package_root: Path) -> Path:
    checksum_path = package_root / "CHECKSUMS.sha256"
    files = sorted(
        (
            path
            for path in package_root.rglob("*")
            if path.is_file() and path != checksum_path
        ),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    checksum_path.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(package_root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return checksum_path


def verify_transfer_tree(package_root: Path) -> None:
    package_root = package_root.expanduser().resolve(strict=True)
    expected_top = {
        "repository-files",
        "review-only",
        "LOCAL_UPLOAD_STEPS.md",
        "CHECKSUMS.sha256",
    }
    if {path.name for path in package_root.iterdir()} != expected_top:
        raise TransferPackageError("Transfer package top-level inventory differs")
    if any(path.is_symlink() for path in package_root.rglob("*")):
        raise TransferPackageError("Transfer package contains a symbolic link")
    checksums = _parse_checksums(package_root / "CHECKSUMS.sha256")
    actual_files = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if set(checksums) != actual_files:
        raise TransferPackageError("Transfer checksum inventory differs")
    for name, digest in checksums.items():
        if file_sha256(package_root / name) != digest:
            raise TransferPackageError(f"Transfer checksum failed: {name}")
    repository = package_root / "repository-files"
    clean_notebooks = {path.name for path in repository.rglob("*.ipynb")}
    if not REQUIRED_NOTEBOOKS.issubset(clean_notebooks):
        raise TransferPackageError("Repository-files is missing a required clean notebook")
    for notebook in repository.rglob("*.ipynb"):
        _validate_clean_notebook(notebook)
    executed_root = package_root / "review-only" / "executed-notebooks"
    required_executed_notebooks = {
        executed_root / notebook_name for notebook_name in REQUIRED_NOTEBOOKS
    }
    if not all(path.is_file() for path in required_executed_notebooks):
        raise TransferPackageError(
            "Review-only is missing a required notebook under executed-notebooks"
        )
    for notebook in required_executed_notebooks:
        _validate_executed_notebook(notebook)
    verification_note = package_root / "review-only" / "VERIFICATION.md"
    if not verification_note.is_file():
        raise TransferPackageError("Review-only is missing VERIFICATION.md")
    for stage in REQUIRED_STAGES:
        stage_review = package_root / "review-only" / "stage-verification" / stage
        required = {
            stage_review / f"{stage}-verification.json",
            stage_review / "SHA256SUMS",
        }
        if not all(path.is_file() for path in required):
            raise TransferPackageError(
                f"Review-only is missing the stage verification record: {stage}"
            )
    for path in repository.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository)
        _validate_repository_destination(relative, path)
    for path in (package_root / "review-only").rglob("*"):
        if path.is_file():
            _validate_review_destination(path.relative_to(package_root / "review-only"), path)
    stage_records = {
        stage: json.loads(
            (
                package_root
                / "review-only"
                / "stage-verification"
                / stage
                / f"{stage}-verification.json"
            ).read_text(encoding="utf-8")
        )
        for stage in REQUIRED_STAGES
    }
    _validate_artifact_manifests(
        repository,
        {
            stage: str(record["safe_manifest_sha256"])
            for stage, record in stage_records.items()
        },
    )


def _write_deterministic_zip(archive_path: Path, package_root: Path) -> None:
    temporary = archive_path.with_suffix(".zip.incomplete")
    files = sorted(
        (path for path in package_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(package_root.parent).as_posix(),
    )
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in files:
            name = path.relative_to(package_root.parent).as_posix()
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, archive_path)
    _verify_zip(archive_path)


def build_transfer_package(*, manifest_path: Path, output_parent: Path) -> dict[str, Any]:
    manifest, roots = _load_manifest(manifest_path)
    output_parent = output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    package_root = output_parent / PACKAGE_NAME
    archive_path = output_parent / f"{PACKAGE_NAME}.zip"
    if package_root.exists() or archive_path.exists():
        raise FileExistsError("Week 8 transfer output already exists and will not be overwritten")

    stage_entries = manifest.get("required_stages")
    if not isinstance(stage_entries, list):
        raise TransferPackageError("required_stages must be a list")
    found_stages: set[str] = set()
    verified_stage_files: dict[str, tuple[Path, Path]] = {}
    verified_manifest_sha256s: dict[str, str] = {}
    for index, raw_entry in enumerate(stage_entries):
        if not isinstance(raw_entry, dict):
            raise TransferPackageError(f"Stage entry {index} must be an object")
        _require_exact_keys(raw_entry, STAGE_ENTRY_KEYS, f"stage entry {index}")
        stage = str(raw_entry["stage"])
        if stage in found_stages:
            raise TransferPackageError(f"Required stage is duplicated: {stage}")
        root_name = str(raw_entry["root"])
        if root_name not in roots:
            raise TransferPackageError(f"Stage uses an unknown source root: {root_name}")
        bundle_relative = _safe_relative(
            str(raw_entry["bundle_root"]), label="stage bundle root"
        )
        stage_record = _verify_stage_bundle(
            stage=stage,
            root=roots[root_name],
            bundle_relative=bundle_relative,
        )
        bundle_root = roots[root_name] / bundle_relative
        verified_stage_files[stage] = (
            bundle_root / f"{stage}-verification.json",
            bundle_root / "SHA256SUMS",
        )
        verified_manifest_sha256s[stage] = str(stage_record["safe_manifest_sha256"])
        found_stages.add(stage)
    if found_stages != REQUIRED_STAGES:
        raise TransferPackageError(
            f"Required stage set differs: expected {sorted(REQUIRED_STAGES)}, found {sorted(found_stages)}"
        )

    repository_entries = manifest.get("repository_files")
    review_entries = manifest.get("review_files")
    if not isinstance(repository_entries, list) or not repository_entries:
        raise TransferPackageError("repository_files must be a non-empty list")
    if not isinstance(review_entries, list) or not review_entries:
        raise TransferPackageError("review_files must be a non-empty list")
    instructions = manifest.get("instructions")
    if not isinstance(instructions, dict):
        raise TransferPackageError("instructions must be an object")
    _require_exact_keys(instructions, INSTRUCTION_KEYS, "instructions")

    temporary_root = output_parent / f".{PACKAGE_NAME}.incomplete"
    if temporary_root.exists():
        raise FileExistsError("Incomplete Week 8 transfer directory already exists")
    temporary_root.mkdir()
    try:
        repository_root = temporary_root / "repository-files"
        review_root = temporary_root / "review-only"
        repository_root.mkdir()
        review_root.mkdir()
        seen_sources: set[tuple[str, str]] = set()
        seen_destinations: set[tuple[str, str]] = set()

        for group, entries, target_root in (
            ("repository-files", repository_entries, repository_root),
            ("review-only", review_entries, review_root),
        ):
            for index, raw_entry in enumerate(entries):
                if not isinstance(raw_entry, dict):
                    raise TransferPackageError(f"{group} entry {index} must be an object")
                source, destination = _resolve_source(
                    roots, raw_entry, label=f"{group} entry {index}"
                )
                source_key = (str(raw_entry["root"]), str(raw_entry["source"]))
                destination_key = (group, destination.as_posix().casefold())
                if source_key in seen_sources:
                    raise TransferPackageError(f"Transfer source is duplicated: {source_key}")
                if destination_key in seen_destinations:
                    raise TransferPackageError(f"Transfer destination is duplicated: {destination}")
                seen_sources.add(source_key)
                seen_destinations.add(destination_key)
                if group == "repository-files":
                    _validate_repository_destination(destination, source)
                    if source.suffix.lower() == ".ipynb":
                        _validate_clean_notebook(source)
                else:
                    _validate_review_destination(destination, source)
                target = target_root / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

        instruction_root_name = str(instructions["root"])
        if instruction_root_name not in roots:
            raise TransferPackageError("Instructions use an unknown source root")
        instruction_relative = _safe_relative(
            str(instructions["source"]), label="instructions source"
        )
        instruction_root = roots[instruction_root_name]
        if _path_has_symlink(instruction_root, instruction_relative):
            raise TransferPackageError("Instructions source contains a symbolic link")
        instruction_source = instruction_root / instruction_relative
        if not instruction_source.is_file() or instruction_source.suffix.lower() != ".md":
            raise TransferPackageError("Instructions source must be one Markdown file")
        _validate_no_secret_or_local_path(instruction_source)
        instruction_text = instruction_source.read_text(encoding="utf-8")
        required_phrases = (
            "Copy only `repository-files/`",
            "Do not copy `review-only/`",
            "Do not add a co-author line",
        )
        if not all(phrase in instruction_text for phrase in required_phrases):
            raise TransferPackageError("Upload instructions do not contain the required boundaries")
        shutil.copyfile(instruction_source, temporary_root / "LOCAL_UPLOAD_STEPS.md")

        for stage, (verification_source, checksums_source) in verified_stage_files.items():
            review_stage = review_root / "stage-verification" / stage
            expected = {
                review_stage / verification_source.name: verification_source,
                review_stage / checksums_source.name: checksums_source,
            }
            for copied, source in expected.items():
                if not copied.is_file() or file_sha256(copied) != file_sha256(source):
                    raise TransferPackageError(
                        f"Review-only stage verification does not match its source: {stage}"
                    )

        _validate_artifact_manifests(repository_root, verified_manifest_sha256s)
        _write_checksums(temporary_root)
        verify_transfer_tree(temporary_root)
        os.replace(temporary_root, package_root)
        _write_deterministic_zip(archive_path, package_root)
        return {
            "schema_version": 1,
            "status": "completed",
            "package": package_root.name,
            "package_file_count": sum(path.is_file() for path in package_root.rglob("*")),
            "zip": archive_path.name,
            "zip_sha256": file_sha256(archive_path),
            "stage_count": len(found_stages),
        }
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print(
        json.dumps(
            build_transfer_package(
                manifest_path=arguments.manifest,
                output_parent=arguments.output_parent,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
