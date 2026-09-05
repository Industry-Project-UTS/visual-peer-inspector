#!/usr/bin/env python3
"""Create compact, checksum-verified archives for one Kaggle stage.

Large stage outputs should be written below ``/kaggle/temp``. This script then
places a small number of deterministic archives in ``/kaggle/working`` so the
Kaggle CLI does not need to enumerate thousands of individual result files.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence
import zipfile


ALLOWED_SAFE_SUFFIXES = frozenset({".csv", ".json", ".png"})
REVIEW_SUFFIXES = frozenset({".csv", ".ipynb", ".json", ".log", ".md", ".png", ".txt"})
FORBIDDEN_NAMES = frozenset({
    ".env",
    "clearml.conf",
    "credentials.json",
    "kaggle.json",
    "secrets.json",
})
FORBIDDEN_PARTS = frozenset({"credential", "credentials", "secret", "secrets"})
FORBIDDEN_SAFE_TEXT = re.compile(
    r"S\d{6}|O\d{2}|opaque_scene_id|labels_by_object|mask_rle|box_xyxy|"
    r"/kaggle/input/|/(?:Users|home)/|CLEARML_API_(ACCESS|SECRET)_KEY|HF_TOKEN|"
    r"source_scene_id",
    re.IGNORECASE,
)
STAGE_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_kaggle_roots(input_root: Path) -> list[Path]:
    """Return only known Kaggle mount shapes without a recursive walk."""

    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        return []
    candidates: set[Path] = set()
    for pattern in ("*", "datasets/*/*", "datasets/*/*/versions/*"):
        candidates.update(path.resolve() for path in input_root.glob(pattern) if path.is_dir())
    return sorted(candidates, key=lambda path: path.as_posix())


def find_bounded_marker_root(
    input_root: Path,
    marker: str,
    *,
    expected_names: Sequence[str] = (),
    maximum_children: int = 200,
) -> Path:
    """Find a mounted package marker while avoiding the full archive tree."""

    marker_path = Path(marker)
    if marker_path.is_absolute() or ".." in marker_path.parts:
        raise ValueError("Marker must be a safe relative path")
    roots = bounded_kaggle_roots(input_root)
    preferred = {name for name in expected_names if name}
    ordered = sorted(roots, key=lambda path: (path.name not in preferred, path.as_posix()))
    checked: set[Path] = set()
    for root in ordered:
        direct = root / marker_path
        if direct.is_file():
            return root
        checked.add(root)
        children = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
        if len(children) > maximum_children:
            continue
        for child in children:
            if child in checked:
                continue
            if (child / marker_path).is_file():
                return child.resolve()
    raise FileNotFoundError(f"Could not find bounded Kaggle marker: {marker}")


def _safe_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return relative


def _reject_protected_path(relative: Path) -> None:
    lowered_parts = tuple(part.lower() for part in relative.parts)
    if relative.name.lower() in FORBIDDEN_NAMES or any(
        marker in part for part in lowered_parts for marker in FORBIDDEN_PARTS
    ):
        raise ValueError(f"Protected file cannot enter a Kaggle bundle: {relative}")


def _files_for_paths(root: Path, values: Sequence[str], *, allowed_suffixes: frozenset[str] | None) -> list[Path]:
    files: set[Path] = set()
    for value in values:
        relative = _safe_relative(value)
        _reject_protected_path(relative)
        source = root / relative
        if source.is_symlink() or not source.exists():
            raise FileNotFoundError(f"Bundle input is missing or unsafe: {relative}")
        candidates: Iterable[Path]
        candidates = (source,) if source.is_file() else source.rglob("*")
        for path in candidates:
            if path.is_symlink():
                raise ValueError(f"Bundle input contains a symbolic link: {path}")
            if not path.is_file():
                continue
            child_relative = path.relative_to(root)
            _reject_protected_path(child_relative)
            if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
                raise ValueError(f"Bundle input has an unsupported file type: {child_relative}")
            files.add(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def validate_safe_root(safe_root: Path) -> list[Path]:
    """Validate aggregate evidence and its complete artifact manifest."""

    safe_root = safe_root.expanduser().resolve(strict=True)
    if safe_root.is_symlink():
        raise ValueError("Safe evidence root cannot be a symbolic link")
    files = sorted(path for path in safe_root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in safe_root.rglob("*")):
        raise ValueError("Safe evidence cannot contain symbolic links")
    relative_files = {path.relative_to(safe_root).as_posix() for path in files}
    required = {"artifact_manifest.json", "run_record.json"}
    if not required.issubset(relative_files):
        raise ValueError("Safe evidence is missing its manifest or run record")
    for path in files:
        relative = path.relative_to(safe_root)
        _reject_protected_path(relative)
        if path.suffix.lower() not in ALLOWED_SAFE_SUFFIXES:
            raise ValueError(f"Safe evidence has an unsupported file type: {relative}")
        if path.suffix.lower() in {".csv", ".json"} and path.name != "artifact_manifest.json":
            if FORBIDDEN_SAFE_TEXT.search(path.read_text(encoding="utf-8")):
                raise ValueError(f"Protected detail found in safe evidence: {relative}")

    manifest = json.loads((safe_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise ValueError("Safe artifact manifest is invalid")
    declared = {str(row.get("path")) for row in rows}
    if declared != relative_files - {"artifact_manifest.json"}:
        raise ValueError("Safe artifact manifest coverage differs")
    for row in rows:
        path = safe_root / _safe_relative(str(row["path"]))
        if path.stat().st_size != int(row["size_bytes"]) or file_sha256(path) != str(row["sha256"]):
            raise ValueError(f"Safe artifact identity differs: {row['path']}")
    run_record = json.loads((safe_root / "run_record.json").read_text(encoding="utf-8"))
    if run_record.get("status") != "completed":
        raise ValueError("Safe run record is not complete")
    return files


def _write_deterministic_zip(archive_path: Path, root: Path, files: Sequence[Path], *, prefix: str = "") -> None:
    temporary = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive_name = f"{prefix.rstrip('/')}/{relative}" if prefix else relative
            info = zipfile.ZipInfo(archive_name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, archive_path)


def build_bundles(
    *,
    stage_root: Path,
    output_root: Path,
    stage_name: str,
    review_paths: Sequence[str] = (),
    downstream_paths: Sequence[str] = (),
) -> dict[str, object]:
    """Build compact stage archives and return their verification record."""

    if STAGE_NAME.fullmatch(stage_name) is None:
        raise ValueError("Stage name must contain lowercase letters, numbers and hyphens")
    stage_root = stage_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if output_root == stage_root or stage_root in output_root.parents:
        raise ValueError("Bundle output must be outside the stage output tree")
    output_root.mkdir(parents=True, exist_ok=True)

    safe_root = stage_root / "safe"
    safe_files = validate_safe_root(safe_root)
    review_files = _files_for_paths(stage_root, review_paths, allowed_suffixes=REVIEW_SUFFIXES)
    downstream_files = _files_for_paths(stage_root, downstream_paths, allowed_suffixes=None)

    specifications = [
        (output_root / f"{stage_name}-repository-evidence.zip", safe_root, safe_files, "safe"),
    ]
    if review_files:
        specifications.append((output_root / f"{stage_name}-review-evidence.zip", stage_root, review_files, ""))
    if downstream_files:
        specifications.append((output_root / f"{stage_name}-downstream-input.zip", stage_root, downstream_files, ""))
    verification_path = output_root / f"{stage_name}-verification.json"
    checksums_path = output_root / "SHA256SUMS"
    expected_outputs = [item[0] for item in specifications] + [verification_path, checksums_path]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Bundle outputs already exist: {[path.name for path in existing]}")

    for archive_path, root, files, prefix in specifications:
        _write_deterministic_zip(archive_path, root, files, prefix=prefix)

    archives = []
    for archive_path, _root, files, _prefix in specifications:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Archive integrity failed: {bad_member}")
        archives.append({
            "path": archive_path.name,
            "sha256": file_sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "source_file_count": len(files),
        })
    verification = {
        "schema_version": 1,
        "stage": stage_name,
        "status": "completed",
        "safe_manifest_sha256": file_sha256(safe_root / "artifact_manifest.json"),
        "safe_file_count": len(safe_files),
        "review_file_count": len(review_files),
        "downstream_file_count": len(downstream_files),
        "archives": archives,
    }
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_files = [item[0] for item in specifications] + [verification_path]
    checksums_path.write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in sorted(checksum_files, key=lambda path: path.name)),
        encoding="utf-8",
    )
    return verification


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--review-path", action="append", default=[])
    parser.add_argument("--downstream-path", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    record = build_bundles(
        stage_root=args.stage_root,
        output_root=args.output_root,
        stage_name=args.stage_name,
        review_paths=args.review_path,
        downstream_paths=args.downstream_path,
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
