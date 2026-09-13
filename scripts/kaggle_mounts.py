"""Bounded discovery for small, versioned Kaggle input resources."""

from __future__ import annotations

import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
from zipfile import ZipFile


def bounded_kaggle_roots(input_root: Path) -> list[Path]:
    """Return direct and namespaced Kaggle mounts without scanning their contents."""

    root = Path(input_root)
    if not root.is_dir():
        return []
    candidates = set()
    for pattern in (
        "*",
        "datasets/*/*",
        "datasets/*/*/versions/*",
        "notebooks/*/*",
        "notebooks/*/*/versions/*",
    ):
        candidates.update(path.resolve() for path in root.glob(pattern) if path.is_dir())
    return sorted(candidates, key=lambda path: path.as_posix())


def _candidate_roots(input_root: Path, marker: str) -> list[Path]:
    marker_path = Path(marker)
    if marker_path.is_absolute() or ".." in marker_path.parts:
        raise ValueError("Marker must be a safe relative path")
    matches = []
    for root in bounded_kaggle_roots(input_root):
        if (root / marker_path).is_file():
            matches.append(root)
        children = sorted(path for path in root.iterdir() if path.is_dir())
        if len(children) > 200:
            continue
        matches.extend(child for child in children if (child / marker_path).is_file())
    return sorted(set(path.resolve() for path in matches))


def find_source(input_root: Path) -> Path:
    """Locate exactly one scoring source package without a recursive scan."""

    candidates = [
        root
        for root in _candidate_roots(input_root, "comparison_base.json")
        if (root / "scripts/run_scoring_comparison.py").is_file()
        and (root / "src/visual_peer_inspector").is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one scoring source resource, found {len(candidates)}")
    return candidates[0]


def find_sealed_resource(input_root: Path, *, kind: str, expected_lock_sha256: str | None = None) -> Path:
    """Locate a model or evaluator package from its small lock metadata."""

    candidates = []
    for root in _candidate_roots(input_root, "artifact_lock.json"):
        try:
            record = json.loads((root / "artifact_lock.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("metadata", {}).get("kind") != kind:
            continue
        if expected_lock_sha256 is not None:
            from hashlib import sha256

            digest = sha256((root / "artifact_lock.json").read_bytes()).hexdigest()
            if digest != expected_lock_sha256:
                continue
        candidates.append(root)
    if len(candidates) != 1:
        raise ValueError(f"Expected one {kind} resource, found {len(candidates)}")
    return candidates[0]


def materialize_sealed_resource(
    input_root: Path,
    *,
    kind: str,
    archive_name: str,
    archive_sha256: str,
    destination: Path,
    expected_lock_sha256: str | None = None,
) -> Path:
    """Use an expanded resource or safely unpack one exact private ZIP."""

    try:
        return find_sealed_resource(
            input_root,
            kind=kind,
            expected_lock_sha256=expected_lock_sha256,
        )
    except ValueError as error:
        if "found 0" not in str(error):
            raise
    root = Path(input_root)
    archives = []
    for mounted in bounded_kaggle_roots(root):
        direct = mounted / archive_name
        if direct.is_file():
            archives.append(direct.resolve())
        children = sorted(path for path in mounted.iterdir() if path.is_dir())
        if len(children) <= 200:
            archives.extend(
                (child / archive_name).resolve()
                for child in children
                if (child / archive_name).is_file()
            )
    archives = sorted(set(archives))
    if len(archives) != 1:
        raise ValueError(
            f"Expected one expanded {kind} resource or one {archive_name}, "
            f"found {len(archives)} archives"
        )
    from hashlib import sha256

    archive = archives[0]
    digest = sha256(archive.read_bytes()).hexdigest()
    if digest != archive_sha256:
        raise ValueError(f"The attached {archive_name} hash differs")
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Resource extraction destination is not empty")
    target.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError("An incomplete resource extraction already exists")
    temporary.mkdir(parents=True)
    total = 0
    try:
        with ZipFile(archive) as handle:
            entries = handle.infolist()
            names = [entry.filename for entry in entries]
            if not entries or len(names) != len(set(names)) or handle.testzip() is not None:
                raise ValueError("Resource ZIP is empty, duplicated or corrupt")
            for entry in entries:
                relative = PurePosixPath(entry.filename)
                mode = entry.external_attr >> 16
                if (
                    entry.is_dir()
                    or relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or stat.S_ISLNK(mode)
                ):
                    raise ValueError("Resource ZIP contains an unsafe path")
                total += entry.file_size
                if total > 2 * 1024**3:
                    raise ValueError("Resource ZIP exceeds the declared 2 GiB limit")
                output = temporary.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(entry) as source, output.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    package = find_sealed_resource(
        target,
        kind=kind,
        expected_lock_sha256=expected_lock_sha256,
    )
    return package


def find_prediction(input_root: Path, *, expected_input_lock_sha256: str) -> Path:
    """Locate one completed 08G output bound to the expected model input."""

    candidates = []
    for root in _candidate_roots(input_root, "prediction_lock.json"):
        binding = root / "run_binding.json"
        status = root / "run_status.json"
        if not binding.is_file() or not status.is_file():
            continue
        try:
            binding_record = json.loads(binding.read_text(encoding="utf-8"))
            status_record = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            binding_record.get("input_lock_sha256") == expected_input_lock_sha256
            and status_record.get("status") == "complete"
        ):
            candidates.append(root)
    if len(candidates) != 1:
        raise ValueError(f"Expected one completed prediction resource, found {len(candidates)}")
    return candidates[0]


def find_recovery(input_root: Path) -> tuple[Path, Path] | None:
    """Return one recovery ZIP and sidecar pair, or none when not attached."""

    root = Path(input_root)
    zips = []
    manifests = []
    expanded = []
    archive_patterns = ("*-recovery.zip", "*-recovery.zip.bin")
    for mounted in bounded_kaggle_roots(root):
        for pattern in archive_patterns:
            zips.extend(path.resolve() for path in mounted.glob(pattern) if path.is_file())
        manifests.extend(
            path.resolve()
            for path in mounted.glob("*-recovery-manifest.json")
            if path.is_file()
        )
        expanded.extend(
            path.resolve()
            for path in mounted.glob("*-recovery")
            if path.is_dir()
        )
        children = sorted(path for path in mounted.iterdir() if path.is_dir())
        if len(children) <= 200:
            for child in children:
                for pattern in archive_patterns:
                    zips.extend(path.resolve() for path in child.glob(pattern) if path.is_file())
                manifests.extend(
                    path.resolve()
                    for path in child.glob("*-recovery-manifest.json")
                    if path.is_file()
                )
                expanded.extend(
                    path.resolve()
                    for path in child.glob("*-recovery")
                    if path.is_dir()
                )
    zips = sorted(set(zips))
    manifests = sorted(set(manifests))
    expanded = sorted(set(expanded))
    if expanded:
        raise ValueError(
            "A recovery resource was expanded by Kaggle. Attach the verified archive as "
            ".zip.bin so its sealed bytes are preserved"
        )
    pairs = []
    paired_manifests = set()
    for archive in zips:
        suffix = "-recovery.zip.bin" if archive.name.endswith("-recovery.zip.bin") else "-recovery.zip"
        stem = archive.name.removesuffix(suffix)
        sidecar = archive.with_name(f"{stem}-recovery-manifest.json")
        if sidecar.is_file():
            pairs.append((archive, sidecar))
            paired_manifests.add(sidecar.resolve())
    if zips and len(pairs) != len(zips):
        raise ValueError("A recovery archive is attached without its manifest")
    if manifests and paired_manifests != set(manifests):
        raise ValueError("A recovery manifest is attached without a sealed archive")
    if len(pairs) > 1:
        raise ValueError("More than one recovery package is attached")
    return pairs[0] if pairs else None


__all__ = [
    "bounded_kaggle_roots",
    "find_prediction",
    "find_recovery",
    "find_sealed_resource",
    "find_source",
    "materialize_sealed_resource",
]
