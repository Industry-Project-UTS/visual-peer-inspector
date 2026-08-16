from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile

from .chunks import verify_chunks
from .segments import SegmentedFile


def _safe_name(value: str, label: str) -> str:
    if not value or any(character in value for character in ("/", "\\")):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def _scene_members(
    archive: zipfile.ZipFile,
    scene_id: str,
    view_id: str,
) -> list[str]:
    scene_id = _safe_name(scene_id, "scene id")
    prefix = f"toysad8k/{scene_id}"
    scene2d_name = f"{prefix}/scene2d.metadata.json"
    scene3d_name = f"{prefix}/scene3d.metadata.json"
    scene3d = json.loads(archive.read(scene3d_name))
    objects = scene3d.get("objects", [])
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"No object metadata found for {scene_id}")

    members = [
        f"{prefix}/RGB/{view_id}.png",
        scene2d_name,
        scene3d_name,
    ]
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid object metadata for {scene_id}")
        object_name = _safe_name(str(item.get("name", "")), "object name")
        members.append(
            f"{prefix}/segmentations/{object_name}/{object_name}_{view_id}.png"
        )
    return members


def _write_member(
    archive: zipfile.ZipFile,
    member_name: str,
    output_root: Path,
) -> dict[str, Any]:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {member_name}")

    info = archive.getinfo(member_name)
    if info.is_dir():
        raise ValueError(f"Expected a file: {member_name}")
    if info.file_size > 32 * 1024 * 1024:
        raise ValueError(f"Selected member is unexpectedly large: {member_name}")

    content = archive.read(info)
    output_path = output_root.joinpath(*path.parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(content)
    return {
        "member": member_name,
        "bytes": len(content),
        "crc32": f"{info.CRC:08x}",
        "sha256": sha256(content).hexdigest(),
    }


def extract_cohort(
    chunk_directory: Path,
    chunk_manifest: dict[str, Any],
    cohort: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    quick_check = verify_chunks(
        chunk_directory,
        chunk_manifest,
        check_hashes=False,
    )
    if not quick_check.ok:
        raise ValueError("Archive chunks do not match the manifest")

    output = output.expanduser().resolve()
    staging = output.parent / f".{output.name}.partial"
    if output.exists() or staging.exists():
        raise FileExistsError(f"Output already exists: {output} or {staging}")
    staging.mkdir(parents=True)

    view_id = str(cohort["view_id"])
    if len(view_id) != 4 or not view_id.isdigit():
        raise ValueError(f"Invalid view id: {view_id!r}")
    scene_ids = [
        str(item["scene_id"]) if isinstance(item, dict) else str(item)
        for item in cohort["scenes"]
    ]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("The cohort contains duplicate scene ids")

    chunk_paths = [
        chunk_directory / str(item["name"])
        for item in chunk_manifest["files"]
    ]
    expected_sizes = [
        int(item["size_bytes"])
        for item in chunk_manifest["files"]
    ]
    extracted = []
    with SegmentedFile(chunk_paths, expected_sizes=expected_sizes) as stream:
        with zipfile.ZipFile(stream) as archive:
            for scene_id in scene_ids:
                for member_name in _scene_members(archive, scene_id, view_id):
                    extracted.append(_write_member(archive, member_name, staging))

    report = {
        "schema_version": 1,
        "dataset": str(cohort["dataset"]),
        "dataset_revision": str(cohort["dataset_revision"]),
        "cohort_id": str(cohort["id"]),
        "split": str(cohort["split"]),
        "view_id": view_id,
        "scene_count": len(scene_ids),
        "file_count": len(extracted),
        "total_bytes": sum(int(item["bytes"]) for item in extracted),
        "files": extracted,
    }
    with (staging / "extraction.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    staging.replace(output)
    return report
