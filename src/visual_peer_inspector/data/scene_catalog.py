"""Build a compact scene/label catalogue from ToysAD metadata members."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import BinaryIO, Callable

from .archive_members import IndexedMember, read_indexed_member


SCENE_METADATA_SUFFIX = "/scene2d.metadata.json"
SCENE3D_METADATA_SUFFIX = "/scene3d.metadata.json"
SCENE_ID_PATTERN = re.compile(r"^(?P<category>.+)_\d{3}-\d{3}$")


def anomaly_variant(blend_path: str) -> str | None:
    """Return the raw hyphenated defect variant used by the official loader."""

    stem = Path(blend_path).stem
    if "-" not in stem:
        return None
    return stem.split("-", 1)[1]


def scene2d_anomaly_flags(labels: list[int]) -> list[int]:
    """Convert released scene2d labels (0=anomaly) to binary anomaly flags."""

    if any(label not in (0, 1) for label in labels):
        raise ValueError("scene2d labels must be binary")
    return [int(label == 0) for label in labels]


def scene_id_from_member(name: str, suffix: str) -> str:
    if not name.startswith("toysad8k/") or not name.endswith(suffix):
        raise ValueError(f"Unexpected ToysAD metadata path: {name}")
    return name[len("toysad8k/") : -len(suffix)]


def load_split_mapping(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Official split file must contain an object")

    mapping: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for split_name, scene_ids in raw.items():
        if not isinstance(scene_ids, list):
            raise ValueError(f"Split {split_name!r} must contain a list")
        for scene_id in scene_ids:
            scene_id = str(scene_id)
            if scene_id in mapping:
                duplicates.setdefault(scene_id, [mapping[scene_id]]).append(split_name)
            else:
                mapping[scene_id] = str(split_name)
    if duplicates:
        raise ValueError(f"Scenes occur in multiple official splits: {duplicates}")
    return mapping, {str(key): [str(item) for item in value] for key, value in raw.items()}


def build_scene_catalog(
    archive_stream: BinaryIO,
    archive_database: Path,
    split_path: Path,
    output_database: Path,
    summary_path: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Read scene2d JSON members with CRC checks and build a scene catalogue."""

    output_database = output_database.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    for output in (output_database, summary_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    split_mapping, raw_splits = load_split_mapping(split_path)
    with sqlite3.connect(archive_database) as archive_index:
        metadata_rows = archive_index.execute(
            "SELECT name, compressed_size, uncompressed_size, crc32, "
            "compression_method, flags, local_header_offset, is_directory, "
            "is_symlink, unsafe_reason FROM members "
            "WHERE name LIKE ? ORDER BY local_header_offset",
            (f"%{SCENE_METADATA_SUFFIX}",),
        ).fetchall()
        scene3d_names = [
            row[0]
            for row in archive_index.execute(
                "SELECT name FROM members WHERE name LIKE ?",
                (f"%{SCENE3D_METADATA_SUFFIX}",),
            )
        ]

    scene3d_ids = {
        scene_id_from_member(str(name), SCENE3D_METADATA_SUFFIX)
        for name in scene3d_names
    }
    scene2d_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    anomaly_counts: Counter[int] = Counter()
    split_anomaly_counts: Counter[tuple[str, int]] = Counter()
    categories: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    label_path_mismatch_count = 0
    label_path_mismatches: list[dict[str, object]] = []

    temporary_database = output_database.with_suffix(output_database.suffix + ".tmp")
    if temporary_database.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_database}")
    connection = sqlite3.connect(temporary_database)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE scenes (
                scene_id TEXT PRIMARY KEY,
                split TEXT,
                category TEXT NOT NULL,
                object_count INTEGER NOT NULL,
                anomaly_count INTEGER NOT NULL,
                anomaly_labels_json TEXT NOT NULL,
                blend_paths_json TEXT NOT NULL,
                anomaly_variants_json TEXT NOT NULL,
                metadata_member TEXT NOT NULL
            );
            CREATE TABLE objects (
                scene_id TEXT NOT NULL,
                object_index INTEGER NOT NULL,
                is_anomaly INTEGER NOT NULL,
                blend_path TEXT NOT NULL,
                anomaly_variant TEXT,
                PRIMARY KEY (scene_id, object_index),
                FOREIGN KEY (scene_id) REFERENCES scenes(scene_id)
            );
            """
        )
        connection.execute("BEGIN")
        for index, row in enumerate(metadata_rows):
            member = IndexedMember(
                name=str(row[0]),
                compressed_size=int(row[1]),
                uncompressed_size=int(row[2]),
                crc32=int(row[3]),
                compression_method=int(row[4]),
                flags=int(row[5]),
                local_header_offset=int(row[6]),
                is_directory=bool(row[7]),
                is_symlink=bool(row[8]),
                unsafe_reason=row[9],
            )
            metadata = json.loads(read_indexed_member(archive_stream, member))
            labels = [int(label) for label in metadata.get("anomaly_labels", [])]
            blend_paths = [str(path) for path in metadata.get("blend_paths", [])]
            if len(labels) != len(blend_paths) or not labels:
                raise ValueError(
                    f"Invalid label/path arrays in metadata member {member.name}"
                )
            try:
                anomaly_flags = scene2d_anomaly_flags(labels)
            except ValueError as error:
                raise ValueError(
                    f"Non-binary scene2d label in {member.name}"
                ) from error
            scene_id = scene_id_from_member(member.name, SCENE_METADATA_SUFFIX)
            if scene_id in scene2d_ids:
                raise ValueError(f"Duplicate scene metadata: {scene_id}")
            scene2d_ids.add(scene_id)
            category_match = SCENE_ID_PATTERN.match(scene_id)
            category = category_match.group("category") if category_match else "[unknown]"
            split_name = split_mapping.get(scene_id)
            raw_variants = [anomaly_variant(path) for path in blend_paths]
            # The released scene2d metadata uses 0 for an anomalous object and
            # 1 for a normal peer. The official 3D loader separately rebuilds
            # targets from object paths and uses 1 for a hyphenated defect.
            labelled_variants = [
                variant
                for is_anomaly, variant in zip(
                    anomaly_flags, raw_variants, strict=True
                )
                if is_anomaly
            ]
            path_labels = [int(variant is not None) for variant in raw_variants]
            if anomaly_flags != path_labels:
                label_path_mismatch_count += 1
                if len(label_path_mismatches) < 100:
                    label_path_mismatches.append(
                        {
                            "scene_id": scene_id,
                            "scene2d_labels": labels,
                            "derived_anomaly_flags": anomaly_flags,
                            "path_anomaly_flags": path_labels,
                        }
                    )

            anomaly_count = sum(anomaly_flags)
            split_key = split_name or "[not-in-official-split]"
            split_counts[split_key] += 1
            anomaly_counts[anomaly_count] += 1
            split_anomaly_counts[(split_key, anomaly_count)] += 1
            categories[category] += 1
            variants.update(variant for variant in labelled_variants if variant)

            connection.execute(
                "INSERT INTO scenes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scene_id,
                    split_name,
                    category,
                    len(labels),
                    anomaly_count,
                    json.dumps(labels, separators=(",", ":")),
                    json.dumps(blend_paths, separators=(",", ":")),
                    json.dumps(labelled_variants, separators=(",", ":")),
                    member.name,
                ),
            )
            connection.executemany(
                "INSERT INTO objects VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        scene_id,
                        object_index,
                        is_anomaly,
                        blend_path,
                        variant if is_anomaly else None,
                    )
                    for object_index, (is_anomaly, blend_path, variant) in enumerate(
                        zip(anomaly_flags, blend_paths, raw_variants, strict=True)
                    )
                ],
            )
            if progress and (index + 1) % 500 == 0:
                progress(index + 1, len(metadata_rows))

        connection.execute("CREATE INDEX scenes_split_idx ON scenes(split)")
        connection.execute(
            "CREATE INDEX scenes_anomaly_count_idx ON scenes(anomaly_count)"
        )
        connection.execute("CREATE INDEX scenes_category_idx ON scenes(category)")
        connection.execute(
            "CREATE INDEX objects_variant_idx ON objects(anomaly_variant)"
        )
        connection.commit()

        split_ids = set(split_mapping)
        summary: dict[str, object] = {
            "scene2d_metadata_count": len(scene2d_ids),
            "scene3d_metadata_count": len(scene3d_ids),
            "official_split_scene_count": len(split_ids),
            "official_split_declared_counts": {
                key: len(value) for key, value in raw_splits.items()
            },
            "catalogue_split_counts": dict(split_counts.most_common()),
            "anomaly_count_distribution": {
                str(key): value for key, value in sorted(anomaly_counts.items())
            },
            "split_anomaly_count_distribution": {
                f"{split_name}|{anomaly_count}": value
                for (split_name, anomaly_count), value in sorted(
                    split_anomaly_counts.items()
                )
            },
            "category_count": len(categories),
            "categories": dict(categories.most_common()),
            "anomaly_variants": dict(variants.most_common()),
            "scene3d_without_scene2d": sorted(scene3d_ids - scene2d_ids),
            "scene2d_without_scene3d": sorted(scene2d_ids - scene3d_ids),
            "archive_scenes_not_in_official_split": sorted(scene2d_ids - split_ids),
            "official_split_scenes_missing_scene2d": sorted(split_ids - scene2d_ids),
            "scene2d_label_convention": "0=anomaly, 1=normal",
            "label_path_mismatch_count": label_path_mismatch_count,
            "label_path_mismatch_samples": label_path_mismatches,
        }
        temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        with temporary_summary.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_database.replace(output_database)
        temporary_summary.replace(summary_path)
        return summary
    except Exception:
        connection.rollback()
        if temporary_database.exists():
            temporary_database.unlink()
        raise
    finally:
        connection.close()
