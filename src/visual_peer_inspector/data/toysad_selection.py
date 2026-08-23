"""Build and audit the frozen ToysAD-8K 2,000-scene experiment corpus.

The selector works from metadata and the archive index. It never extracts the
full archive and it never reads model predictions. Related renders are grouped
by their base asset before development, calibration and lock roles are formed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import platform
import sqlite3
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .archive_members import IndexedMember, read_indexed_member


RELEASE_ID = "toysad8k-train-2000-v2"
SELECTION_SEED = "toysad8k-train-2000-v2|20260812-lock-reset"
ROLES = ("development", "calibration", "internal_lock")
PILOT_STAGES = ("screen_100", "confirm_300", "remainder_600")
STRATA = ((3, 1), (4, 1), (4, 2), (5, 1), (5, 2))
RAW_ANOMALY_PREFIXES = (
    "bump",
    "crack",
    "defr",
    "disc",
    "frac",
    "miss",
    "rota",
    "tran",
)

ROLE_STRATUM_QUOTAS: dict[str, dict[tuple[int, int], int]] = {
    "development": dict(zip(STRATA, (307, 185, 206, 158, 144), strict=True)),
    "calibration": dict(zip(STRATA, (296, 129, 56, 85, 34), strict=True)),
    "internal_lock": dict(zip(STRATA, (197, 86, 38, 57, 22), strict=True)),
}
ROLE_ASSET_QUOTAS = {
    "development": 273,
    "calibration": 163,
    "internal_lock": 109,
}
ROLE_FOUR_SCENE_ASSET_QUOTAS = {
    "development": 181,
    "calibration": 111,
    "internal_lock": 73,
}
PILOT_STRATUM_QUOTAS: dict[str, dict[tuple[int, int], int]] = {
    "screen_100": dict(zip(STRATA, (40, 20, 15, 15, 10), strict=True)),
    "confirm_300": dict(zip(STRATA, (120, 60, 45, 45, 30), strict=True)),
    "remainder_600": dict(zip(STRATA, (147, 105, 146, 98, 104), strict=True)),
}
PILOT_ASSET_QUOTAS = {
    "screen_100": 27,
    "confirm_300": 82,
    "remainder_600": 164,
}
PILOT_FOUR_SCENE_ASSET_QUOTAS = {
    "screen_100": 19,
    "confirm_300": 54,
    "remainder_600": 108,
}

EXPECTED_INPUT_HASHES = {
    "master_json": "6a2cbc575365208d72a4509d25fde9565e24a3f68341d38535ce67e13d9b9f24",
    "catalogue": "4745aa61ea223aada35a2b72803315d30f8d474913056ca9b80ad7bcb4b2475a",
    "archive_index": "c502d779f9b9e78ed7e2766402b7ba10337b1ee43fa467da03b6f997d6bc7e36",
    "readiness_audit": "81b781739e98c40c722bf68c044a47cfb387c4fd10d159d6b9b8e1101497da53",
    "official_split": "62e4bfc9e98bbb5e3083df1f4a361642144084e355c2619d76cdc075ff9b5dac",
}
EXPECTED_CANONICAL_SCENES_HASH = (
    "b174bdf0156542b6cef1604318b079fd1d9a5f9c1d05fa48f387b628085cd092"
)
EXPECTED_ARCHIVE_SIZE_BYTES = 52_993_751_003
EXPECTED_ARCHIVE_SHA256 = (
    "e49c30857b96fdede3f374c773ce0758174f225958ba29c9ad5ca5b075dfc733"
)
EXPECTED_POOL_COUNTS = {
    "scenes": 4_107,
    "assets": 545,
    "categories": 39,
}
EXPECTED_POOL_STRATA = dict(zip(STRATA, (2_025, 883, 386, 582, 231), strict=True))
EXPECTED_SELECTED_STRATA = dict(zip(STRATA, (800, 400, 300, 300, 200), strict=True))
EXPECTED_OFFICIAL_IDS_MISSING_FROM_MASTER = ("cup_000-009", "hammer_004-006")
EXPECTED_MASTER_IDS_OUTSIDE_OFFICIAL_SPLIT = (
    "bottle_019-008",
    "glass_061-007",
    "hammer_083-003",
    "penguin_024-001",
    "robot_042-001",
    "robot_073-014",
)


@dataclass(frozen=True)
class SourcePaths:
    """Pinned files used to construct one selection release."""

    master_json: Path
    catalogue: Path
    readiness_audit: Path
    archive_index: Path
    archive_zip: Path
    archive_manifest: Path
    official_split: Path


def calculate_file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest for one file."""

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stable_fraction(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    integer = int.from_bytes(sha256(payload).digest()[:8], "big")
    return (integer + 1) / ((1 << 64) + 1)


def anomaly_prefixes(scene: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the sorted raw anomaly prefixes present in one scene."""

    prefixes = {
        str(variant).split("-", 1)[0]
        for variant in scene.get("anomaly_variants", [])
        if variant
    }
    return tuple(sorted(prefixes))


def deterministic_object_permutation(scene_id: str, object_count: int) -> tuple[int, ...]:
    """Return source object indices in deterministic runtime order."""

    return tuple(
        sorted(
            range(object_count),
            key=lambda index: (
                _stable_fraction(SELECTION_SEED, "object", scene_id, index),
                index,
            ),
        )
    )


def verify_pinned_sources(paths: SourcePaths) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify the compact source files and the previously checked archive record."""

    file_paths = {
        "master_json": paths.master_json,
        "catalogue": paths.catalogue,
        "archive_index": paths.archive_index,
        "readiness_audit": paths.readiness_audit,
        "official_split": paths.official_split,
    }
    actual_hashes = {
        label: calculate_file_sha256(path) for label, path in file_paths.items()
    }
    mismatches = {
        label: {"expected": EXPECTED_INPUT_HASHES[label], "actual": actual}
        for label, actual in actual_hashes.items()
        if actual != EXPECTED_INPUT_HASHES[label]
    }
    if mismatches:
        raise ValueError(f"Pinned selection input mismatch: {mismatches}")

    master = _load_json_object(paths.master_json)
    scenes = master.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Master JSON is missing its scenes array")
    canonical_hash = sha256(_canonical_json_bytes(scenes)).hexdigest()
    if canonical_hash != EXPECTED_CANONICAL_SCENES_HASH:
        raise ValueError(
            "Canonical scene payload mismatch: "
            f"{canonical_hash} != {EXPECTED_CANONICAL_SCENES_HASH}"
        )

    archive_manifest = _load_json_object(paths.archive_manifest)
    reconstruction = archive_manifest.get("reconstruction", {})
    if not isinstance(reconstruction, dict):
        raise ValueError("Archive manifest has no reconstruction record")
    archive_size = paths.archive_zip.stat().st_size
    if archive_size != EXPECTED_ARCHIVE_SIZE_BYTES:
        raise ValueError(
            f"Archive size mismatch: {archive_size} != {EXPECTED_ARCHIVE_SIZE_BYTES}"
        )
    if reconstruction.get("sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive manifest has an unexpected archive SHA-256")
    if reconstruction.get("central_directory_validation") != "passed":
        raise ValueError("Archive central-directory validation is not recorded as passed")
    if reconstruction.get("full_crc_validation") != "passed":
        raise ValueError("Archive full CRC validation is not recorded as passed")

    with sqlite3.connect(f"file:{paths.catalogue.resolve()}?mode=ro", uri=True) as db:
        catalogue_scene_count = int(db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0])
    if catalogue_scene_count != len(scenes):
        raise ValueError("Catalogue and master scene counts disagree")

    with sqlite3.connect(f"file:{paths.archive_index.resolve()}?mode=ro", uri=True) as db:
        archive_member_count = int(db.execute("SELECT COUNT(*) FROM members").fetchone()[0])
    declared_members = int(archive_manifest.get("archive", {}).get("member_count_including_directories", -1))
    if archive_member_count != declared_members:
        raise ValueError("Archive index and archive manifest member counts disagree")

    hashes = dict(actual_hashes)
    hashes["canonical_scenes"] = canonical_hash
    hashes["archive_zip_declared"] = EXPECTED_ARCHIVE_SHA256
    return master, hashes


def _official_split_mapping(path: Path) -> dict[str, str]:
    raw = _load_json_object(path)
    mapping: dict[str, str] = {}
    for split, scene_ids in raw.items():
        if not isinstance(scene_ids, list):
            raise ValueError(f"Official split {split!r} is not a list")
        for value in scene_ids:
            scene_id = str(value)
            if scene_id in mapping:
                raise ValueError(f"Official scene occurs in multiple splits: {scene_id}")
            mapping[scene_id] = str(split)
    return mapping


def assemble_eligible_scene_pool(
    master: Mapping[str, Any],
    official_split_path: Path,
    *,
    enforce_expected_counts: bool = True,
) -> list[dict[str, Any]]:
    """Apply the frozen train, availability and seen-asset exclusion rules."""

    scenes_value = master.get("scenes")
    if not isinstance(scenes_value, list):
        raise ValueError("Master JSON is missing its scenes array")
    scenes = [dict(scene) for scene in scenes_value]
    official_mapping = _official_split_mapping(official_split_path)
    by_id = {str(scene["scene_id"]): scene for scene in scenes}
    if len(by_id) != len(scenes):
        raise ValueError("Master JSON contains duplicate scene IDs")
    official_ids_missing_from_master = set(official_mapping) - set(by_id)
    master_ids_outside_official_split = set(by_id) - set(official_mapping)
    if enforce_expected_counts and official_ids_missing_from_master != set(
        EXPECTED_OFFICIAL_IDS_MISSING_FROM_MASTER
    ):
        raise ValueError(
            "Unexpected official IDs missing from the released master: "
            f"{sorted(official_ids_missing_from_master)}"
        )
    if enforce_expected_counts and master_ids_outside_official_split != set(
        EXPECTED_MASTER_IDS_OUTSIDE_OFFICIAL_SPLIT
    ):
        raise ValueError(
            "Unexpected master IDs outside the official split: "
            f"{sorted(master_ids_outside_official_split)}"
        )
    for scene_id, split in official_mapping.items():
        if scene_id not in by_id:
            continue
        if by_id[scene_id].get("split") != split:
            raise ValueError(f"Master split mismatch for {scene_id}")

    seen_assets = {
        str(by_id[scene_id]["base_asset_id"])
        for scene_id, split in official_mapping.items()
        if split == "seen-test" and scene_id in by_id
    }
    eligible = sorted(
        (
            scene
            for scene in scenes
            if scene.get("split") == "train"
            and scene.get("availability_ok") is True
            and str(scene["base_asset_id"]) not in seen_assets
        ),
        key=lambda scene: str(scene["scene_id"]),
    )

    if enforce_expected_counts:
        assets = {str(scene["base_asset_id"]) for scene in eligible}
        categories = {str(scene["category"]) for scene in eligible}
        strata = Counter(
            (int(scene["object_count"]), int(scene["anomaly_count"]))
            for scene in eligible
        )
        actual = {
            "scenes": len(eligible),
            "assets": len(assets),
            "categories": len(categories),
        }
        if actual != EXPECTED_POOL_COUNTS:
            raise ValueError(f"Eligible pool count mismatch: {actual}")
        if dict(strata) != EXPECTED_POOL_STRATA:
            raise ValueError(f"Eligible pool stratum mismatch: {dict(strata)}")
        prefixes = {prefix for scene in eligible for prefix in anomaly_prefixes(scene)}
        if prefixes != set(RAW_ANOMALY_PREFIXES):
            raise ValueError(f"Eligible pool prefix mismatch: {sorted(prefixes)}")
    return eligible


def _member_from_row(row: Sequence[Any]) -> IndexedMember:
    return IndexedMember(
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


def _fetch_member_records(
    database_path: Path,
    names: Sequence[str],
    *,
    batch_size: int = 700,
) -> dict[str, IndexedMember]:
    unique_names = sorted(set(names))
    records: dict[str, IndexedMember] = {}
    query_prefix = (
        "SELECT name, compressed_size, uncompressed_size, crc32, "
        "compression_method, flags, local_header_offset, is_directory, "
        "is_symlink, unsafe_reason FROM members WHERE name IN "
    )
    with sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True) as db:
        for start in range(0, len(unique_names), batch_size):
            batch = unique_names[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            for row in db.execute(f"{query_prefix}({placeholders})", batch):
                member = _member_from_row(row)
                if member.name in records:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                records[member.name] = member
    missing = sorted(set(unique_names) - records.keys())
    if missing:
        raise ValueError(
            f"Archive is missing {len(missing):,} required members; first: {missing[:5]}"
        )
    return records


def _source_member_specs(
    scene: Mapping[str, Any],
    object_names: Sequence[str],
) -> list[tuple[str, str, int | None, int | None]]:
    scene_id = str(scene["scene_id"])
    root = f"toysad8k/{scene_id}"
    specs: list[tuple[str, str, int | None, int | None]] = [
        (f"{root}/scene2d.metadata.json", "scene2d_metadata", None, None),
        (f"{root}/scene3d.metadata.json", "scene3d_metadata", None, None),
    ]
    for view_id in range(20):
        specs.append((f"{root}/RGB/{view_id:04d}.png", "rgb", None, view_id))
        specs.append((f"{root}/masks/{view_id:04d}.png", "scene_mask", None, view_id))
    for source_object_index, object_name in enumerate(object_names):
        for view_id in range(20):
            specs.append(
                (
                    f"{root}/segmentations/{object_name}/{object_name}_{view_id:04d}.png",
                    "object_mask",
                    source_object_index,
                    view_id,
                )
            )
    return specs


def audit_eligible_media(
    eligible: Sequence[Mapping[str, Any]],
    archive_index: Path,
    archive_zip: Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Validate exact RGB, scene-mask and metadata-linked object-mask paths."""

    scene3d_names = [
        str(scene["archive_members"]["scene3d_metadata"]) for scene in eligible
    ]
    metadata_records = _fetch_member_records(archive_index, scene3d_names)
    object_names_by_scene: dict[str, tuple[str, ...]] = {}
    metadata_bytes = 0
    with archive_zip.open("rb") as stream:
        ordered_records = sorted(
            metadata_records.values(), key=lambda member: member.local_header_offset
        )
        scenes_by_metadata = {
            str(scene["archive_members"]["scene3d_metadata"]): scene
            for scene in eligible
        }
        for member in ordered_records:
            content = read_indexed_member(stream, member)
            metadata_bytes += len(content)
            metadata = json.loads(content)
            objects = metadata.get("objects")
            if not isinstance(objects, list):
                raise ValueError(f"scene3d metadata has no objects list: {member.name}")
            scene = scenes_by_metadata[member.name]
            expected_paths = [str(value) for value in scene["blend_paths"]]
            actual_paths = [str(item.get("path")) for item in objects]
            object_names = tuple(str(item.get("name")) for item in objects)
            if actual_paths != expected_paths:
                raise ValueError(f"scene3d object path mismatch: {scene['scene_id']}")
            if len(object_names) != int(scene["object_count"]):
                raise ValueError(f"scene3d object count mismatch: {scene['scene_id']}")
            if len(set(object_names)) != len(object_names) or any(
                not name or "/" in name or "\\" in name for name in object_names
            ):
                raise ValueError(f"Unsafe or duplicate object names: {scene['scene_id']}")
            object_names_by_scene[str(scene["scene_id"])] = object_names

    required_names: list[str] = []
    component_counts: Counter[str] = Counter()
    for scene in eligible:
        specs = _source_member_specs(
            scene, object_names_by_scene[str(scene["scene_id"])]
        )
        required_names.extend(spec[0] for spec in specs)
        component_counts.update(spec[1] for spec in specs)
    if len(required_names) != len(set(required_names)):
        raise ValueError("Eligible media requirement contains duplicate paths")

    records = _fetch_member_records(archive_index, required_names)
    invalid = [
        member.name
        for member in records.values()
        if member.is_directory
        or member.is_symlink
        or member.unsafe_reason is not None
        or member.uncompressed_size <= 0
    ]
    if invalid:
        raise ValueError(f"Eligible pool contains invalid members: {invalid[:5]}")
    return object_names_by_scene, {
        "scene_count": len(eligible),
        "member_count": len(records),
        "component_counts": dict(sorted(component_counts.items())),
        "compressed_bytes": sum(member.compressed_size for member in records.values()),
        "uncompressed_bytes": sum(
            member.uncompressed_size for member in records.values()
        ),
        "scene3d_metadata_crc_checked_bytes": metadata_bytes,
        "all_required_paths_present_and_nonzero": True,
    }


def _solver_modules() -> tuple[Any, Any, Any, Any, Any, dict[str, str]]:
    try:
        import numpy as np
        import scipy
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
        import scipy.optimize._highspy._core as highs_core
    except ImportError as error:
        raise RuntimeError(
            "Selection requires the pinned NumPy and SciPy environment. "
            "Install code/requirements/selection-lock.txt."
        ) from error
    highs_version = ".".join(
        str(getattr(highs_core, name))
        for name in (
            "HIGHS_VERSION_MAJOR",
            "HIGHS_VERSION_MINOR",
            "HIGHS_VERSION_PATCH",
        )
    )
    versions = {
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "highs": highs_version,
    }
    return np, Bounds, LinearConstraint, milp, coo_array, versions


class _ConstraintMatrix:
    def __init__(self) -> None:
        self.row_indices: list[int] = []
        self.column_indices: list[int] = []
        self.values: list[float] = []
        self.lower_bounds: list[float] = []
        self.upper_bounds: list[float] = []

    def add(
        self,
        coefficients: Iterable[tuple[int, float]],
        lower: float,
        upper: float,
    ) -> None:
        row = len(self.lower_bounds)
        for column, value in coefficients:
            if value:
                self.row_indices.append(row)
                self.column_indices.append(column)
                self.values.append(float(value))
        self.lower_bounds.append(float(lower))
        self.upper_bounds.append(float(upper))


def _solve_binary_program(
    costs: Sequence[float],
    constraints: _ConstraintMatrix,
    *,
    time_limit_seconds: int = 300,
) -> tuple[list[float], dict[str, Any]]:
    np, Bounds, LinearConstraint, milp, coo_array, versions = _solver_modules()
    variable_count = len(costs)
    matrix = coo_array(
        (
            np.asarray(constraints.values, dtype=float),
            (
                np.asarray(constraints.row_indices, dtype=np.int32),
                np.asarray(constraints.column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(constraints.lower_bounds), variable_count),
    ).tocsc()
    result = milp(
        c=np.asarray(costs, dtype=float),
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraints.lower_bounds, dtype=float),
            np.asarray(constraints.upper_bounds, dtype=float),
        ),
        options={
            "disp": False,
            "mip_rel_gap": 0.0,
            "presolve": True,
            "time_limit": float(time_limit_seconds),
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Selection solver failed with status {result.status}: {result.message}"
        )
    solution = [float(value) for value in result.x]
    if any(min(abs(value), abs(value - 1.0)) > 1e-6 for value in solution):
        raise RuntimeError("Selection solver returned a non-binary solution")
    return solution, {
        **versions,
        "variable_count": variable_count,
        "constraint_count": len(constraints.lower_bounds),
        "objective_value": round(float(result.fun), 12),
        "optimal": True,
        "mip_relative_gap": round(float(getattr(result, "mip_gap", 0.0)), 12),
    }


def solve_asset_guarded_roles(
    eligible: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve the exact role and stratum allocation with whole-asset grouping."""

    scenes = sorted((dict(scene) for scene in eligible), key=lambda item: item["scene_id"])
    assets = sorted({str(scene["base_asset_id"]) for scene in scenes})
    scenes_by_asset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in scenes:
        scenes_by_asset[str(scene["base_asset_id"])].append(scene)

    scene_variable: dict[tuple[str, str], int] = {}
    asset_variable: dict[tuple[str, str], int] = {}
    costs: list[float] = []
    for scene in scenes:
        for role in ROLES:
            scene_variable[(str(scene["scene_id"]), role)] = len(costs)
            costs.append(_stable_fraction(SELECTION_SEED, "scene-role", scene["scene_id"], role))
    for asset in assets:
        for role in ROLES:
            asset_variable[(asset, role)] = len(costs)
            costs.append(
                0.1 * _stable_fraction(SELECTION_SEED, "asset-role", asset, role)
            )

    constraints = _ConstraintMatrix()
    for asset in assets:
        constraints.add(
            ((asset_variable[(asset, role)], 1.0) for role in ROLES), 1.0, 1.0
        )
        for role in ROLES:
            y = asset_variable[(asset, role)]
            asset_scene_variables = [
                scene_variable[(str(scene["scene_id"]), role)]
                for scene in scenes_by_asset[asset]
            ]
            for x in asset_scene_variables:
                constraints.add(((x, 1.0), (y, -1.0)), float("-inf"), 0.0)
            constraints.add(
                [(x, 1.0) for x in asset_scene_variables] + [(y, -3.0)],
                0.0,
                float("inf"),
            )
            constraints.add(
                [(x, 1.0) for x in asset_scene_variables] + [(y, -4.0)],
                float("-inf"),
                0.0,
            )

    for role in ROLES:
        constraints.add(
            ((asset_variable[(asset, role)], 1.0) for asset in assets),
            ROLE_ASSET_QUOTAS[role],
            ROLE_ASSET_QUOTAS[role],
        )
        for stratum in STRATA:
            variables = [
                scene_variable[(str(scene["scene_id"]), role)]
                for scene in scenes
                if (int(scene["object_count"]), int(scene["anomaly_count"]))
                == stratum
            ]
            quota = ROLE_STRATUM_QUOTAS[role][stratum]
            constraints.add(((variable, 1.0) for variable in variables), quota, quota)

    categories = sorted({str(scene["category"]) for scene in scenes})
    asset_category = {
        asset: str(scenes_by_asset[asset][0]["category"]) for asset in assets
    }
    for role in ROLES:
        required_categories = (
            [category for category in categories if category != "butterfly"]
            if role == "calibration"
            else categories
        )
        for category in required_categories:
            constraints.add(
                (
                    (asset_variable[(asset, role)], 1.0)
                    for asset in assets
                    if asset_category[asset] == category
                ),
                1.0,
                float("inf"),
            )
        for prefix in RAW_ANOMALY_PREFIXES:
            constraints.add(
                (
                    (scene_variable[(str(scene["scene_id"]), role)], 1.0)
                    for scene in scenes
                    if prefix in anomaly_prefixes(scene)
                ),
                1.0,
                float("inf"),
            )

    solution, solver_report = _solve_binary_program(costs, constraints)
    selected: list[dict[str, Any]] = []
    for scene in scenes:
        for role in ROLES:
            variable = scene_variable[(str(scene["scene_id"]), role)]
            if solution[variable] > 0.5:
                row = dict(scene)
                row["role"] = role
                selected.append(row)
    return selected, solver_report


def solve_development_pilot(
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Partition the development role into 100, 300 and 600-scene stages."""

    development = [dict(scene) for scene in selected if scene["role"] == "development"]
    scenes_by_asset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in development:
        scenes_by_asset[str(scene["base_asset_id"])].append(scene)
    assets = sorted(scenes_by_asset)

    variable: dict[tuple[str, str], int] = {}
    costs: list[float] = []
    for asset in assets:
        for stage in PILOT_STAGES:
            variable[(asset, stage)] = len(costs)
            costs.append(_stable_fraction(SELECTION_SEED, "pilot", asset, stage))

    categories = sorted({str(scene["category"]) for scene in development})
    category_variable: dict[tuple[str, str], int] = {}
    for stage, reward in (("screen_100", 100.0), ("confirm_300", 10.0)):
        for category in categories:
            category_variable[(category, stage)] = len(costs)
            costs.append(
                -reward
                + 0.001
                * _stable_fraction(
                    SELECTION_SEED, "pilot-category", stage, category
                )
            )

    constraints = _ConstraintMatrix()
    for asset in assets:
        constraints.add(
            ((variable[(asset, stage)], 1.0) for stage in PILOT_STAGES), 1.0, 1.0
        )
    for stage in PILOT_STAGES:
        constraints.add(
            ((variable[(asset, stage)], 1.0) for asset in assets),
            PILOT_ASSET_QUOTAS[stage],
            PILOT_ASSET_QUOTAS[stage],
        )
        for stratum in STRATA:
            constraints.add(
                (
                    (
                        variable[(asset, stage)],
                        float(
                            sum(
                                (
                                    int(scene["object_count"]),
                                    int(scene["anomaly_count"]),
                                )
                                == stratum
                                for scene in scenes_by_asset[asset]
                            )
                        ),
                    )
                    for asset in assets
                ),
                PILOT_STRATUM_QUOTAS[stage][stratum],
                PILOT_STRATUM_QUOTAS[stage][stratum],
            )

    for category in categories:
        constraints.add(
            (
                (variable[(asset, stage)], 1.0)
                for asset in assets
                for stage in ("screen_100", "confirm_300")
                if str(scenes_by_asset[asset][0]["category"]) == category
            ),
            1.0,
            float("inf"),
        )
    for stage in ("screen_100", "confirm_300"):
        for category in categories:
            covered_assets = [
                asset
                for asset in assets
                if str(scenes_by_asset[asset][0]["category"]) == category
            ]
            constraints.add(
                [(category_variable[(category, stage)], 1.0)]
                + [(variable[(asset, stage)], -1.0) for asset in covered_assets],
                float("-inf"),
                0.0,
            )
    for stage in ("screen_100", "confirm_300"):
        for prefix in RAW_ANOMALY_PREFIXES:
            constraints.add(
                (
                    (variable[(asset, stage)], 1.0)
                    for asset in assets
                    if any(
                        prefix in anomaly_prefixes(scene)
                        for scene in scenes_by_asset[asset]
                    )
                ),
                1.0,
                float("inf"),
            )

    solution, solver_report = _solve_binary_program(costs, constraints)
    stage_by_asset: dict[str, str] = {}
    for asset in assets:
        assigned = [
            stage for stage in PILOT_STAGES if solution[variable[(asset, stage)]] > 0.5
        ]
        if len(assigned) != 1:
            raise RuntimeError(f"Pilot solver did not assign one stage to {asset}")
        stage_by_asset[asset] = assigned[0]
    return stage_by_asset, solver_report


def _stratum_key(scene: Mapping[str, Any]) -> str:
    return f"{int(scene['object_count'])}x{int(scene['anomaly_count'])}"


def _role_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assets = {str(row["base_asset_id"]) for row in rows}
    asset_scene_counts = Counter(str(row["base_asset_id"]) for row in rows)
    return {
        "scene_count": len(rows),
        "asset_count": len(assets),
        "asset_multiplicity": {
            str(count): sum(value == count for value in asset_scene_counts.values())
            for count in (3, 4)
        },
        "strata": dict(sorted(Counter(_stratum_key(row) for row in rows).items())),
        "categories": sorted({str(row["category"]) for row in rows}),
        "category_count": len({str(row["category"]) for row in rows}),
        "prefix_scene_presence_counts": dict(
            sorted(
                Counter(
                    prefix for row in rows for prefix in set(anomaly_prefixes(row))
                ).items()
            )
        ),
        "prefix_anomalous_object_counts": dict(
            sorted(
                Counter(
                    str(variant).split("-", 1)[0]
                    for row in rows
                    for variant in row.get("anomaly_variants", [])
                    if variant
                ).items()
            )
        ),
    }


def audit_selected_design(
    eligible: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    pilot_stage_by_asset: Mapping[str, str],
) -> dict[str, Any]:
    """Recalculate every frozen selection invariant from the chosen rows."""

    eligible_ids = {str(scene["scene_id"]) for scene in eligible}
    selected_ids = [str(scene["scene_id"]) for scene in selected]
    if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= eligible_ids:
        raise ValueError("Selection contains duplicate or ineligible scenes")
    if len(selected) != 2_000:
        raise ValueError(f"Selection must contain 2,000 scenes, found {len(selected)}")

    selected_strata = Counter(
        (int(scene["object_count"]), int(scene["anomaly_count"]))
        for scene in selected
    )
    if dict(selected_strata) != EXPECTED_SELECTED_STRATA:
        raise ValueError(f"Selected stratum mismatch: {dict(selected_strata)}")

    assets_by_role: dict[str, set[str]] = {}
    role_reports: dict[str, Any] = {}
    for role in ROLES:
        rows = [scene for scene in selected if scene["role"] == role]
        role_reports[role] = _role_summary(rows)
        assets = {str(scene["base_asset_id"]) for scene in rows}
        assets_by_role[role] = assets
        actual_strata = Counter(
            (int(scene["object_count"]), int(scene["anomaly_count"]))
            for scene in rows
        )
        if dict(actual_strata) != ROLE_STRATUM_QUOTAS[role]:
            raise ValueError(f"Role stratum mismatch for {role}: {dict(actual_strata)}")
        if len(assets) != ROLE_ASSET_QUOTAS[role]:
            raise ValueError(f"Role asset count mismatch for {role}")
        multiplicities = Counter(str(scene["base_asset_id"]) for scene in rows)
        if set(multiplicities.values()) - {3, 4}:
            raise ValueError(f"Invalid asset multiplicity in {role}")
        if sum(value == 4 for value in multiplicities.values()) != ROLE_FOUR_SCENE_ASSET_QUOTAS[role]:
            raise ValueError(f"Four-scene asset count mismatch for {role}")
        prefixes = {prefix for scene in rows for prefix in anomaly_prefixes(scene)}
        if prefixes != set(RAW_ANOMALY_PREFIXES):
            raise ValueError(f"Role prefix coverage mismatch for {role}")

    for index, first in enumerate(ROLES):
        for second in ROLES[index + 1 :]:
            overlap = assets_by_role[first] & assets_by_role[second]
            if overlap:
                raise ValueError(f"Asset leakage between {first} and {second}")
    if set().union(*assets_by_role.values()) != {
        str(scene["base_asset_id"]) for scene in eligible
    }:
        raise ValueError("Selected roles do not use the complete eligible asset set")

    development_assets = assets_by_role["development"]
    if set(pilot_stage_by_asset) != development_assets:
        raise ValueError("Pilot stage map does not cover every development asset")
    pilot_reports: dict[str, Any] = {}
    for stage in PILOT_STAGES:
        rows = [
            scene
            for scene in selected
            if scene["role"] == "development"
            and pilot_stage_by_asset[str(scene["base_asset_id"])] == stage
        ]
        pilot_reports[stage] = _role_summary(rows)
        actual_strata = Counter(
            (int(scene["object_count"]), int(scene["anomaly_count"]))
            for scene in rows
        )
        if dict(actual_strata) != PILOT_STRATUM_QUOTAS[stage]:
            raise ValueError(f"Pilot stratum mismatch for {stage}")
        stage_assets = {str(scene["base_asset_id"]) for scene in rows}
        if len(stage_assets) != PILOT_ASSET_QUOTAS[stage]:
            raise ValueError(f"Pilot asset count mismatch for {stage}")
        multiplicities = Counter(str(scene["base_asset_id"]) for scene in rows)
        if sum(value == 4 for value in multiplicities.values()) != PILOT_FOUR_SCENE_ASSET_QUOTAS[stage]:
            raise ValueError(f"Pilot four-scene asset count mismatch for {stage}")
    pilot_rows = [
        scene
        for scene in selected
        if scene["role"] == "development"
        and pilot_stage_by_asset[str(scene["base_asset_id"])]
        in {"screen_100", "confirm_300"}
    ]
    if len({str(scene["category"]) for scene in pilot_rows}) != 39:
        raise ValueError("The 400-scene pilot does not cover all eligible categories")
    for stage in ("screen_100", "confirm_300"):
        rows = [
            scene
            for scene in selected
            if scene["role"] == "development"
            and pilot_stage_by_asset[str(scene["base_asset_id"])] == stage
        ]
        prefixes = {prefix for scene in rows for prefix in anomaly_prefixes(scene)}
        if prefixes != set(RAW_ANOMALY_PREFIXES):
            raise ValueError(f"Pilot prefix coverage mismatch for {stage}")

    return {
        "passed": True,
        "eligible_pool": {
            "scene_count": len(eligible),
            "asset_count": len({str(scene["base_asset_id"]) for scene in eligible}),
            "category_count": len({str(scene["category"]) for scene in eligible}),
            "official_ids_missing_from_released_master": list(
                EXPECTED_OFFICIAL_IDS_MISSING_FROM_MASTER
            ),
            "released_master_ids_outside_official_split": list(
                EXPECTED_MASTER_IDS_OUTSIDE_OFFICIAL_SPLIT
            ),
            "strata": dict(
                sorted(Counter(_stratum_key(scene) for scene in eligible).items())
            ),
        },
        "selected_corpus": {
            "scene_count": len(selected),
            "asset_count": len({str(scene["base_asset_id"]) for scene in selected}),
            "strata": dict(
                sorted(Counter(_stratum_key(scene) for scene in selected).items())
            ),
            "role_reports": role_reports,
            "pilot_reports": pilot_reports,
            "official_test_scene_count": sum(
                scene.get("split") in {"seen-test", "unseen-test"}
                for scene in selected
            ),
            "asset_role_overlap_count": 0,
        },
    }


def _opaque_identifiers(
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    assets = sorted(
        {str(scene["base_asset_id"]) for scene in selected},
        key=lambda value: (_stable_fraction(SELECTION_SEED, "opaque-asset", value), value),
    )
    scenes = sorted(
        (str(scene["scene_id"]) for scene in selected),
        key=lambda value: (_stable_fraction(SELECTION_SEED, "opaque-scene", value), value),
    )
    return (
        {asset: f"A{index:06d}" for index, asset in enumerate(assets, 1)},
        {scene: f"S{index:06d}" for index, scene in enumerate(scenes, 1)},
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(row).decode("utf-8"))
            handle.write("\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object: {path}")
                yield value


ROUTINE_RELEASE_FILES = {
    "experiment_roles.jsonl",
    "neutral_corpus_manifest.jsonl",
    "release_config.json",
    "selection_audit.json",
}
DEVELOPMENT_WORKSPACE_FILES = {
    "development_labels.jsonl",
    "development_source_map.jsonl",
    "workspace_config.json",
}
ADMINISTRATIVE_VAULT_FILES = {
    "administrative_config.json",
    "required_source_members.jsonl",
    "restricted_source_map.jsonl",
    "selection_design_audit.json",
}
CALIBRATION_VAULT_FILES = {"calibration_labels.jsonl", "vault_config.json"}
LOCK_VAULT_FILES = {"internal_lock_labels.jsonl", "vault_config.json"}
MODEL_FACING_MEDIA_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "opaque_scene_id",
        "object_count",
        "object_ids",
        "view_ids",
        "media_root",
    }
)


def _write_directory_checksums(directory: Path) -> None:
    lines = [
        f"{calculate_file_sha256(path)}  {path.name}\n"
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "checksums.sha256"
    ]
    with (directory / "checksums.sha256").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        handle.writelines(lines)


def validate_model_facing_media_record(row: Mapping[str, Any]) -> None:
    """Reject labels, groups, protocol fields and malformed media identifiers."""

    if set(row) != set(MODEL_FACING_MEDIA_FIELDS):
        raise ValueError(
            f"Model-facing media fields are not exact: {sorted(row)}"
        )
    if row.get("schema_version") != 1 or row.get("release_id") != RELEASE_ID:
        raise ValueError("Model-facing record has the wrong schema or release")
    opaque_scene_id = str(row["opaque_scene_id"])
    if not opaque_scene_id.startswith("S") or not opaque_scene_id[1:].isdigit():
        raise ValueError(f"Invalid opaque scene ID: {opaque_scene_id}")
    object_count = int(row["object_count"])
    if object_count not in {3, 4, 5}:
        raise ValueError(f"Invalid object count: {opaque_scene_id}")
    if row["object_ids"] != [f"O{index:02d}" for index in range(object_count)]:
        raise ValueError(f"Invalid object IDs: {opaque_scene_id}")
    if row["view_ids"] != [f"V{index:02d}" for index in range(20)]:
        raise ValueError(f"Invalid view IDs: {opaque_scene_id}")
    if row["media_root"] != f"media/{opaque_scene_id}":
        raise ValueError(f"Invalid media root: {opaque_scene_id}")


def _sanitized_design_audit(full_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Remove per-family locked information from the routine aggregate audit."""

    value = json.loads(json.dumps(full_audit))
    eligible = value["eligible_pool"]
    eligible["official_ids_missing_from_released_master_count"] = len(
        eligible.pop("official_ids_missing_from_released_master")
    )
    eligible["released_master_ids_outside_official_split_count"] = len(
        eligible.pop("released_master_ids_outside_official_split")
    )
    for role, report in value["selected_corpus"]["role_reports"].items():
        report["anomaly_prefix_coverage_count"] = len(
            report["prefix_scene_presence_counts"]
        )
        if role != "development":
            report.pop("categories", None)
            report.pop("prefix_scene_presence_counts", None)
            report.pop("prefix_anomalous_object_counts", None)
    return value


def _selection_build_provenance(selected_digest: str) -> dict[str, str]:
    code_root = Path(__file__).resolve().parents[3]
    requirements_path = code_root / "requirements" / "selection-lock.txt"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "selector_source_sha256": calculate_file_sha256(Path(__file__).resolve()),
        "selection_requirements_sha256": calculate_file_sha256(requirements_path),
        "selected_scene_assignment_sha256": selected_digest,
    }


def _target_member_path(
    opaque_scene_id: str,
    component: str,
    opaque_object_id: str | None,
    view_id: int | None,
) -> str | None:
    if component == "rgb":
        return f"media/{opaque_scene_id}/rgb/V{view_id:02d}.png"
    if component == "scene_mask":
        return f"media/{opaque_scene_id}/scene_masks/V{view_id:02d}.png"
    if component == "object_mask":
        return (
            f"media/{opaque_scene_id}/objects/{opaque_object_id}/masks/"
            f"V{view_id:02d}.png"
        )
    return None


def write_selection_release(
    output_directory: Path,
    source_paths: SourcePaths,
    development_workspace_directory: Path | None = None,
    administrative_vault_directory: Path | None = None,
    calibration_vault_directory: Path | None = None,
    lock_vault_directory: Path | None = None,
) -> dict[str, Any]:
    """Build one release with physically separate development and label stores."""

    output_directory = output_directory.expanduser().resolve()
    if development_workspace_directory is None:
        development_workspace_directory = output_directory.with_name(
            output_directory.name + "-development"
        )
    if administrative_vault_directory is None:
        administrative_vault_directory = output_directory.with_name(
            output_directory.name + "-administrative-vault"
        )
    if calibration_vault_directory is None:
        calibration_vault_directory = output_directory.with_name(
            output_directory.name + "-calibration-vault"
        )
    if lock_vault_directory is None:
        lock_vault_directory = output_directory.with_name(output_directory.name + "-lock-vault")
    development_workspace_directory = development_workspace_directory.expanduser().resolve()
    administrative_vault_directory = administrative_vault_directory.expanduser().resolve()
    calibration_vault_directory = calibration_vault_directory.expanduser().resolve()
    lock_vault_directory = lock_vault_directory.expanduser().resolve()
    directories = {
        "routine release": output_directory,
        "development workspace": development_workspace_directory,
        "administrative vault": administrative_vault_directory,
        "calibration vault": calibration_vault_directory,
        "lock vault": lock_vault_directory,
    }
    for label, directory in directories.items():
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {label}: {directory}")
    for directory in directories.values():
        directory.mkdir(parents=True)

    master, input_hashes = verify_pinned_sources(source_paths)
    eligible = assemble_eligible_scene_pool(master, source_paths.official_split)
    object_names_by_scene, eligible_media_report = audit_eligible_media(
        eligible, source_paths.archive_index, source_paths.archive_zip
    )
    selected, main_solver_report = solve_asset_guarded_roles(eligible)
    pilot_stage_by_asset, pilot_solver_report = solve_development_pilot(selected)
    design_audit = audit_selected_design(eligible, selected, pilot_stage_by_asset)

    opaque_assets, opaque_scenes = _opaque_identifiers(selected)
    selected_by_opaque = sorted(
        selected, key=lambda scene: opaque_scenes[str(scene["scene_id"])]
    )

    source_map_rows: list[dict[str, Any]] = []
    development_source_rows: list[dict[str, Any]] = []
    neutral_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    development_labels: list[dict[str, Any]] = []
    calibration_labels: list[dict[str, Any]] = []
    lock_labels: list[dict[str, Any]] = []
    source_member_rows: list[dict[str, Any]] = []
    all_selected_member_names: list[str] = []
    scene_specs: dict[str, list[tuple[str, str, int | None, int | None]]] = {}

    for scene in selected_by_opaque:
        scene_id = str(scene["scene_id"])
        specs = _source_member_specs(scene, object_names_by_scene[scene_id])
        scene_specs[scene_id] = specs
        all_selected_member_names.extend(spec[0] for spec in specs)
    selected_member_records = _fetch_member_records(
        source_paths.archive_index, all_selected_member_names
    )

    component_counts: Counter[str] = Counter()
    component_uncompressed_bytes: Counter[str] = Counter()
    for scene in selected_by_opaque:
        scene_id = str(scene["scene_id"])
        asset_id = str(scene["base_asset_id"])
        opaque_scene = opaque_scenes[scene_id]
        opaque_asset = opaque_assets[asset_id]
        object_count = int(scene["object_count"])
        permutation = deterministic_object_permutation(scene_id, object_count)
        runtime_object_id_by_source = {
            source_index: f"O{runtime_index:02d}"
            for runtime_index, source_index in enumerate(permutation)
        }
        labels = [
            int(scene["normalized_anomaly_flags"][source_index])
            for source_index in permutation
        ]
        role = str(scene["role"])
        pilot_stage = (
            pilot_stage_by_asset[asset_id] if role == "development" else None
        )
        member_bytes = sum(
            selected_member_records[name].uncompressed_size
            for name, _component, _object_index, _view_id in scene_specs[scene_id]
        )
        source_map_row = {
            "schema_version": 1,
            "release_id": RELEASE_ID,
            "source_scene_id": scene_id,
            "opaque_scene_id": opaque_scene,
            "source_base_asset_id": asset_id,
            "opaque_asset_group_id": opaque_asset,
            "official_split": str(scene["split"]),
            "role": role,
            "pilot_stage": pilot_stage,
            "object_count": object_count,
            "object_permutation_source_indices": list(permutation),
            "runtime_object_ids": [f"O{index:02d}" for index in range(object_count)],
            "scene3d_object_names_in_source_order": list(
                object_names_by_scene[scene_id]
            ),
            "selected_source_member_count": len(scene_specs[scene_id]),
            "selected_source_uncompressed_bytes": member_bytes,
            "source_member_manifest": "required_source_members.jsonl",
            "availability_evidence": {
                "master_availability_ok": True,
                "all_exact_members_present_and_nonzero": True,
                "archive_full_crc_validation_recorded": True,
            },
            "selection_seed": SELECTION_SEED,
            "input_hashes": input_hashes,
        }
        source_map_rows.append(source_map_row)
        if role == "development":
            development_source_rows.append(
                {
                    **source_map_row,
                    "category": str(scene["category"]),
                    "anomaly_count": int(scene["anomaly_count"]),
                    "raw_anomaly_prefixes": list(anomaly_prefixes(scene)),
                    "permuted_anomaly_labels": labels,
                }
            )
        neutral_rows.append(
            {
                "schema_version": 1,
                "release_id": RELEASE_ID,
                "opaque_scene_id": opaque_scene,
                "object_count": object_count,
                "object_ids": [f"O{index:02d}" for index in range(object_count)],
                "view_ids": [f"V{index:02d}" for index in range(20)],
                "media_root": f"media/{opaque_scene}",
            }
        )
        role_rows.append(
            {
                "schema_version": 1,
                "release_id": RELEASE_ID,
                "opaque_scene_id": opaque_scene,
                "opaque_asset_group_id": opaque_asset,
                "role": role,
                "pilot_stage": pilot_stage,
            }
        )
        evaluator_row = {
            "schema_version": 1,
            "release_id": RELEASE_ID,
            "opaque_scene_id": opaque_scene,
            "labels_by_object_id": {
                f"O{index:02d}": label for index, label in enumerate(labels)
            },
            "anomaly_count": int(scene["anomaly_count"]),
            "role": role,
        }
        if role == "internal_lock":
            lock_labels.append(evaluator_row)
        elif role == "calibration":
            calibration_labels.append(evaluator_row)
        else:
            development_labels.append(evaluator_row)

        for source_name, component, source_object_index, view_id in scene_specs[scene_id]:
            member = selected_member_records[source_name]
            opaque_object_id = (
                runtime_object_id_by_source[source_object_index]
                if source_object_index is not None
                else None
            )
            target = _target_member_path(
                opaque_scene, component, opaque_object_id, view_id
            )
            include_in_media = component in {"rgb", "scene_mask", "object_mask"}
            source_member_rows.append(
                {
                    "schema_version": 1,
                    "release_id": RELEASE_ID,
                    "opaque_scene_id": opaque_scene,
                    "component": component,
                    "opaque_object_id": opaque_object_id,
                    "view_id": f"V{view_id:02d}" if view_id is not None else None,
                    "source_member": source_name,
                    "target_member": target,
                    "include_in_media_package": include_in_media,
                    "crc32": f"{member.crc32:08x}",
                    "compressed_size": member.compressed_size,
                    "uncompressed_size": member.uncompressed_size,
                    "compression_method": member.compression_method,
                    "local_header_offset": member.local_header_offset,
                }
            )
            component_counts[component] += 1
            component_uncompressed_bytes[component] += member.uncompressed_size

    selected_digest = sha256(
        _canonical_json_bytes(
            [
                {
                    "source_scene_id": row["source_scene_id"],
                    "opaque_scene_id": row["opaque_scene_id"],
                    "role": row["role"],
                    "pilot_stage": row["pilot_stage"],
                }
                for row in source_map_rows
            ]
        )
    ).hexdigest()
    provenance = _selection_build_provenance(selected_digest)
    release_config = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "selection_seed": SELECTION_SEED,
        "input_hashes": input_hashes,
        "build_provenance": provenance,
        "archive_integrity_basis": {
            "archive_size_bytes_checked_now": EXPECTED_ARCHIVE_SIZE_BYTES,
            "archive_sha256_from_pinned_manifest": EXPECTED_ARCHIVE_SHA256,
            "central_directory_validation_from_manifest": "passed",
            "full_crc_validation_from_manifest": "passed",
            "archive_sha256_recomputed_during_selection": False,
        },
        "roles": list(ROLES),
        "pilot_stages": list(PILOT_STAGES),
        "views_per_scene": 20,
        "full_archive_extracted": False,
        "selection_uses_model_outputs": False,
        "model_facing_file": "neutral_corpus_manifest.jsonl",
        "protocol_file": "experiment_roles.jsonl",
        "upload_allowlist": [
            "neutral_corpus_manifest.jsonl",
            "release_config.json",
        ],
    }
    full_audit = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "selection_seed": SELECTION_SEED,
        "passed": True,
        "input_hashes": input_hashes,
        "build_provenance": provenance,
        "design": design_audit,
        "eligible_media": eligible_media_report,
        "selected_media": {
            "member_count": len(source_member_rows),
            "component_counts": dict(sorted(component_counts.items())),
            "component_uncompressed_bytes": dict(
                sorted(component_uncompressed_bytes.items())
            ),
            "all_required_paths_indexed_and_nonzero": True,
            "all_selected_media_crc_read_during_selection": False,
            "all_selected_images_decoded_during_selection": False,
            "raw_source_metadata_upload_allowed": False,
        },
        "solver": {
            "role_selection": main_solver_report,
            "development_pilot": pilot_solver_report,
        },
        "leakage_controls": {
            "selection_grouped_by_base_asset": True,
            "official_tests_excluded": True,
            "object_order_permuted_consistently_across_views": True,
            "neutral_manifest_contains_source_ids": False,
            "neutral_manifest_contains_asset_groups": False,
            "lock_labels_absent_from_routine_release": True,
            "calibration_labels_absent_from_routine_release": True,
            "lock_labels_written_to_separate_vault": True,
        },
    }
    routine_audit = dict(full_audit)
    routine_audit["design"] = _sanitized_design_audit(design_audit)

    _write_jsonl(output_directory / "neutral_corpus_manifest.jsonl", neutral_rows)
    _write_jsonl(output_directory / "experiment_roles.jsonl", role_rows)
    _write_json(output_directory / "release_config.json", release_config)
    _write_json(output_directory / "selection_audit.json", routine_audit)

    _write_jsonl(
        development_workspace_directory / "development_source_map.jsonl",
        development_source_rows,
    )
    _write_jsonl(
        development_workspace_directory / "development_labels.jsonl",
        development_labels,
    )
    _write_json(
        development_workspace_directory / "workspace_config.json",
        {
            "schema_version": 1,
            "release_id": RELEASE_ID,
            "scene_count": len(development_labels),
            "allowed_use": "Method development and component selection only.",
        },
    )

    _write_jsonl(
        administrative_vault_directory / "restricted_source_map.jsonl",
        source_map_rows,
    )
    _write_jsonl(
        administrative_vault_directory / "required_source_members.jsonl",
        source_member_rows,
    )
    _write_json(
        administrative_vault_directory / "selection_design_audit.json", full_audit
    )
    _write_json(
        administrative_vault_directory / "administrative_config.json",
        {
            "schema_version": 1,
            "release_id": RELEASE_ID,
            "scene_count": len(source_map_rows),
            "upload_allowed": False,
            "purpose": "Selection verification and source-to-opaque packaging only.",
        },
    )

    _write_jsonl(
        calibration_vault_directory / "calibration_labels.jsonl",
        calibration_labels,
    )
    _write_json(
        calibration_vault_directory / "vault_config.json",
        {
            "schema_version": 1,
            "release_id": RELEASE_ID,
            "scene_count": len(calibration_labels),
            "access_rule": (
                "Open only after the method, view policy and fusion rule are frozen."
            ),
        },
    )

    _write_jsonl(lock_vault_directory / "internal_lock_labels.jsonl", lock_labels)
    _write_json(
        lock_vault_directory / "vault_config.json",
        {
            "schema_version": 1,
            "release_id": RELEASE_ID,
            "scene_count": len(lock_labels),
            "access_rule": (
                "Do not open until frozen internal-lock predictions are saved and hashed."
            ),
        },
    )

    for directory in directories.values():
        _write_directory_checksums(directory)
    return routine_audit


def verify_release_checksums(
    output_directory: Path,
    expected_files: set[str] | frozenset[str] | None = None,
) -> dict[str, str]:
    """Verify an exact checksummed file set and reject unlisted extras."""

    expected: dict[str, str] = {}
    checksum_path = output_directory / "checksums.sha256"
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("  ", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed checksum line: {line!r}")
            digest, filename = parts
            if filename in expected:
                raise ValueError(f"Duplicate checksum entry: {filename}")
            member_path = PurePosixPath(filename)
            if (
                not filename
                or member_path.is_absolute()
                or ".." in member_path.parts
                or len(member_path.parts) != 1
            ):
                raise ValueError(f"Unsafe checksum filename: {filename}")
            expected[filename] = digest
    if expected_files is not None and set(expected) != set(expected_files):
        raise ValueError(
            "Checksummed file set mismatch: "
            f"expected {sorted(expected_files)}, found {sorted(expected)}"
        )
    actual_files = {
        path.name
        for path in output_directory.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    }
    if actual_files != set(expected):
        raise ValueError(
            "Directory contains missing or unlisted files: "
            f"expected {sorted(expected)}, found {sorted(actual_files)}"
        )
    actual = {
        filename: calculate_file_sha256(output_directory / filename)
        for filename in expected
    }
    mismatches = {
        filename: {"expected": expected[filename], "actual": actual[filename]}
        for filename in expected
        if actual[filename] != expected[filename]
    }
    if mismatches:
        raise ValueError(f"Selection release checksum mismatch: {mismatches}")
    return actual


def audit_routine_release(
    output_directory: Path,
    development_workspace_directory: Path | None = None,
) -> dict[str, Any]:
    """Audit only the routine release and open development workspace."""

    if development_workspace_directory is None:
        development_workspace_directory = output_directory.with_name(
            output_directory.name + "-development"
        )
    release_hashes = verify_release_checksums(
        output_directory, ROUTINE_RELEASE_FILES
    )
    development_hashes = verify_release_checksums(
        development_workspace_directory, DEVELOPMENT_WORKSPACE_FILES
    )
    neutral = list(_read_jsonl(output_directory / "neutral_corpus_manifest.jsonl"))
    roles = list(_read_jsonl(output_directory / "experiment_roles.jsonl"))
    development_labels = list(
        _read_jsonl(development_workspace_directory / "development_labels.jsonl")
    )
    development_source = list(
        _read_jsonl(
            development_workspace_directory / "development_source_map.jsonl"
        )
    )
    media_ids = [str(row["opaque_scene_id"]) for row in neutral]
    protocol_ids = [str(row["opaque_scene_id"]) for row in roles]
    if len(media_ids) != 2_000 or len(set(media_ids)) != 2_000:
        raise ValueError("Routine media manifest must have 2,000 unique scenes")
    if set(protocol_ids) != set(media_ids) or len(protocol_ids) != 2_000:
        raise ValueError("Routine media and protocol scene sets differ")
    for row in neutral:
        validate_model_facing_media_record(row)
    role_counts = Counter(str(row["role"]) for row in roles)
    if role_counts != Counter(
        {"development": 1_000, "calibration": 600, "internal_lock": 400}
    ):
        raise ValueError(f"Routine protocol role counts differ: {dict(role_counts)}")
    development_ids = {
        str(row["opaque_scene_id"])
        for row in roles
        if row["role"] == "development"
    }
    for name, rows in (
        ("development labels", development_labels),
        ("development source", development_source),
    ):
        ids = [str(row["opaque_scene_id"]) for row in rows]
        if len(ids) != 1_000 or set(ids) != development_ids:
            raise ValueError(f"{name} does not exactly cover development")
    pilot_counts = Counter(
        str(row["pilot_stage"])
        for row in roles
        if row["role"] == "development"
    )
    if pilot_counts != Counter(
        {"screen_100": 100, "confirm_300": 300, "remainder_600": 600}
    ):
        raise ValueError(f"Development pilot counts differ: {dict(pilot_counts)}")
    audit = _load_json_object(output_directory / "selection_audit.json")
    release_config = _load_json_object(output_directory / "release_config.json")
    if audit.get("release_id") != RELEASE_ID or release_config.get("release_id") != RELEASE_ID:
        raise ValueError("Routine release identifiers differ")
    return {
        "passed": True,
        "release_id": RELEASE_ID,
        "neutral_scene_count": len(neutral),
        "role_counts": dict(sorted(role_counts.items())),
        "pilot_counts": dict(sorted(pilot_counts.items())),
        "model_facing_fields_verified": True,
        "development_only_labels_verified": True,
        "selection_audit": audit,
        "release_config": release_config,
        "bundle_hashes": {
            "routine_release": release_hashes,
            "development_workspace": development_hashes,
        },
    }


def audit_existing_release(
    output_directory: Path,
    source_paths: SourcePaths,
    development_workspace_directory: Path | None = None,
    administrative_vault_directory: Path | None = None,
    calibration_vault_directory: Path | None = None,
    lock_vault_directory: Path | None = None,
) -> dict[str, Any]:
    """Run the pre-experiment administrative audit across separated stores."""

    if development_workspace_directory is None:
        development_workspace_directory = output_directory.with_name(
            output_directory.name + "-development"
        )
    if administrative_vault_directory is None:
        administrative_vault_directory = output_directory.with_name(
            output_directory.name + "-administrative-vault"
        )
    if calibration_vault_directory is None:
        calibration_vault_directory = output_directory.with_name(
            output_directory.name + "-calibration-vault"
        )
    if lock_vault_directory is None:
        lock_vault_directory = output_directory.with_name(
            output_directory.name + "-lock-vault"
        )
    verify_release_checksums(output_directory, ROUTINE_RELEASE_FILES)
    verify_release_checksums(
        development_workspace_directory, DEVELOPMENT_WORKSPACE_FILES
    )
    verify_release_checksums(
        administrative_vault_directory, ADMINISTRATIVE_VAULT_FILES
    )
    verify_release_checksums(calibration_vault_directory, CALIBRATION_VAULT_FILES)
    verify_release_checksums(lock_vault_directory, LOCK_VAULT_FILES)
    master, input_hashes = verify_pinned_sources(source_paths)
    eligible = assemble_eligible_scene_pool(master, source_paths.official_split)
    eligible_by_id = {str(scene["scene_id"]): scene for scene in eligible}
    source_map = list(
        _read_jsonl(administrative_vault_directory / "restricted_source_map.jsonl")
    )
    roles = list(_read_jsonl(output_directory / "experiment_roles.jsonl"))
    neutral = list(_read_jsonl(output_directory / "neutral_corpus_manifest.jsonl"))
    development_source = list(
        _read_jsonl(development_workspace_directory / "development_source_map.jsonl")
    )
    required_members = list(
        _read_jsonl(administrative_vault_directory / "required_source_members.jsonl")
    )
    development_evaluators = list(
        _read_jsonl(development_workspace_directory / "development_labels.jsonl")
    )
    calibration_evaluators = list(
        _read_jsonl(calibration_vault_directory / "calibration_labels.jsonl")
    )
    lock_evaluators = list(
        _read_jsonl(lock_vault_directory / "internal_lock_labels.jsonl")
    )
    evaluators = development_evaluators + calibration_evaluators + lock_evaluators
    release_config = _load_json_object(output_directory / "release_config.json")

    for name, rows, expected_count in (
        ("source map", source_map, 2_000),
        ("protocol", roles, 2_000),
        ("media", neutral, 2_000),
        ("development source", development_source, 1_000),
        ("all separated evaluators", evaluators, 2_000),
    ):
        if len(rows) != expected_count:
            raise ValueError(
                f"{name} manifest count mismatch: {len(rows)} != {expected_count}"
            )
        scene_ids = [str(row["opaque_scene_id"]) for row in rows]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError(f"{name} manifest contains duplicate opaque scene IDs")

    source_by_opaque = {str(row["opaque_scene_id"]): row for row in source_map}
    role_by_opaque = {str(row["opaque_scene_id"]): row for row in roles}
    neutral_by_opaque = {str(row["opaque_scene_id"]): row for row in neutral}
    evaluator_by_opaque = {str(row["opaque_scene_id"]): row for row in evaluators}
    opaque_ids = set(source_by_opaque)
    if set(role_by_opaque) != opaque_ids or set(neutral_by_opaque) != opaque_ids:
        raise ValueError("Source, protocol and media manifests cover different scenes")
    if set(evaluator_by_opaque) != opaque_ids:
        raise ValueError("Separated evaluator records do not cover every selected scene")
    expected_development_ids = {
        scene_id for scene_id, row in role_by_opaque.items() if row["role"] == "development"
    }
    development_source_by_opaque = {
        str(row["opaque_scene_id"]): row for row in development_source
    }
    if set(development_source_by_opaque) != expected_development_ids:
        raise ValueError("Development source map does not exactly cover development")

    exact_protocol_fields = {
        "schema_version",
        "release_id",
        "opaque_scene_id",
        "opaque_asset_group_id",
        "role",
        "pilot_stage",
    }
    for opaque_scene_id, protocol_row in role_by_opaque.items():
        if set(protocol_row) != exact_protocol_fields:
            raise ValueError(f"Protocol fields are not exact: {opaque_scene_id}")
        if protocol_row["role"] not in ROLES:
            raise ValueError(f"Unknown protocol role: {opaque_scene_id}")
        if (
            protocol_row["role"] == "development"
            and protocol_row["pilot_stage"] not in PILOT_STAGES
        ):
            raise ValueError(f"Invalid development pilot stage: {opaque_scene_id}")
        if (
            protocol_row["role"] != "development"
            and protocol_row["pilot_stage"] is not None
        ):
            raise ValueError(f"Non-development scene has a pilot stage: {opaque_scene_id}")
    for opaque_scene_id, media_row in neutral_by_opaque.items():
        validate_model_facing_media_record(media_row)

    selected: list[dict[str, Any]] = []
    pilot_stage_by_asset: dict[str, str] = {}
    opaque_by_source: dict[str, str] = {}
    for row in source_map:
        scene_id = str(row["source_scene_id"])
        if scene_id not in eligible_by_id:
            raise ValueError(f"Release contains ineligible scene: {scene_id}")
        if scene_id in opaque_by_source:
            raise ValueError(f"Source scene appears more than once: {scene_id}")
        opaque_scene_id = str(row["opaque_scene_id"])
        opaque_by_source[scene_id] = opaque_scene_id
        scene = dict(eligible_by_id[scene_id])
        protocol = role_by_opaque[opaque_scene_id]
        if row["role"] != protocol["role"] or row["pilot_stage"] != protocol["pilot_stage"]:
            raise ValueError(f"Source and protocol roles differ: {opaque_scene_id}")
        if row["opaque_asset_group_id"] != protocol["opaque_asset_group_id"]:
            raise ValueError(f"Source and protocol asset groups differ: {opaque_scene_id}")
        if int(row["object_count"]) != int(neutral_by_opaque[opaque_scene_id]["object_count"]):
            raise ValueError(f"Source and media object counts differ: {opaque_scene_id}")
        scene["role"] = str(protocol["role"])
        selected.append(scene)
        permutation = deterministic_object_permutation(scene_id, int(scene["object_count"]))
        if row["object_permutation_source_indices"] != list(permutation):
            raise ValueError(f"Object permutation mismatch: {opaque_scene_id}")
        labels = [
            int(scene["normalized_anomaly_flags"][index]) for index in permutation
        ]
        forbidden_source_map_fields = {
            "category",
            "anomaly_count",
            "raw_anomaly_prefixes",
            "permuted_anomaly_labels",
        }
        if forbidden_source_map_fields & set(row):
            raise ValueError(f"Administrative source map exposes labels: {opaque_scene_id}")
        evaluator = evaluator_by_opaque[opaque_scene_id]
        expected_labels = {
            f"O{index:02d}": label for index, label in enumerate(labels)
        }
        if evaluator["labels_by_object_id"] != expected_labels:
            raise ValueError(f"Evaluator labels mismatch: {opaque_scene_id}")
        if evaluator["role"] != protocol["role"]:
            raise ValueError(f"Evaluator role mismatch: {opaque_scene_id}")
        if protocol["role"] == "development":
            development_row = development_source_by_opaque[opaque_scene_id]
            if development_row.get("category") != scene["category"]:
                raise ValueError(f"Development category mismatch: {opaque_scene_id}")
            if development_row.get("permuted_anomaly_labels") != labels:
                raise ValueError(f"Development labels mismatch: {opaque_scene_id}")
            for field, value in row.items():
                if development_row.get(field) != value:
                    raise ValueError(
                        f"Development and administrative source maps differ: "
                        f"{opaque_scene_id}, {field}"
                    )
        if protocol["role"] == "development":
            asset = str(scene["base_asset_id"])
            stage = str(protocol["pilot_stage"])
            previous = pilot_stage_by_asset.setdefault(asset, stage)
            if previous != stage:
                raise ValueError(f"Development asset crosses pilot stages: {asset}")
        if row.get("input_hashes") != input_hashes:
            raise ValueError("Restricted manifest input hashes do not match current inputs")
    audit = audit_selected_design(eligible, selected, pilot_stage_by_asset)

    expected_component_counts = {
        "object_mask": 154_000,
        "rgb": 40_000,
        "scene2d_metadata": 2_000,
        "scene3d_metadata": 2_000,
        "scene_mask": 40_000,
    }
    actual_component_counts = dict(
        sorted(Counter(str(row["component"]) for row in required_members).items())
    )
    if actual_component_counts != expected_component_counts:
        raise ValueError(
            f"Required member component mismatch: {actual_component_counts}"
        )
    required_by_scene: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    target_members: list[str] = []
    for row in required_members:
        opaque_scene_id = str(row["opaque_scene_id"])
        if opaque_scene_id not in opaque_ids:
            raise ValueError(f"Required member references unknown scene: {opaque_scene_id}")
        required_by_scene[opaque_scene_id].append(row)
        if row["include_in_media_package"]:
            target = row.get("target_member")
            if not target:
                raise ValueError(f"Packaged media has no target path: {opaque_scene_id}")
            target_members.append(str(target))
        elif row.get("target_member") is not None:
            raise ValueError(f"Source metadata has a package target: {opaque_scene_id}")
    if len(target_members) != len(set(target_members)):
        raise ValueError("Required member manifest contains duplicate package targets")
    for opaque_scene_id, media_row in neutral_by_opaque.items():
        expected_count = 42 + int(media_row["object_count"]) * 20
        if len(required_by_scene[opaque_scene_id]) != expected_count:
            raise ValueError(f"Required member count mismatch: {opaque_scene_id}")

    archive_members = _fetch_member_records(
        source_paths.archive_index,
        [str(row["source_member"]) for row in required_members],
    )
    for row in required_members:
        source_name = str(row["source_member"])
        member = archive_members[source_name]
        if (
            member.crc32 != int(str(row["crc32"]), 16)
            or member.compressed_size != int(row["compressed_size"])
            or member.uncompressed_size != int(row["uncompressed_size"])
            or member.compression_method != int(row["compression_method"])
            or member.local_header_offset != int(row["local_header_offset"])
        ):
            raise ValueError(f"Required member index evidence differs: {source_name}")

    if release_config.get("release_id") != RELEASE_ID:
        raise ValueError("Release configuration has the wrong release ID")
    if release_config.get("input_hashes") != input_hashes:
        raise ValueError("Release configuration input hashes differ")
    selected_digest = sha256(
        _canonical_json_bytes(
            [
                {
                    "source_scene_id": row["source_scene_id"],
                    "opaque_scene_id": row["opaque_scene_id"],
                    "role": row["role"],
                    "pilot_stage": row["pilot_stage"],
                }
                for row in source_map
            ]
        )
    ).hexdigest()
    if (
        release_config.get("build_provenance", {}).get(
            "selected_scene_assignment_sha256"
        )
        != selected_digest
    ):
        raise ValueError("Selected scene assignment digest differs")
    return {
        "passed": True,
        "release_id": RELEASE_ID,
        "design": audit,
        "neutral_scene_count": len(neutral),
        "required_member_count": len(required_members),
        "required_component_counts": actual_component_counts,
        "model_facing_fields_verified": True,
        "cross_manifest_alignment_verified": True,
        "required_members_rechecked_against_archive_index": True,
        "lock_labels_absent_from_routine_release": True,
        "calibration_labels_absent_from_routine_release": True,
        "selected_scene_assignment_sha256": selected_digest,
        "checksums_verified": True,
    }


def compare_release_directories(first: Path, second: Path) -> dict[str, str]:
    """Require two release directories to have identical file sets and bytes."""

    first_files = sorted(path.name for path in first.iterdir() if path.is_file())
    second_files = sorted(path.name for path in second.iterdir() if path.is_file())
    if first_files != second_files:
        raise ValueError("Selection rebuild file sets differ")
    hashes: dict[str, str] = {}
    for filename in first_files:
        first_hash = calculate_file_sha256(first / filename)
        second_hash = calculate_file_sha256(second / filename)
        if first_hash != second_hash:
            raise ValueError(f"Selection rebuild differs: {filename}")
        hashes[filename] = first_hash
    return hashes


def verify_repeat_build_evidence(
    evidence_path: Path,
    bundle_directories: Mapping[str, Path],
) -> dict[str, Any]:
    """Bind repeat-build evidence to the bytes in the current separated stores."""

    evidence = _load_json_object(evidence_path)
    if evidence.get("release_id") != RELEASE_ID or evidence.get("byte_identical") is not True:
        raise ValueError("Repeat-build evidence does not certify this release")
    recorded = evidence.get("bundle_file_hashes")
    if not isinstance(recorded, dict) or not set(bundle_directories) <= set(recorded):
        raise ValueError("Repeat-build evidence does not cover the requested bundles")
    for bundle, directory in bundle_directories.items():
        actual = {
            path.name: calculate_file_sha256(path)
            for path in sorted(directory.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        if recorded[bundle] != actual:
            raise ValueError(f"Repeat-build evidence is stale for bundle: {bundle}")
    return evidence
