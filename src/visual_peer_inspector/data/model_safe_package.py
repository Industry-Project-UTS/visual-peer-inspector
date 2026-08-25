"""Build and verify the opaque RGB package used by model notebooks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import binascii
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from PIL import Image

from .cohort_contract import CohortContract, DS1_SCREEN_CONTRACT
from .toysad_selection import RELEASE_ID, validate_model_facing_media_record


SMOKE20_COHORT_ID = "toysad8k-development-smoke-20-v1"
SMOKE20_SELECTION_SEED = "toysad8k-smoke-20-v1"
SMOKE20_MEMBERSHIP_SHA256 = (
    "5439c5db10b19a47706d2f5eb135e672bb0387076462d967c326b1bcc15f0460"
)
EXPECTED_RGB_SIZE = (256, 256)
EXPECTED_SCENE_COUNT = 20
EXPECTED_VIEWS_PER_SCENE = 20
EXPECTED_ARCHIVE_SIZE_BYTES = 52_993_751_003
EXPECTED_ARCHIVE_SHA256 = (
    "e49c30857b96fdede3f374c773ce0758174f225958ba29c9ad5ca5b075dfc733"
)
EXPECTED_SELECTION_CHECKSUM_MANIFEST_SHA256 = (
    "06645dd4dca16dab234983a306592eb78f1d6520fe9d18104693bdb6b06eabaf"
)
EXPECTED_DEVELOPMENT_CHECKSUM_MANIFEST_SHA256 = (
    "f8896d3ff4bc7d0329a5d3442ca215534bf777548e9fb3e6b102217f62577d47"
)
EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256 = (
    "bde49d0a0121965bf9d33a3c2473ee0574196d207fa206824ba61648c6b4cc56"
)
EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256 = (
    "0630cc7c011bbea0b80036c9ea4d71f6abd6fc178ff105c84ef3543ad9a0b6fa"
)

SMOKE20_CONTRACT = CohortContract(
    cohort_id=SMOKE20_COHORT_ID,
    release_id=RELEASE_ID,
    role="development",
    pilot_stage="screen_100",
    membership_sha256=SMOKE20_MEMBERSHIP_SHA256,
    scene_count=EXPECTED_SCENE_COUNT,
    views_per_scene=EXPECTED_VIEWS_PER_SCENE,
    image_count=EXPECTED_SCENE_COUNT * EXPECTED_VIEWS_PER_SCENE,
    asset_group_count=EXPECTED_SCENE_COUNT,
)

_COHORT_CONTRACTS = {
    contract.cohort_id: contract
    for contract in (SMOKE20_CONTRACT, DS1_SCREEN_CONTRACT)
}

_FORBIDDEN_MODEL_FIELD_PARTS = (
    "anomaly",
    "asset",
    "category",
    "crosswalk",
    "label",
    "official_split",
    "pilot",
    "role",
    "source",
)


@dataclass(frozen=True)
class PackageAudit:
    """Safe aggregate evidence for one model-input package."""

    cohort_id: str
    release_id: str
    scene_count: int
    view_count: int
    image_count: int
    width: int
    height: int
    media_mode: str
    total_media_bytes: int
    manifest_sha256: str
    membership_sha256: str
    content_checksums_sha256: str
    verified_image_count: int
    duplicate_archive_member_count: int
    duplicate_package_path_count: int
    decoded_image_failure_count: int
    all_checks_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file as a list of objects."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def membership_sha256(opaque_scene_ids: Iterable[str]) -> str:
    """Hash one sorted set of opaque scene IDs."""

    payload = "\n".join(sorted(str(value) for value in opaque_scene_ids))
    return sha256(payload.encode("utf-8")).hexdigest()


def _stable_tie_break(opaque_scene_id: str) -> str:
    payload = f"{SMOKE20_SELECTION_SEED}|{opaque_scene_id}".encode("utf-8")
    return sha256(payload).hexdigest()


def choose_smoke20_rows(
    development_source_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_frozen_hash: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose a broad 20-scene smoke cohort inside the fixed 100-scene screen."""

    candidates = [
        dict(row)
        for row in development_source_rows
        if row.get("role") == "development"
        and row.get("pilot_stage") == "screen_100"
    ]
    if len(candidates) != 100:
        raise ValueError(f"Expected 100 screen scenes, found {len(candidates)}")

    required_fields = {
        "opaque_scene_id",
        "opaque_asset_group_id",
        "category",
        "object_count",
        "anomaly_count",
        "raw_anomaly_prefixes",
    }
    for row in candidates:
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(f"Screen row is missing fields: {sorted(missing)}")

    selected: list[dict[str, Any]] = []
    selected_assets: set[str] = set()
    covered_categories: set[str] = set()
    covered_prefixes: set[str] = set()
    covered_strata: set[tuple[int, int]] = set()

    while len(selected) < EXPECTED_SCENE_COUNT:
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for row in candidates:
            asset_id = str(row["opaque_asset_group_id"])
            if asset_id in selected_assets:
                continue
            prefixes = {str(value) for value in row["raw_anomaly_prefixes"]}
            stratum = (int(row["object_count"]), int(row["anomaly_count"]))
            score = (
                100 * len(prefixes.difference(covered_prefixes))
                + 50 * int(stratum not in covered_strata)
                + 10 * int(str(row["category"]) not in covered_categories)
            )
            ranked.append(
                (-score, _stable_tie_break(str(row["opaque_scene_id"])), row)
            )
        if not ranked:
            raise ValueError("Could not complete the smoke cohort")
        ranked.sort(key=lambda item: (item[0], item[1]))
        chosen = ranked[0][2]
        selected.append(chosen)
        selected_assets.add(str(chosen["opaque_asset_group_id"]))
        covered_categories.add(str(chosen["category"]))
        covered_prefixes.update(str(value) for value in chosen["raw_anomaly_prefixes"])
        covered_strata.add(
            (int(chosen["object_count"]), int(chosen["anomaly_count"]))
        )

    cohort_hash = membership_sha256(row["opaque_scene_id"] for row in selected)
    if enforce_frozen_hash and cohort_hash != SMOKE20_MEMBERSHIP_SHA256:
        raise ValueError("The smoke cohort does not match the frozen membership hash")
    if len(selected_assets) != EXPECTED_SCENE_COUNT:
        raise ValueError("The smoke cohort must use one scene per asset group")
    if len(covered_categories) != EXPECTED_SCENE_COUNT:
        raise ValueError("The smoke cohort must cover 20 different categories")
    if len(covered_prefixes) != 8:
        raise ValueError("The smoke cohort does not cover all anomaly prefixes")
    if len(covered_strata) != 5:
        raise ValueError("The smoke cohort does not cover all object/anomaly strata")

    audit = {
        "cohort_id": SMOKE20_COHORT_ID,
        "membership_sha256": cohort_hash,
        "scene_count": len(selected),
        "asset_group_count": len(selected_assets),
        "category_count": len(covered_categories),
        "prefix_coverage_count": len(covered_prefixes),
        "stratum_coverage_count": len(covered_strata),
        "selection_seed": SMOKE20_SELECTION_SEED,
        "all_checks_passed": True,
    }
    return selected, audit


def cohort_contract_for(cohort_id: str) -> CohortContract:
    """Return the frozen contract for a supported model-safe cohort."""

    try:
        return _COHORT_CONTRACTS[cohort_id]
    except KeyError as error:
        raise ValueError(f"Unsupported model-safe cohort: {cohort_id}") from error


def select_contract_rows(
    development_source_rows: Sequence[Mapping[str, Any]],
    contract: CohortContract,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select and verify every trusted row declared by a cohort contract."""

    selected = [
        dict(row)
        for row in development_source_rows
        if row.get("role") == contract.role
        and row.get("pilot_stage") == contract.pilot_stage
    ]
    required_fields = {
        "opaque_scene_id",
        "opaque_asset_group_id",
        "category",
        "object_count",
        "anomaly_count",
        "raw_anomaly_prefixes",
    }
    for row in selected:
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(f"Cohort row is missing fields: {sorted(missing)}")
    if len(selected) != contract.scene_count:
        raise ValueError(
            f"Expected {contract.scene_count} cohort scenes, found {len(selected)}"
        )
    scene_ids = [str(row["opaque_scene_id"]) for row in selected]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Cohort contains duplicate opaque scene IDs")
    observed_membership = membership_sha256(scene_ids)
    if observed_membership != contract.membership_sha256:
        raise ValueError("Cohort does not match the frozen membership hash")
    asset_ids = {str(row["opaque_asset_group_id"]) for row in selected}
    if len(asset_ids) != contract.asset_group_count:
        raise ValueError(
            "Cohort asset-group count does not match the frozen contract"
        )
    categories = {str(row["category"]) for row in selected}
    prefixes = {
        str(prefix)
        for row in selected
        for prefix in row["raw_anomaly_prefixes"]
    }
    strata = {
        (int(row["object_count"]), int(row["anomaly_count"]))
        for row in selected
    }
    audit = {
        "cohort_id": contract.cohort_id,
        "membership_sha256": observed_membership,
        "scene_count": len(selected),
        "asset_group_count": len(asset_ids),
        "category_count": len(categories),
        "prefix_coverage_count": len(prefixes),
        "stratum_coverage_count": len(strata),
        "all_checks_passed": True,
    }
    return sorted(selected, key=lambda row: str(row["opaque_scene_id"])), audit


def _assert_no_forbidden_fields(value: Any, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _FORBIDDEN_MODEL_FIELD_PARTS):
                raise ValueError(f"Forbidden model-facing field at {location}.{key}")
            _assert_no_forbidden_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_fields(child, f"{location}[{index}]")


def _validate_rgb_bytes(data: bytes, expected_size: tuple[int, int]) -> str:
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError("Packaged RGB media must be PNG")
        if image.mode == "RGBA":
            if image.getchannel("A").getextrema() != (255, 255):
                raise ValueError("Packaged RGB media cannot contain transparency")
        elif image.mode != "RGB":
            raise ValueError("Packaged RGB media must use RGB or opaque RGBA mode")
        if image.size != expected_size:
            raise ValueError(
                f"Unexpected RGB dimensions: {image.size} != {expected_size}"
            )
        return image.mode


def _read_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"Malformed checksum line {line_number}: {path.name}") from error
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe checksum path in {path.name}")
        if relative in checksums:
            raise ValueError(f"Duplicate checksum path in {path.name}")
        if not _is_sha256(digest):
            raise ValueError(f"Invalid SHA-256 digest in {path.name}")
        checksums[relative] = digest
    if not checksums:
        raise ValueError(f"Checksum manifest is empty: {path.name}")
    return checksums


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _find_extracted_archive_root(source_root: Path) -> Path | None:
    """Locate the extracted directory that contains the archive member paths."""

    for candidate in (source_root / "toysAD8K", source_root):
        if (candidate / "toysad8k").is_dir():
            return candidate
    return None


def _read_extracted_member(
    extracted_root: Path,
    source_name: str,
    *,
    expected_crc32: int,
    expected_size: int,
) -> bytes:
    """Read one extracted archive member after path, size, and CRC checks."""

    relative = Path(source_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Extracted source member has an unsafe path")
    member_path = extracted_root / relative
    if member_path.is_symlink() or not member_path.is_file():
        raise ValueError("A required RGB member is missing or unsafe")
    data = member_path.read_bytes()
    actual_crc = binascii.crc32(data) & 0xFFFFFFFF
    if actual_crc != expected_crc32 or len(data) != expected_size:
        raise ValueError("Current RGB bytes fail CRC or size verification")
    return data


def verify_checksum_manifest(
    root: Path,
    *,
    manifest_name: str,
    required_names: Sequence[str] | None = None,
) -> dict[str, str]:
    """Verify selected files against one accepted checksum manifest."""

    root = root.expanduser().resolve()
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checksum manifest is missing: {manifest_name}")
    declared = _read_checksum_manifest(manifest_path)
    names = sorted(required_names or declared)
    missing = set(names).difference(declared)
    if missing:
        raise ValueError(f"Checksum manifest is missing required files: {sorted(missing)}")
    verified: dict[str, str] = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Checksummed input is missing or unsafe: {name}")
        actual = file_sha256(path)
        if actual != declared[name]:
            raise ValueError(f"Input checksum mismatch: {name}")
        verified[name] = actual
    verified["checksum_manifest_sha256"] = file_sha256(manifest_path)
    return verified


def verify_trusted_package_inputs(
    *,
    selection_root: Path,
    development_root: Path,
    administrative_root: Path,
    source_root: Path,
    verify_archive_bytes: bool = True,
    expected_image_count: int = EXPECTED_SCENE_COUNT * EXPECTED_VIEWS_PER_SCENE,
) -> dict[str, Any]:
    """Verify the trusted inputs before building a model-facing package."""

    selection_hashes = verify_checksum_manifest(
        selection_root,
        manifest_name="checksums.sha256",
    )
    development_hashes = verify_checksum_manifest(
        development_root,
        manifest_name="checksums.sha256",
    )
    administrative_hashes = verify_checksum_manifest(
        administrative_root,
        manifest_name="checksums.sha256",
    )
    accepted_manifest_hashes = {
        "selection": EXPECTED_SELECTION_CHECKSUM_MANIFEST_SHA256,
        "development": EXPECTED_DEVELOPMENT_CHECKSUM_MANIFEST_SHA256,
        "administrative": EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256,
    }
    observed_manifest_hashes = {
        "selection": selection_hashes["checksum_manifest_sha256"],
        "development": development_hashes["checksum_manifest_sha256"],
        "administrative": administrative_hashes["checksum_manifest_sha256"],
    }
    for name, expected_hash in accepted_manifest_hashes.items():
        if observed_manifest_hashes[name] != expected_hash:
            raise ValueError(f"The {name} checksum manifest is not the accepted version")
    source_root = source_root.expanduser().resolve()
    source_checksum_path = source_root / "package-checksums.sha256"
    if file_sha256(source_checksum_path) != EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256:
        raise ValueError("The source checksum manifest is not the accepted version")
    source_manifest = _read_checksum_manifest(source_checksum_path)
    archive_path = source_root / "toysAD8K.zip"
    if source_manifest.get("toysAD8K.zip") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("The source checksum manifest identifies another archive")
    extracted_root = _find_extracted_archive_root(source_root)
    if archive_path.is_file() and not archive_path.is_symlink():
        if archive_path.stat().st_size != EXPECTED_ARCHIVE_SIZE_BYTES:
            raise ValueError("The source archive size differs from the accepted source")
        archive_sha256 = (
            file_sha256(archive_path) if verify_archive_bytes else EXPECTED_ARCHIVE_SHA256
        )
        if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("The source archive checksum differs from the accepted source")
        source_access_mode = "zip_archive"
        archive_bytes_verified = verify_archive_bytes
        identity_file_count = 1
    elif extracted_root is not None:
        identity_names = (
            "archive-manifest.json",
            "archive_summary.json",
            "source-chunks-manifest.json",
        )
        for name in identity_names:
            expected_digest = source_manifest.get(name)
            if expected_digest is None or file_sha256(source_root / name) != expected_digest:
                raise ValueError(f"Extracted source identity mismatch: {name}")
        archive_sha256 = EXPECTED_ARCHIVE_SHA256
        source_access_mode = "kaggle_extracted_members"
        archive_bytes_verified = False
        identity_file_count = len(identity_names)
    else:
        raise ValueError("Neither the accepted source archive nor its extracted tree was found")

    config_paths = (
        (selection_root / "release_config.json", "release_id"),
        (development_root / "workspace_config.json", "release_id"),
        (administrative_root / "administrative_config.json", "release_id"),
    )
    for config_path, release_key in config_paths:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get(release_key) != RELEASE_ID:
            raise ValueError(f"Trusted input has another release ID: {config_path.name}")

    return {
        "release_id": RELEASE_ID,
        "selection_checksum_manifest_sha256": selection_hashes[
            "checksum_manifest_sha256"
        ],
        "development_checksum_manifest_sha256": development_hashes[
            "checksum_manifest_sha256"
        ],
        "administrative_checksum_manifest_sha256": administrative_hashes[
            "checksum_manifest_sha256"
        ],
        "source_checksum_manifest_sha256": file_sha256(
            source_root / "package-checksums.sha256"
        ),
        "source_archive_sha256": archive_sha256,
        "source_archive_size_bytes": EXPECTED_ARCHIVE_SIZE_BYTES,
        "source_archive_bytes_verified": archive_bytes_verified,
        "source_access_mode": source_access_mode,
        "source_identity_file_count_verified": identity_file_count,
        "selected_source_members_verified": expected_image_count,
        "all_checks_passed": True,
    }


def _write_bytes_if_consistent(path: Path, data: bytes) -> bool:
    """Write one file atomically, or verify the existing bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing output differs: {path}")
        return False
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return True


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _content_checksum_bytes(model_input_root: Path, relative_paths: Sequence[Path]) -> bytes:
    lines = [
        f"{file_sha256(model_input_root / relative)}  {relative.as_posix()}\n"
        for relative in sorted(relative_paths, key=lambda value: value.as_posix())
    ]
    return "".join(lines).encode("utf-8")


def _build_selected_rgb_package(
    *,
    selection_root: Path,
    development_root: Path,
    administrative_root: Path,
    source_root: Path,
    output_root: Path,
    contract: CohortContract,
    selected_rows: Sequence[Mapping[str, Any]],
    selection_audit: Mapping[str, Any],
    verify_archive_bytes: bool = True,
) -> tuple[PackageAudit, dict[str, Any]]:
    """Build one resumable opaque RGB package from verified trusted rows."""

    if contract.release_id != RELEASE_ID:
        raise ValueError("Cohort contract identifies another selection release")
    if len(selected_rows) != contract.scene_count:
        raise ValueError("Selected rows do not match the cohort scene count")
    selected_ids = {str(row["opaque_scene_id"]) for row in selected_rows}
    if len(selected_ids) != contract.scene_count:
        raise ValueError("Selected rows contain duplicate opaque scene IDs")
    if membership_sha256(selected_ids) != contract.membership_sha256:
        raise ValueError("Selected rows do not match the cohort membership hash")

    input_audit = verify_trusted_package_inputs(
        selection_root=selection_root,
        development_root=development_root,
        administrative_root=administrative_root,
        source_root=source_root,
        verify_archive_bytes=verify_archive_bytes,
        expected_image_count=contract.image_count,
    )
    neutral_rows = read_jsonl(selection_root / "neutral_corpus_manifest.jsonl")
    required_rows = read_jsonl(
        administrative_root / "required_source_members.jsonl"
    )

    neutral_by_id = {str(row["opaque_scene_id"]): row for row in neutral_rows}
    if len(neutral_by_id) != len(neutral_rows):
        raise ValueError("Neutral manifest contains duplicate opaque scene IDs")
    selected_neutral: list[dict[str, Any]] = []
    for opaque_scene_id in sorted(selected_ids):
        if opaque_scene_id not in neutral_by_id:
            raise ValueError("A cohort scene is missing from the neutral manifest")
        row = dict(neutral_by_id[opaque_scene_id])
        validate_model_facing_media_record(row)
        _assert_no_forbidden_fields(row)
        selected_neutral.append(row)

    rgb_rows = [
        row
        for row in required_rows
        if str(row.get("opaque_scene_id")) in selected_ids
        and row.get("component") == "rgb"
        and row.get("include_in_media_package") is True
    ]
    expected_images = contract.image_count
    if len(rgb_rows) != expected_images:
        raise ValueError(f"Expected {expected_images} RGB members, found {len(rgb_rows)}")
    source_names = [str(row["source_member"]) for row in rgb_rows]
    target_names = [str(row["target_member"]) for row in rgb_rows]
    duplicate_source_count = len(source_names) - len(set(source_names))
    duplicate_target_count = len(target_names) - len(set(target_names))
    if duplicate_source_count or duplicate_target_count:
        raise ValueError("Required RGB members contain duplicate source or target paths")

    model_input_root = output_root.expanduser().resolve() / "model-input"
    extracted_count = 0
    resumed_count = 0
    decoded_failures = 0
    media_modes: Counter[str] = Counter()
    total_media_bytes = 0
    media_relative_paths: list[Path] = []
    archive_path = source_root / "toysAD8K.zip"
    extracted_root = _find_extracted_archive_root(source_root)
    archive = zipfile.ZipFile(archive_path) if archive_path.is_file() else None
    try:
        required_source_names = set(source_names)
        if archive is not None:
            archive_name_counts = Counter(
                info.filename
                for info in archive.infolist()
                if info.filename in required_source_names
            )
            duplicate_archive_entries = sum(
                max(0, count - 1) for count in archive_name_counts.values()
            )
            missing_archive_entries = required_source_names.difference(archive_name_counts)
            if missing_archive_entries:
                raise ValueError("Required RGB members are missing from the archive")
            if duplicate_archive_entries:
                raise ValueError("Required RGB members are duplicated in the archive")
        else:
            if extracted_root is None:
                raise ValueError("The extracted source tree is unavailable")
            duplicate_archive_entries = 0
        for row in sorted(rgb_rows, key=lambda value: str(value["target_member"])):
            source_name = str(row["source_member"])
            target_name = str(row["target_member"])
            opaque_scene_id = str(row["opaque_scene_id"])
            view_id = str(row["view_id"])
            expected_target = f"media/{opaque_scene_id}/rgb/{view_id}.png"
            if target_name != expected_target:
                raise ValueError("Required RGB target does not follow the opaque schema")
            expected_crc = int(str(row["crc32"]), 16)
            expected_size = int(row["uncompressed_size"])
            if archive is not None:
                try:
                    info = archive.getinfo(source_name)
                except KeyError as error:
                    raise ValueError("A required RGB member is missing from the archive") from error
                if info.CRC != expected_crc:
                    raise ValueError("Archive CRC differs from the trusted member record")
                data = archive.read(source_name)
                actual_crc = binascii.crc32(data) & 0xFFFFFFFF
                if actual_crc != info.CRC or len(data) != expected_size:
                    raise ValueError("Current RGB bytes fail CRC or size verification")
            else:
                data = _read_extracted_member(
                    extracted_root,
                    source_name,
                    expected_crc32=expected_crc,
                    expected_size=expected_size,
                )
            try:
                media_modes[_validate_rgb_bytes(data, EXPECTED_RGB_SIZE)] += 1
            except Exception:
                decoded_failures += 1
                raise
            total_media_bytes += len(data)
            relative_target = Path(target_name)
            media_relative_paths.append(relative_target)
            if _write_bytes_if_consistent(model_input_root / relative_target, data):
                extracted_count += 1
            else:
                resumed_count += 1
    finally:
        if archive is not None:
            archive.close()

    manifest_relative = Path("neutral_corpus_manifest.jsonl")
    manifest_bytes = _jsonl_bytes(selected_neutral)
    _write_bytes_if_consistent(model_input_root / manifest_relative, manifest_bytes)
    content_paths = [manifest_relative, *media_relative_paths]
    checksum_bytes = _content_checksum_bytes(model_input_root, content_paths)
    checksum_relative = Path("content_checksums.sha256")
    _write_bytes_if_consistent(model_input_root / checksum_relative, checksum_bytes)

    package_audit = PackageAudit(
        cohort_id=contract.cohort_id,
        release_id=contract.release_id,
        scene_count=contract.scene_count,
        view_count=contract.views_per_scene,
        image_count=expected_images,
        width=EXPECTED_RGB_SIZE[0],
        height=EXPECTED_RGB_SIZE[1],
        media_mode=(
            next(iter(media_modes)) if len(media_modes) == 1 else "mixed"
        ),
        total_media_bytes=total_media_bytes,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        membership_sha256=membership_sha256(selected_ids),
        content_checksums_sha256=sha256(checksum_bytes).hexdigest(),
        verified_image_count=expected_images,
        duplicate_archive_member_count=(
            duplicate_source_count + duplicate_archive_entries
        ),
        duplicate_package_path_count=duplicate_target_count,
        decoded_image_failure_count=decoded_failures,
        all_checks_passed=True,
    )
    record_bytes = json.dumps(
        package_audit.to_dict(), indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _write_bytes_if_consistent(model_input_root / "package_record.json", record_bytes)
    selection_audit = dict(selection_audit)
    selection_audit["files_written_this_run"] = extracted_count
    selection_audit["files_reused_this_run"] = resumed_count
    selection_audit["trusted_input_audit"] = input_audit
    verified = verify_model_input_package(model_input_root, contract=contract)
    if verified.manifest_sha256 != package_audit.manifest_sha256:
        raise ValueError("Package verification returned a different manifest hash")
    return package_audit, selection_audit


def build_smoke20_rgb_package(
    *,
    selection_root: Path,
    development_root: Path,
    administrative_root: Path,
    source_root: Path,
    output_root: Path,
    verify_archive_bytes: bool = True,
) -> tuple[PackageAudit, dict[str, Any]]:
    """Build a resumable opaque colour package for the frozen smoke cohort."""

    development_rows = read_jsonl(development_root / "development_source_map.jsonl")
    selected_rows, selection_audit = choose_smoke20_rows(development_rows)
    return _build_selected_rgb_package(
        selection_root=selection_root,
        development_root=development_root,
        administrative_root=administrative_root,
        source_root=source_root,
        output_root=output_root,
        contract=SMOKE20_CONTRACT,
        selected_rows=selected_rows,
        selection_audit=selection_audit,
        verify_archive_bytes=verify_archive_bytes,
    )


def build_cohort_rgb_package(
    *,
    selection_root: Path,
    development_root: Path,
    administrative_root: Path,
    source_root: Path,
    output_root: Path,
    contract: CohortContract,
    verify_archive_bytes: bool = True,
) -> tuple[PackageAudit, dict[str, Any]]:
    """Build a complete model-safe package for a frozen cohort contract."""

    development_rows = read_jsonl(development_root / "development_source_map.jsonl")
    selected_rows, selection_audit = select_contract_rows(
        development_rows,
        contract,
    )
    return _build_selected_rgb_package(
        selection_root=selection_root,
        development_root=development_root,
        administrative_root=administrative_root,
        source_root=source_root,
        output_root=output_root,
        contract=contract,
        selected_rows=selected_rows,
        selection_audit=selection_audit,
        verify_archive_bytes=verify_archive_bytes,
    )


def verify_model_input_package(
    model_input_root: Path,
    *,
    contract: CohortContract | None = None,
) -> PackageAudit:
    """Verify a model-safe package without any trusted source records."""

    model_input_root = model_input_root.expanduser().resolve()
    manifest_path = model_input_root / "neutral_corpus_manifest.jsonl"
    record_path = model_input_root / "package_record.json"
    checksum_path = model_input_root / "content_checksums.sha256"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    _assert_no_forbidden_fields(record)
    resolved_contract = contract or cohort_contract_for(str(record.get("cohort_id")))
    rows = read_jsonl(manifest_path)
    if len(rows) != resolved_contract.scene_count:
        raise ValueError(
            "Model input manifest scene count does not match the cohort contract"
        )
    scene_ids: list[str] = []
    expected_media: list[Path] = []
    for row in rows:
        validate_model_facing_media_record(row)
        _assert_no_forbidden_fields(row)
        scene_id = str(row["opaque_scene_id"])
        scene_ids.append(scene_id)
        view_ids = [str(value) for value in row["view_ids"]]
        if (
            len(view_ids) != resolved_contract.views_per_scene
            or len(set(view_ids)) != len(view_ids)
        ):
            raise ValueError(
                "Every model-input scene must contain the contracted unique views"
            )
        for view_id in view_ids:
            expected_media.append(Path(f"media/{scene_id}/rgb/{view_id}.png"))
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Model input manifest contains duplicate scenes")
    if membership_sha256(scene_ids) != resolved_contract.membership_sha256:
        raise ValueError("Model input manifest has the wrong cohort membership")

    actual_media = sorted(
        path.relative_to(model_input_root)
        for path in (model_input_root / "media").rglob("*.png")
    )
    if actual_media != sorted(expected_media):
        raise ValueError("Model input media paths do not match the manifest")
    expected_files = {
        Path("neutral_corpus_manifest.jsonl"),
        Path("package_record.json"),
        Path("content_checksums.sha256"),
        *expected_media,
    }
    actual_files: set[Path] = set()
    for path in model_input_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Model input package cannot contain symbolic links")
        if path.is_file():
            actual_files.add(path.relative_to(model_input_root))
    if actual_files != expected_files:
        extra = sorted((actual_files - expected_files), key=lambda value: value.as_posix())
        missing = sorted((expected_files - actual_files), key=lambda value: value.as_posix())
        raise ValueError(
            "Model input file allowlist mismatch: "
            f"extra={[value.as_posix() for value in extra]}, "
            f"missing={[value.as_posix() for value in missing]}"
        )
    media_modes: Counter[str] = Counter()
    total_media_bytes = 0
    for relative_path in actual_media:
        data = (model_input_root / relative_path).read_bytes()
        media_modes[_validate_rgb_bytes(data, EXPECTED_RGB_SIZE)] += 1
        total_media_bytes += len(data)

    checksum_lines = [line for line in checksum_path.read_text().splitlines() if line]
    checksum_map: dict[str, str] = {}
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        if relative in checksum_map:
            raise ValueError("Content checksum file contains duplicate paths")
        checksum_map[relative] = digest
    expected_content = {
        "neutral_corpus_manifest.jsonl",
        *(path.as_posix() for path in expected_media),
    }
    if set(checksum_map) != expected_content:
        raise ValueError("Content checksum paths do not match the model input")
    for relative, expected_digest in checksum_map.items():
        if file_sha256(model_input_root / relative) != expected_digest:
            raise ValueError("Model input content checksum mismatch")

    audit = PackageAudit(**record)
    if (
        audit.cohort_id != resolved_contract.cohort_id
        or audit.release_id != resolved_contract.release_id
    ):
        raise ValueError("Package record identifies a different cohort or release")
    derived = {
        "scene_count": len(rows),
        "view_count": resolved_contract.views_per_scene,
        "image_count": len(actual_media),
        "width": EXPECTED_RGB_SIZE[0],
        "height": EXPECTED_RGB_SIZE[1],
        "media_mode": next(iter(media_modes)) if len(media_modes) == 1 else "mixed",
        "total_media_bytes": total_media_bytes,
        "manifest_sha256": file_sha256(manifest_path),
        "membership_sha256": membership_sha256(scene_ids),
        "content_checksums_sha256": file_sha256(checksum_path),
        "verified_image_count": len(actual_media),
        "duplicate_archive_member_count": 0,
        "duplicate_package_path_count": 0,
        "decoded_image_failure_count": 0,
        "all_checks_passed": True,
    }
    for field_name, expected_value in derived.items():
        if getattr(audit, field_name) != expected_value:
            raise ValueError(f"Package record field mismatch: {field_name}")
    return audit


def collect_model_input_evidence(
    model_input_root: Path,
    *,
    contract: CohortContract | None = None,
) -> dict[str, Any]:
    """Collect safe tables and plotting values from a verified package."""

    model_input_root = model_input_root.expanduser().resolve()
    audit = verify_model_input_package(model_input_root, contract=contract)
    resolved_contract = contract or cohort_contract_for(audit.cohort_id)
    rows = read_jsonl(model_input_root / "neutral_corpus_manifest.jsonl")
    view_counts: Counter[str] = Counter()
    scene_rows: list[dict[str, Any]] = []
    media_sizes_kib: list[float] = []
    sample_items: list[dict[str, Any]] = []
    sample_view_ids = {"V00", "V05", "V10", "V15", "V19"}
    for scene_index, row in enumerate(
        sorted(rows, key=lambda value: str(value["opaque_scene_id"])), 1
    ):
        scene_id = str(row["opaque_scene_id"])
        view_ids = [str(value) for value in row["view_ids"]]
        scene_rows.append(
            {
                "scene_index": scene_index,
                "opaque_scene_id": scene_id,
                "view_count": len(view_ids),
                "expected_view_count": resolved_contract.views_per_scene,
                "complete": len(view_ids) == resolved_contract.views_per_scene,
            }
        )
        for view_id in view_ids:
            view_counts[view_id] += 1
            image_path = model_input_root / "media" / scene_id / "rgb" / f"{view_id}.png"
            media_sizes_kib.append(image_path.stat().st_size / 1024)
            if scene_index <= 4 and view_id in sample_view_ids:
                sample_items.append(
                    {
                        "label": f"Scene {scene_index:02d} / {view_id}",
                        "path": image_path,
                    }
                )
    view_rows = [
        {
            "view_id": view_id,
            "scene_count": view_counts[view_id],
            "expected_scene_count": resolved_contract.scene_count,
            "complete": view_counts[view_id] == resolved_contract.scene_count,
        }
        for view_id in sorted(view_counts)
    ]
    gate_rows = [
        {
            "gate": "scene count",
            "status": "pass",
            "observed": audit.scene_count,
            "expected": resolved_contract.scene_count,
        },
        {
            "gate": "view count per scene",
            "status": "pass",
            "observed": audit.view_count,
            "expected": resolved_contract.views_per_scene,
        },
        {
            "gate": "image count",
            "status": "pass",
            "observed": audit.image_count,
            "expected": resolved_contract.image_count,
        },
        {
            "gate": "verified images",
            "status": "pass",
            "observed": audit.verified_image_count,
            "expected": resolved_contract.image_count,
        },
        {
            "gate": "duplicate package paths",
            "status": "pass",
            "observed": audit.duplicate_package_path_count,
            "expected": 0,
        },
        {
            "gate": "duplicate archive members",
            "status": "pass",
            "observed": audit.duplicate_archive_member_count,
            "expected": 0,
        },
        {
            "gate": "decode failures",
            "status": "pass",
            "observed": audit.decoded_image_failure_count,
            "expected": 0,
        },
        {
            "gate": "exact file allowlist",
            "status": "pass",
            "observed": audit.image_count + 3,
            "expected": resolved_contract.image_count + 3,
        },
    ]
    media_summary = {
        "image_count": len(media_sizes_kib),
        "total_media_mib": round(audit.total_media_bytes / (1024 * 1024), 3),
        "minimum_image_kib": round(min(media_sizes_kib), 3),
        "median_image_kib": round(median(media_sizes_kib), 3),
        "mean_image_kib": round(mean(media_sizes_kib), 3),
        "maximum_image_kib": round(max(media_sizes_kib), 3),
        "media_mode": audit.media_mode,
        "width": audit.width,
        "height": audit.height,
    }
    return {
        "audit": audit.to_dict(),
        "gate_rows": gate_rows,
        "scene_rows": scene_rows,
        "view_rows": view_rows,
        "media_summary": media_summary,
        "media_sizes_kib": media_sizes_kib,
        "sample_items": sample_items,
    }


__all__ = [
    "EXPECTED_RGB_SIZE",
    "EXPECTED_ARCHIVE_SHA256",
    "EXPECTED_ARCHIVE_SIZE_BYTES",
    "EXPECTED_SCENE_COUNT",
    "EXPECTED_VIEWS_PER_SCENE",
    "PackageAudit",
    "SMOKE20_COHORT_ID",
    "SMOKE20_CONTRACT",
    "SMOKE20_MEMBERSHIP_SHA256",
    "build_cohort_rgb_package",
    "build_smoke20_rgb_package",
    "canonical_json_bytes",
    "cohort_contract_for",
    "choose_smoke20_rows",
    "collect_model_input_evidence",
    "file_sha256",
    "membership_sha256",
    "read_jsonl",
    "select_contract_rows",
    "verify_model_input_package",
    "verify_checksum_manifest",
    "verify_trusted_package_inputs",
]
