"""Verification contract for the private DS2 route reuse input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.confirmation_mask_routes import (
    verify_ds2_mask_route_package,
)
from visual_peer_inspector.data.model_safe_package import file_sha256

PACKAGE_KIND = "ds2-frozen-route-reuse"
SOURCE_FILES = {
    "selected_prompt.json",
    "locateanything_run_record.json",
    "automatic_sam_run_record.json",
    "reference_sam_run_record.json",
}
ALLOWED_FILES = SOURCE_FILES | {
    "package_record.json",
    "content_checksums.sha256",
    "mask-routes/mask_routes.jsonl",
    "mask-routes/package_record.json",
    "mask-routes/content_checksums.sha256",
}


def _read_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def checksum_inventory(root: Path) -> list[str]:
    """Return the canonical checksum rows covered by the package record."""

    excluded = {"package_record.json", "content_checksums.sha256"}
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("DS2 reuse input cannot contain symbolic links")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            lines.append(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}")
    return lines


def verify_ds2_reuse_input(root: Path) -> dict[str, Any]:
    """Verify package coverage, file allowlist, checksums and route bindings."""

    root = root.expanduser().resolve()
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("DS2 reuse input is missing or unsafe")
    record = _read_record(root / "package_record.json")
    expected = {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "release_id": DS2_CONFIRM_CONTRACT.release_id,
        "membership_sha256": DS2_CONFIRM_CONTRACT.membership_sha256,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "image_count": DS2_CONFIRM_CONTRACT.image_count,
        "views_per_scene": DS2_CONFIRM_CONTRACT.views_per_scene,
        "all_checks_passed": True,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"DS2 reuse input mismatch: {key}")

    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual != ALLOWED_FILES:
        raise ValueError("DS2 reuse input file allowlist differs")
    expected_checksums = "\n".join(checksum_inventory(root)) + "\n"
    checksum_path = root / "content_checksums.sha256"
    if checksum_path.read_text(encoding="utf-8") != expected_checksums:
        raise ValueError("DS2 reuse input checksum inventory differs")
    if file_sha256(checksum_path) != record.get("content_checksums_sha256"):
        raise ValueError("DS2 reuse input checksum digest differs")

    routes = verify_ds2_mask_route_package(root / "mask-routes")
    bindings = {
        "selected_prompt_sha256": file_sha256(root / "selected_prompt.json"),
        "locate_run_record_sha256": file_sha256(
            root / "locateanything_run_record.json"
        ),
        "automatic_sam_run_record_sha256": file_sha256(
            root / "automatic_sam_run_record.json"
        ),
        "reference_sam_run_record_sha256": file_sha256(
            root / "reference_sam_run_record.json"
        ),
        "mask_route_package_record_sha256": file_sha256(
            root / "mask-routes/package_record.json"
        ),
        "mask_routes_sha256": routes["mask_routes_sha256"],
    }
    for key, value in bindings.items():
        if record.get(key) != value:
            raise ValueError(f"DS2 reuse input binding differs: {key}")
    return record
