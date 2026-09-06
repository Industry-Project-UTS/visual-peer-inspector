"""Verify the primary split and release only the remaining development media."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from visual_peer_inspector.data.cohort_contract import (
    CALIBRATION_CONTRACT,
    DEVELOPMENT_FULL_CONTRACT,
    DEVELOPMENT_REMAINDER_CONTRACT,
    DS1_SCREEN_CONTRACT,
    DS2_CONFIRM_CONTRACT,
    INTERNAL_LOCK_CONTRACT,
    CohortContract,
)
from visual_peer_inspector.data.model_safe_package import (
    build_cohort_rgb_package,
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    read_jsonl,
    select_contract_rows,
    verify_model_input_package,
)
from visual_peer_inspector.data.toysad_selection import (
    ADMINISTRATIVE_VAULT_FILES,
    audit_routine_release,
    verify_release_checksums,
)
from visual_peer_inspector.evidence import (
    save_bar_chart,
    save_image_grid,
    write_csv_rows,
)
from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
)

STAGE = "primary_protocol_release"
DEVELOPMENT_CONTRACTS = (
    DS1_SCREEN_CONTRACT,
    DS2_CONFIRM_CONTRACT,
    DEVELOPMENT_REMAINDER_CONTRACT,
)
PRIMARY_CONTRACTS = (
    DEVELOPMENT_FULL_CONTRACT,
    CALIBRATION_CONTRACT,
    INTERNAL_LOCK_CONTRACT,
)


def _select_identity_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: CohortContract,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify a cohort using only role, scene, and asset-group identity fields."""

    selected = [
        dict(row)
        for row in rows
        if row.get("role") == contract.role
        and (
            contract.pilot_stage == "all_development"
            or row.get("pilot_stage") == contract.pilot_stage
        )
    ]
    required_fields = {"opaque_scene_id", "opaque_asset_group_id"}
    for row in selected:
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(f"Protocol row is missing fields: {sorted(missing)}")
    if len(selected) != contract.scene_count:
        raise ValueError(
            f"Expected {contract.scene_count} protocol scenes, found {len(selected)}"
        )
    scene_ids = [str(row["opaque_scene_id"]) for row in selected]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Protocol cohort contains duplicate scene IDs")
    observed_membership = membership_sha256(scene_ids)
    if observed_membership != contract.membership_sha256:
        raise ValueError("Protocol cohort membership hash differs")
    asset_ids = {str(row["opaque_asset_group_id"]) for row in selected}
    if len(asset_ids) != contract.asset_group_count:
        raise ValueError("Protocol cohort asset-group count differs")
    return sorted(selected, key=lambda row: str(row["opaque_scene_id"])), {
        "scene_count": len(scene_ids),
        "asset_group_count": len(asset_ids),
        "membership_sha256": observed_membership,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_manifest(root: Path) -> dict[str, Any]:
    run_record_path = root / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Protocol evidence cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    record = {
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_content_sha256": sha256(
            b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        ).hexdigest(),
    }
    _write_json(root / "artifact_manifest.json", record)
    return record


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _protocol_diagram(path: Path) -> None:
    canvas = Image.new("RGB", (1500, 680), "white")
    draw = ImageDraw.Draw(canvas)
    title = _font(31, bold=True)
    heading = _font(20, bold=True)
    body = _font(17)
    draw.text((48, 28), "Primary experiment protocol", fill="#1f2933", font=title)

    boxes = (
        ((45, 140, 345, 300), "Fixed 2,000-scene split", "1,000 development\n600 calibration\n400 internal lock", "#dceaf5"),
        ((445, 90, 785, 250), "Development path", "100 screen + 300 confirmation\n+ 600 remaining scenes", "#dff1e1"),
        ((445, 285, 785, 445), "Calibration path", "Membership verified now\nRGB remains sealed until 09A", "#fff3cd"),
        ((445, 480, 785, 640), "Internal-lock path", "Membership verified now\nRGB remains sealed until 09C", "#fde2e1"),
        ((905, 90, 1245, 250), "Notebook 08D release", "12,000 development RGB images\nNo anomaly labels", "#dff1e1"),
        ((905, 285, 1245, 445), "Notebook 09A release", "Only after method freeze\nLabels remain sealed", "#fff3cd"),
        ((905, 480, 1245, 640), "Notebook 09C release", "Only after policy freeze\nLabels remain sealed", "#fde2e1"),
    )
    for coordinates, label, detail, colour in boxes:
        draw.rounded_rectangle(coordinates, radius=16, fill=colour, outline="#24557f", width=3)
        x1, y1, _x2, _y2 = coordinates
        draw.text((x1 + 20, y1 + 20), label, fill="#1f2933", font=heading)
        draw.multiline_text((x1 + 20, y1 + 64), detail, fill="#1f2933", font=body, spacing=8)

    arrow_colour = "#4b5563"
    for start, end in (
        ((345, 220), (445, 170)),
        ((345, 220), (445, 365)),
        ((345, 220), (445, 560)),
        ((785, 170), (905, 170)),
        ((785, 365), (905, 365)),
        ((785, 560), (905, 560)),
    ):
        draw.line((*start, *end), fill=arrow_colour, width=4)
        x, y = end
        draw.polygon(((x, y), (x - 15, y - 8), (x - 15, y + 8)), fill=arrow_colour)

    draw.rounded_rectangle((1280, 170, 1460, 560), radius=16, outline="#24557f", width=3)
    draw.text((1302, 194), "Boundary", fill="#1f2933", font=heading)
    draw.multiline_text(
        (1302, 246),
        "Model-facing RGB\ncontains no labels.\n\nCalibration and\nlock labels are\nopened only after\npredictions are\nhashed.",
        fill="#1f2933",
        font=body,
        spacing=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _contract_rows(
    role_rows: Sequence[Mapping[str, Any]],
    trusted_rows: Sequence[Mapping[str, Any]],
    contracts: Sequence[CohortContract] = PRIMARY_CONTRACTS,
    expected_primary_scene_count: int = 2_000,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    roles_by_id = {str(row["opaque_scene_id"]): row for row in role_rows}
    trusted_by_id = {str(row["opaque_scene_id"]): row for row in trusted_rows}
    if (
        len(roles_by_id) != expected_primary_scene_count
        or set(roles_by_id) != set(trusted_by_id)
    ):
        raise ValueError(
            "Protocol and trusted source maps must cover the same primary scenes"
        )

    rows = []
    scene_sets: dict[str, set[str]] = {}
    asset_sets: dict[str, set[str]] = {}
    for contract in contracts:
        selected, audit = _select_identity_rows(trusted_rows, contract)
        scene_ids = {str(row["opaque_scene_id"]) for row in selected}
        asset_ids = {str(row["opaque_asset_group_id"]) for row in selected}
        if any(
            roles_by_id[scene_id]["role"] != contract.role
            for scene_id in scene_ids
        ):
            raise ValueError("Trusted source role differs from the public protocol")
        scene_sets[contract.role] = scene_ids
        asset_sets[contract.role] = asset_ids
        rows.append(
            {
                "cohort": contract.cohort_id,
                "role": contract.role,
                "scenes": audit["scene_count"],
                "views": contract.image_count,
                "asset_groups": audit["asset_group_count"],
                "membership_sha256": audit["membership_sha256"],
                "membership_verified": True,
                "rgb_release": (
                    "complete_across_development_stages"
                    if contract.role == "development"
                    else "sealed"
                ),
            }
        )
    return rows, scene_sets, asset_sets


def audit_primary_protocol(
    *,
    selection_root: Path,
    development_root: Path,
    administrative_root: Path,
) -> dict[str, Any]:
    """Verify cohort memberships and separation without opening held-out labels."""

    routine = audit_routine_release(selection_root, development_root)
    administrative_hashes = verify_release_checksums(
        administrative_root,
        ADMINISTRATIVE_VAULT_FILES,
    )
    role_rows = read_jsonl(selection_root / "experiment_roles.jsonl")
    trusted_rows = read_jsonl(administrative_root / "restricted_source_map.jsonl")
    cohort_rows, scene_sets, asset_sets = _contract_rows(role_rows, trusted_rows)

    development_source_rows = read_jsonl(
        development_root / "development_source_map.jsonl"
    )
    development_parts = []
    development_stage_rows = []
    for contract in DEVELOPMENT_CONTRACTS:
        selected, stage_audit = select_contract_rows(development_source_rows, contract)
        development_parts.append({str(row["opaque_scene_id"]) for row in selected})
        development_stage_rows.append(
            {
                "cohort": contract.cohort_id,
                "stage": contract.pilot_stage,
                "scenes": stage_audit["scene_count"],
                "views": contract.image_count,
                "asset_groups": stage_audit["asset_group_count"],
                "membership_sha256": stage_audit["membership_sha256"],
                "membership_verified": True,
            }
        )
    if any(development_parts[left] & development_parts[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Development stages overlap")
    development_union = set().union(*development_parts)
    if development_union != scene_sets["development"]:
        raise ValueError("Development stages do not reconstruct the 1,000-scene role")
    if membership_sha256(development_union) != DEVELOPMENT_FULL_CONTRACT.membership_sha256:
        raise ValueError("Full development membership differs from its frozen contract")

    overlap_rows = []
    pairs = (
        ("development", "calibration"),
        ("development", "internal_lock"),
        ("calibration", "internal_lock"),
    )
    for left, right in pairs:
        scene_overlap = scene_sets[left] & scene_sets[right]
        asset_overlap = asset_sets[left] & asset_sets[right]
        overlap_rows.append(
            {
                "left_role": left,
                "right_role": right,
                "scene_overlap": len(scene_overlap),
                "asset_group_overlap": len(asset_overlap),
                "passed": not scene_overlap and not asset_overlap,
            }
        )
    if not all(row["passed"] for row in overlap_rows):
        raise ValueError("Primary roles overlap by scene or asset group")
    return {
        "schema_version": 1,
        "status": "verified",
        "release_id": DEVELOPMENT_REMAINDER_CONTRACT.release_id,
        "cohorts": cohort_rows,
        "development_stages": development_stage_rows,
        "overlaps": overlap_rows,
        "development_stage_counts": {
            "screen": len(development_parts[0]),
            "confirmation": len(development_parts[1]),
            "remaining": len(development_parts[2]),
            "full_development": len(development_union),
        },
        "routine_release_verified": bool(routine["passed"]),
        "administrative_file_count_verified": len(administrative_hashes) - 1,
        "calibration_labels_opened": False,
        "internal_lock_labels_opened": False,
    }


def _development_contact_sheet(model_input_root: Path, output_path: Path) -> None:
    rows = read_jsonl(model_input_root / "neutral_corpus_manifest.jsonl")
    indices = (0, len(rows) // 3, (2 * len(rows)) // 3, len(rows) - 1)
    items = []
    for index in indices:
        scene = rows[index]
        scene_id = str(scene["opaque_scene_id"])
        for view_id in ("V00", "V06", "V13"):
            items.append(
                {
                    "path": model_input_root / "media" / scene_id / "rgb" / f"{view_id}.png",
                    "label": f"{scene_id} {view_id}",
                }
            )
    save_image_grid(
        items,
        output_path,
        title="Remaining development cohort: deterministic RGB review",
        columns=4,
        cell_size=230,
    )


def run_primary_protocol_release(
    *,
    selection_root: Path,
    development_root: Path,
    administrative_root: Path,
    source_root: Path,
    output_root: Path,
    verify_archive_bytes: bool = True,
) -> dict[str, Any]:
    """Create the restartable 600-scene development RGB release and evidence."""

    output_root = output_root.expanduser().resolve()
    safe_root = output_root / "safe"
    private_root = output_root / "private"
    release_root = output_root / "development-remainder"
    protocol = audit_primary_protocol(
        selection_root=selection_root,
        development_root=development_root,
        administrative_root=administrative_root,
    )
    _package, build_audit = build_cohort_rgb_package(
        selection_root=selection_root,
        development_root=development_root,
        administrative_root=administrative_root,
        source_root=source_root,
        output_root=release_root,
        contract=DEVELOPMENT_REMAINDER_CONTRACT,
        verify_archive_bytes=verify_archive_bytes,
    )
    model_input_root = release_root / "model-input"
    verified = verify_model_input_package(
        model_input_root,
        contract=DEVELOPMENT_REMAINDER_CONTRACT,
    )
    model_scene_ids = {
        str(row["opaque_scene_id"])
        for row in read_jsonl(model_input_root / "neutral_corpus_manifest.jsonl")
    }
    role_rows = read_jsonl(selection_root / "experiment_roles.jsonl")
    held_out_ids = {
        str(row["opaque_scene_id"])
        for row in role_rows
        if str(row["role"]) in {"calibration", "internal_lock"}
    }
    if model_scene_ids & held_out_ids:
        raise ValueError("The development media release contains a held-out scene")

    checkpoint = {
        "schema_version": 1,
        "stage": STAGE,
        "cohort_id": DEVELOPMENT_REMAINDER_CONTRACT.cohort_id,
        "membership_sha256": DEVELOPMENT_REMAINDER_CONTRACT.membership_sha256,
        "manifest_sha256": verified.manifest_sha256,
        "content_checksums_sha256": verified.content_checksums_sha256,
        "image_count": verified.image_count,
        "status": "completed",
    }
    checkpoint_path = output_root / "release_checkpoint.json"
    if checkpoint_path.is_file():
        previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if previous != checkpoint:
            raise ValueError("Existing release checkpoint identifies different content")
    else:
        _write_json(checkpoint_path, checkpoint)

    cohort_rows = list(protocol["cohorts"])
    overlap_rows = list(protocol["overlaps"])
    development_stage_rows = list(protocol["development_stages"])
    write_csv_rows(
        safe_root / "cohort_membership.csv",
        (
            "cohort",
            "role",
            "scenes",
            "views",
            "asset_groups",
            "membership_sha256",
            "membership_verified",
            "rgb_release",
        ),
        cohort_rows,
    )
    write_csv_rows(
        safe_root / "development_stage_membership.csv",
        (
            "cohort",
            "stage",
            "scenes",
            "views",
            "asset_groups",
            "membership_sha256",
            "membership_verified",
        ),
        development_stage_rows,
    )
    write_csv_rows(
        safe_root / "role_overlap_audit.csv",
        ("left_role", "right_role", "scene_overlap", "asset_group_overlap", "passed"),
        overlap_rows,
    )
    release_rows = [
        {
            "role": "development_remaining",
            "scene_count": DEVELOPMENT_REMAINDER_CONTRACT.scene_count,
            "image_count": DEVELOPMENT_REMAINDER_CONTRACT.image_count,
            "media_state": "released",
            "label_state": "not_in_model_package",
            "next_notebook": "08E",
        },
        {
            "role": "calibration",
            "scene_count": CALIBRATION_CONTRACT.scene_count,
            "image_count": CALIBRATION_CONTRACT.image_count,
            "media_state": "sealed",
            "label_state": "sealed",
            "next_notebook": "09A_after_method_freeze",
        },
        {
            "role": "internal_lock",
            "scene_count": INTERNAL_LOCK_CONTRACT.scene_count,
            "image_count": INTERNAL_LOCK_CONTRACT.image_count,
            "media_state": "sealed",
            "label_state": "sealed",
            "next_notebook": "09C_after_policy_freeze",
        },
    ]
    write_csv_rows(
        safe_root / "resource_release_status.csv",
        ("role", "scene_count", "image_count", "media_state", "label_state", "next_notebook"),
        release_rows,
    )
    save_bar_chart(
        ["development", "calibration", "internal lock"],
        [1_000, 600, 400],
        safe_root / "primary_role_counts.png",
        title="Fixed 2,000-scene experiment roles",
        y_label="Scenes",
        integer_ticks=True,
        show_values=True,
    )
    _protocol_diagram(safe_root / "protocol_release_boundary.png")
    _development_contact_sheet(
        model_input_root,
        private_root / "figures" / "development_rgb_contact_sheet.png",
    )

    run_record = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "release_id": DEVELOPMENT_REMAINDER_CONTRACT.release_id,
        "primary_scene_count": 2_000,
        "development_scene_count": 1_000,
        "calibration_scene_count": 600,
        "internal_lock_scene_count": 400,
        "released_scene_count": verified.scene_count,
        "released_image_count": verified.image_count,
        "released_asset_group_count": verified.asset_group_count,
        "released_membership_sha256": verified.membership_sha256,
        "released_manifest_sha256": verified.manifest_sha256,
        "released_content_checksums_sha256": verified.content_checksums_sha256,
        "files_written_this_run": int(build_audit["files_written_this_run"]),
        "files_reused_this_run": int(build_audit["files_reused_this_run"]),
        "zero_scene_overlap": True,
        "zero_asset_group_overlap": True,
        "model_package_contains_held_out_scenes": False,
        "calibration_media_released": False,
        "internal_lock_media_released": False,
        "calibration_labels_opened": False,
        "internal_lock_labels_opened": False,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "all_completion_gates_passed": True,
    }
    _write_json(safe_root / "run_record.json", run_record)
    _artifact_manifest(safe_root)
    _artifact_manifest(private_root)
    return run_record


__all__ = [
    "STAGE",
    "audit_primary_protocol",
    "run_primary_protocol_release",
]
