from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import membership_sha256
from visual_peer_inspector.workflows.primary_protocol_release import (
    _contract_rows,
    _development_contact_sheet,
    _protocol_diagram,
    run_primary_protocol_release,
)


def _row(scene_id: str, asset_id: str, role: str) -> dict:
    return {
        "opaque_scene_id": scene_id,
        "opaque_asset_group_id": asset_id,
        "role": role,
        "pilot_stage": "screen" if role == "development" else None,
        "category": "toy",
        "object_count": 3,
        "anomaly_count": 1,
        "raw_anomaly_prefixes": ["bump"],
    }


def _contract(role: str, rows: list[dict]) -> CohortContract:
    return CohortContract(
        cohort_id=f"{role}-test",
        release_id="test-release",
        role=role,
        pilot_stage="screen" if role == "development" else None,
        membership_sha256=membership_sha256(row["opaque_scene_id"] for row in rows),
        scene_count=len(rows),
        views_per_scene=20,
        image_count=len(rows) * 20,
        asset_group_count=len({row["opaque_asset_group_id"] for row in rows}),
    )


def test_contract_rows_verify_membership_and_role_separation() -> None:
    development = [_row("S000001", "A01", "development")]
    calibration = [_row("S000002", "A02", "calibration")]
    internal_lock = [_row("S000003", "A03", "internal_lock")]
    trusted = development + calibration + internal_lock
    role_rows = [
        {"opaque_scene_id": row["opaque_scene_id"], "role": row["role"]}
        for row in trusted
    ]
    contracts = tuple(
        _contract(role, rows)
        for role, rows in (
            ("development", development),
            ("calibration", calibration),
            ("internal_lock", internal_lock),
        )
    )

    rows, scene_sets, asset_sets = _contract_rows(
        role_rows,
        trusted,
        contracts,
        expected_primary_scene_count=3,
    )

    assert [row["role"] for row in rows] == [
        "development",
        "calibration",
        "internal_lock",
    ]
    assert all(row["membership_verified"] for row in rows)
    assert scene_sets["development"].isdisjoint(scene_sets["calibration"])
    assert asset_sets["development"].isdisjoint(asset_sets["internal_lock"])
    assert rows[0]["rgb_release"] == "complete_across_development_stages"
    assert rows[1]["rgb_release"] == "sealed"


def test_contract_rows_reject_a_role_mismatch() -> None:
    trusted = [_row("S000001", "A01", "development")]
    contract = _contract("development", trusted)
    role_rows = [{"opaque_scene_id": "S000001", "role": "calibration"}]

    try:
        _contract_rows(
            role_rows,
            trusted,
            (contract,),
            expected_primary_scene_count=1,
        )
    except ValueError as error:
        assert "role differs" in str(error)
    else:
        raise AssertionError("A conflicting protocol role was accepted")


def test_protocol_figures_are_deterministic_development_evidence(tmp_path: Path) -> None:
    diagram = tmp_path / "protocol.png"
    _protocol_diagram(diagram)
    with Image.open(diagram) as image:
        assert image.size == (1500, 680)

    model_input = tmp_path / "model-input"
    manifest = []
    for index in range(4):
        scene_id = f"S{index:06d}"
        manifest.append({"opaque_scene_id": scene_id})
        for view_id in ("V00", "V06", "V13"):
            path = model_input / "media" / scene_id / "rgb" / f"{view_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (256, 256), (index * 30, 80, 120)).save(path)
    (model_input / "neutral_corpus_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest),
        encoding="utf-8",
    )

    contact_sheet = tmp_path / "contact.png"
    _development_contact_sheet(model_input, contact_sheet)
    with Image.open(contact_sheet) as image:
        assert image.width > 800
        assert image.height > 500


def test_run_record_does_not_read_private_identity_from_package_audit() -> None:
    source = inspect.getsource(run_primary_protocol_release)

    assert "verified.asset_group_count" not in source
    assert "DEVELOPMENT_REMAINDER_CONTRACT.asset_group_count" in source
