from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from visual_peer_inspector.data.model_safe_package import file_sha256, membership_sha256
from visual_peer_inspector.tracking_safe_artifacts import write_tracking_safe_manifest
from visual_peer_inspector.workflows.full_pool_sensitivity import (
    IMAGE_COUNT,
    MODEL_MEMBERSHIP_SHA256,
    PRIMARY_ROLE_COUNTS,
    SCENE_COUNT,
    full_pool_contract,
    full_pool_opaque_scene_id,
    load_primary_role_map,
    verify_10b_completion,
)
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    restore_recovery_bundle,
)


def test_full_pool_contract_has_exact_counts_and_opaque_membership() -> None:
    source_membership = "a" * 64
    contract = full_pool_contract(source_membership)
    opaque_ids = [full_pool_opaque_scene_id(index) for index in range(1, SCENE_COUNT + 1)]
    assert SCENE_COUNT == 4_107
    assert IMAGE_COUNT == 82_140
    assert contract.scene_count == 4_107
    assert contract.image_count == 82_140
    assert contract.views_per_scene == 20
    assert membership_sha256(opaque_ids) == MODEL_MEMBERSHIP_SHA256
    assert opaque_ids[0] == "F000001" and opaque_ids[-1] == "F004107"


def test_primary_roles_are_preserved_and_not_reclassified_as_remainder(
    tmp_path: Path,
) -> None:
    root = tmp_path / "administrative"
    root.mkdir()
    rows = []
    index = 0
    for role, count in (("development", 1_000), ("calibration", 600), ("internal_lock", 400)):
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "source_scene_id": f"scene-{index:04d}",
                    "role": role,
                    "pilot_stage": None,
                }
            )
    path = root / "restricted_source_map.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mapping = load_primary_role_map(root)
    counts = Counter(item["primary_role"] for item in mapping.values())
    assert len(mapping) == 2_000
    assert counts == Counter({"development": 1_000, "calibration": 600, "internal_lock": 400})
    assert "eligible_remainder" not in counts
    assert PRIMARY_ROLE_COUNTS["eligible_remainder"] == 2_107
    assert sum(PRIMARY_ROLE_COUNTS.values()) == 4_107


def _completed_10b(root: Path) -> Path:
    safe = root / "safe"
    safe.mkdir(parents=True)
    run = {
        "status": "completed",
        "stage": "10b",
        "no_retuning": True,
        "method_changed": False,
        "policy_changed": False,
        "method_record_sha256": "1" * 64,
        "policy_record_sha256": "2" * 64,
    }
    (safe / "run_record.json").write_text(json.dumps(run) + "\n", encoding="utf-8")
    write_tracking_safe_manifest(safe)
    completion = {
        "schema_version": "visual-peer-one-shot-evaluation/v1",
        "status": "complete",
        "stage": "10b",
        "safe_manifest_sha256": file_sha256(safe / "artifact_manifest.json"),
    }
    (root / "evaluation_complete.json").write_text(
        json.dumps(completion) + "\n", encoding="utf-8"
    )
    return root


def test_post_freeze_stage_requires_completed_10b_and_forbids_retuning(
    tmp_path: Path,
) -> None:
    root = _completed_10b(tmp_path / "official")
    authorization = verify_10b_completion(root)
    assert authorization["scope"] == "descriptive_sensitivity_only"
    assert authorization["no_retuning"] is True
    assert authorization["primary_results_unchanged"] is True
    assert authorization["official_results_unchanged"] is True

    run_path = root / "safe/run_record.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["method_changed"] = True
    run_path.write_text(json.dumps(run) + "\n", encoding="utf-8")
    write_tracking_safe_manifest(root / "safe")
    completion = json.loads((root / "evaluation_complete.json").read_text(encoding="utf-8"))
    completion["safe_manifest_sha256"] = file_sha256(root / "safe/artifact_manifest.json")
    (root / "evaluation_complete.json").write_text(json.dumps(completion) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a completed frozen official evaluation"):
        verify_10b_completion(root)


def test_full_pool_recovery_is_deterministic_and_identity_bound(tmp_path: Path) -> None:
    state = tmp_path / "state"
    (state / "trusted-release-state").mkdir(parents=True)
    (state / "trusted-release-state/F000001.json").write_text("{}\n", encoding="utf-8")
    binding = {
        "stage": "post_freeze_full_pool_release",
        "cohort_id": "toysad8k-eligible-train-full-pool-4107-v1",
        "plan_sha256": "3" * 64,
    }
    bundle1 = tmp_path / "first.zip"
    manifest1 = tmp_path / "first.json"
    bundle2 = tmp_path / "second.zip"
    manifest2 = tmp_path / "second.json"
    for bundle, manifest in ((bundle1, manifest1), (bundle2, manifest2)):
        create_recovery_bundle(
            source_root=state,
            bundle_path=bundle,
            manifest_path=manifest,
            binding=binding,
            allowed_top_levels=("trusted-release-state",),
        )
    assert file_sha256(bundle1) == file_sha256(bundle2)
    restored = tmp_path / "restored"
    restore_recovery_bundle(
        bundle_path=bundle1,
        manifest_path=manifest1,
        destination_root=restored,
        expected_binding=binding,
        allowed_top_levels=("trusted-release-state",),
    )
    assert (restored / "trusted-release-state/F000001.json").read_text() == "{}\n"
    with pytest.raises(ValueError, match="identity differs"):
        restore_recovery_bundle(
            bundle_path=bundle1,
            manifest_path=manifest1,
            destination_root=tmp_path / "wrong",
            expected_binding={**binding, "plan_sha256": "4" * 64},
            allowed_top_levels=("trusted-release-state",),
        )
