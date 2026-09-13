from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pytest
from PIL import Image

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.sam2 import encode_binary_mask
from visual_peer_inspector.notebook_provenance import ENVIRONMENT_KEY
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    prune_verified_state,
    restore_recovery_bundle,
    verify_recovery_bundle,
    write_recovery_tracking_root,
    write_safe_artifact_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "recovery_clearml_finalizer",
    ROOT / "scripts/finalize_ds2_clearml.py",
)
assert FINALIZER_SPEC is not None and FINALIZER_SPEC.loader is not None
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
FINALIZER_SPEC.loader.exec_module(FINALIZER)
FROZEN_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "recovery_frozen_runner",
    ROOT / "scripts/run_frozen_development_pipeline.py",
)
assert FROZEN_RUNNER_SPEC is not None and FROZEN_RUNNER_SPEC.loader is not None
FROZEN_RUNNER = importlib.util.module_from_spec(FROZEN_RUNNER_SPEC)
FROZEN_RUNNER_SPEC.loader.exec_module(FROZEN_RUNNER)

BINDING = {
    "stage": "test-stage",
    "cohort_id": "test-cohort",
    "pipeline_identity_sha256": "1" * 64,
    "input_membership_sha256": "2" * 64,
    "crop_background": (238, 241, 245),
}
ALLOWED = ("pipeline_identity.json", "locateanything", "sam21")


def test_reporting_failure_preserves_shards_and_does_not_claim_completion(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        selected_prompt=tmp_path / "prompt.json",
        output_root=tmp_path / "run",
        recovery_output_directory=tmp_path / "recovery",
    )
    cohort = FROZEN_RUNNER.DEVELOPMENT_RECOVERY_TOP_LEVELS[0]
    stage = args.output_root / cohort / "peer_scoring"
    stage.mkdir(parents=True)
    (stage / "shard-00000.receipt.json").write_text('{"status":"complete"}')
    (stage / "shard-00000.jsonl").write_text('{"status":"ok"}\n')
    safe = args.output_root / cohort / "safe"
    safe.mkdir()
    (safe / "run_record.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(
        FROZEN_RUNNER, "build_identity", lambda _path: (None, None, None)
    )
    monkeypatch.setattr(
        FROZEN_RUNNER, "_development_recovery_binding", lambda *_args: BINDING
    )

    def failed_render(_args):
        raise ValueError("injected report failure")

    monkeypatch.setattr(FROZEN_RUNNER, "_run_development", failed_render)
    with pytest.raises(ValueError, match="injected report failure"):
        FROZEN_RUNNER.run_development(args)
    record = verify_recovery_bundle(
        bundle_path=args.recovery_output_directory / "notebook08e-recovery.zip",
        manifest_path=args.recovery_output_directory
        / "notebook08e-recovery-manifest.json",
        expected_binding=BINDING,
        allowed_top_levels=FROZEN_RUNNER.DEVELOPMENT_RECOVERY_TOP_LEVELS,
    )
    assert record["artifact_count"] == 3
    assert json.loads((safe / "run_record.json").read_text())["status"] == "interrupted"


def test_recovery_export_failure_keeps_the_original_error(tmp_path, monkeypatch):
    args = SimpleNamespace(
        selected_prompt=tmp_path / "prompt.json",
        output_root=tmp_path / "run",
        recovery_output_directory=tmp_path / "recovery",
    )
    stage = args.output_root / "cohort/peer_scoring"
    stage.mkdir(parents=True)
    (stage / "shard-00000.receipt.json").write_text("{}")
    monkeypatch.setattr(
        FROZEN_RUNNER, "build_identity", lambda _path: (None, None, None)
    )
    monkeypatch.setattr(
        FROZEN_RUNNER, "_development_recovery_binding", lambda *_args: BINDING
    )

    def fail_run(_args):
        raise ValueError("original error")

    def fail_export(**_kwargs):
        raise OSError("storage error")

    monkeypatch.setattr(FROZEN_RUNNER, "_run_development", fail_run)
    monkeypatch.setattr(FROZEN_RUNNER, "create_recovery_bundle", fail_export)
    with pytest.raises(ValueError, match="original error"):
        FROZEN_RUNNER.run_development(args)


def _state(root: Path) -> None:
    (root / "locateanything").mkdir(parents=True)
    (root / "sam21").mkdir(parents=True)
    (root / "pipeline_identity.json").write_text(
        '{"identity":"one"}\n', encoding="utf-8"
    )
    (root / "locateanything/shard-00000.jsonl").write_text(
        '{"row":1}\n', encoding="utf-8"
    )
    (root / "sam21/shard-00000.npy").write_bytes(b"feature-bytes")
    (root / "ignored.txt").write_text("not recoverable\n", encoding="utf-8")


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    state = tmp_path / "state"
    state.mkdir()
    _state(state)
    bundle = tmp_path / "out/recovery.zip"
    manifest = tmp_path / "out/recovery-manifest.json"
    record = create_recovery_bundle(
        source_root=state,
        bundle_path=bundle,
        manifest_path=manifest,
        binding=BINDING,
        allowed_top_levels=ALLOWED,
    )
    return state, bundle, manifest, record


def test_recovery_bundle_is_deterministic_and_restores_exactly(tmp_path: Path) -> None:
    state, bundle, manifest, record = _bundle(tmp_path)
    first_hash = file_sha256(bundle)
    second_bundle = tmp_path / "out/second.zip"
    second_manifest = tmp_path / "out/second-manifest.json"
    create_recovery_bundle(
        source_root=state,
        bundle_path=second_bundle,
        manifest_path=second_manifest,
        binding=BINDING,
        allowed_top_levels=ALLOWED,
    )
    assert file_sha256(second_bundle) == first_hash
    assert record["artifact_count"] == 3

    restored = tmp_path / "restored"
    restore_recovery_bundle(
        bundle_path=bundle,
        manifest_path=manifest,
        destination_root=restored,
        expected_binding=BINDING,
        allowed_top_levels=ALLOWED,
    )
    assert (restored / "locateanything/shard-00000.jsonl").read_bytes() == (
        state / "locateanything/shard-00000.jsonl"
    ).read_bytes()
    assert not (restored / "ignored.txt").exists()


def test_recovery_identity_includes_notebook_code_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        "snapshot_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "notebook_sha256": "c" * 64,
    }
    monkeypatch.setenv(ENVIRONMENT_KEY, json.dumps(provenance))

    _state_root, _bundle_path, manifest, _record = _bundle(tmp_path)

    sidecar = json.loads(manifest.read_text(encoding="utf-8"))
    assert {name: sidecar["binding"][name] for name in provenance} == provenance
    monkeypatch.setenv(
        ENVIRONMENT_KEY,
        json.dumps({**provenance, "snapshot_sha256": "d" * 64}),
    )
    with pytest.raises(ValueError, match="identity differs"):
        verify_recovery_bundle(
            bundle_path=_bundle_path,
            manifest_path=manifest,
            expected_binding=BINDING,
            allowed_top_levels=ALLOWED,
        )


def test_recovery_rejects_wrong_identity_and_changed_existing_file(
    tmp_path: Path,
) -> None:
    _state_root, bundle, manifest, _record = _bundle(tmp_path)
    with pytest.raises(ValueError, match="identity differs"):
        verify_recovery_bundle(
            bundle_path=bundle,
            manifest_path=manifest,
            expected_binding={**BINDING, "cohort_id": "other"},
            allowed_top_levels=ALLOWED,
        )
    restored = tmp_path / "restored"
    (restored / "sam21").mkdir(parents=True)
    (restored / "sam21/shard-00000.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing output differs"):
        restore_recovery_bundle(
            bundle_path=bundle,
            manifest_path=manifest,
            destination_root=restored,
            expected_binding=BINDING,
            allowed_top_levels=ALLOWED,
        )
    assert not (restored / "locateanything/shard-00000.jsonl").exists()


def test_recovery_rejects_corrupt_archive_before_restore(tmp_path: Path) -> None:
    _state_root, bundle, manifest, _record = _bundle(tmp_path)
    rewritten = bundle.with_name("rewritten-corrupt.zip")
    with ZipFile(bundle) as source, ZipFile(
        rewritten, "w", compression=ZIP_DEFLATED
    ) as destination:
        for entry in source.infolist():
            content = source.read(entry.filename)
            if entry.filename == "state/sam21/shard-00000.npy":
                content = b"changed-feature-bytes"
            destination.writestr(entry, content)
    rewritten.replace(bundle)
    sidecar = json.loads(manifest.read_text(encoding="utf-8"))
    sidecar["bundle_sha256"] = file_sha256(bundle)
    sidecar["bundle_size_bytes"] = bundle.stat().st_size
    manifest.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt|CRC|differs"):
        restore_recovery_bundle(
            bundle_path=bundle,
            manifest_path=manifest,
            destination_root=tmp_path / "restored",
            expected_binding=BINDING,
            allowed_top_levels=ALLOWED,
        )
    assert not (tmp_path / "restored").exists()


def test_recovery_rejects_unsafe_or_symbolic_members(tmp_path: Path) -> None:
    _state_root, _bundle_path, manifest, record = _bundle(tmp_path)
    unsafe_bundle = tmp_path / "out/unsafe.zip"
    internal = {
        key: value
        for key, value in record.items()
        if key not in {"bundle_sha256", "bundle_size_bytes"}
    }
    with ZipFile(unsafe_bundle, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("recovery_manifest.json", json.dumps(internal))
        info = ZipInfo("state/../outside.txt")
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, b"bad")
    changed = dict(record)
    changed["bundle_sha256"] = file_sha256(unsafe_bundle)
    changed["bundle_size_bytes"] = unsafe_bundle.stat().st_size
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist|unsafe"):
        verify_recovery_bundle(
            bundle_path=unsafe_bundle,
            manifest_path=manifest,
            expected_binding=BINDING,
            allowed_top_levels=ALLOWED,
        )

    symlink_bundle = tmp_path / "out/symlink.zip"
    artifact = dict(record["artifacts"][0])
    link_internal = {
        **internal,
        "artifact_count": 1,
        "artifacts": [artifact],
    }
    with ZipFile(symlink_bundle, "w") as archive:
        archive.writestr("recovery_manifest.json", json.dumps(link_internal))
        info = ZipInfo(f"state/{artifact['path']}")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
    link_sidecar = {
        **link_internal,
        "bundle_sha256": file_sha256(symlink_bundle),
        "bundle_size_bytes": symlink_bundle.stat().st_size,
    }
    manifest.write_text(json.dumps(link_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe member"):
        verify_recovery_bundle(
            bundle_path=symlink_bundle,
            manifest_path=manifest,
            expected_binding=BINDING,
            allowed_top_levels=ALLOWED,
        )


def test_pruning_happens_only_after_bundle_and_source_verify(tmp_path: Path) -> None:
    state, bundle, manifest, _record = _bundle(tmp_path)
    changed = state / "sam21/shard-00000.npy"
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before pruning"):
        prune_verified_state(
            source_root=state,
            bundle_path=bundle,
            manifest_path=manifest,
            expected_binding=BINDING,
            allowed_top_levels=ALLOWED,
        )
    assert (state / "locateanything/shard-00000.jsonl").is_file()
    changed.write_bytes(b"feature-bytes")
    removed = prune_verified_state(
        source_root=state,
        bundle_path=bundle,
        manifest_path=manifest,
        expected_binding=BINDING,
        allowed_top_levels=ALLOWED,
    )
    assert removed == 3
    assert (state / "ignored.txt").is_file()


def test_recovery_and_final_tracking_roots_pass_the_shared_clearml_contract(
    tmp_path: Path,
) -> None:
    recovery_root = tmp_path / "recovery-safe"
    write_recovery_tracking_root(
        safe_root=recovery_root,
        stage="frozen_development_pipeline",
        pipeline_status="checkpointed",
        metrics={
            "covered_image_count": 400,
            "target_image_count": 2_000,
            "recovery_artifact_count": 12,
        },
    )
    recovered = FINALIZER.validate_safe_root(
        recovery_root,
        expected_stage="frozen_development_pipeline",
    )
    assert recovered["record_kind"] == "recovery"
    assert recovered["pipeline_status"] == "checkpointed"

    protected_root = tmp_path / "protected-recovery-safe"
    write_recovery_tracking_root(
        safe_root=protected_root,
        stage="protected_calibration_inference",
        pipeline_status="checkpointed",
        metrics={
            "covered_image_count": 1_000,
            "target_image_count": 12_000,
            "recovery_artifact_count": 44,
            "exact_resume_available": True,
        },
    )
    protected = FINALIZER.validate_safe_root(
        protected_root,
        expected_stage="protected_calibration_inference",
    )
    assert protected["record_kind"] == "recovery"

    final_root = tmp_path / "final-safe"
    final_root.mkdir()
    (final_root / "run_record.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "frozen_development_pipeline",
                "image_count": 2_000,
            }
        ),
        encoding="utf-8",
    )
    write_safe_artifact_manifest(final_root)
    final = FINALIZER.validate_safe_root(
        final_root,
        expected_stage="frozen_development_pipeline",
    )
    assert final["image_count"] == 2_000


def test_frozen_development_visuals_stay_outside_the_clearml_contract(
    tmp_path: Path,
) -> None:
    cohort_root = tmp_path / "cohort"
    safe_root = cohort_root / "safe"
    review_root = cohort_root / "review-only"
    safe_root.mkdir(parents=True)
    image_paths = []
    for index in range(2):
        path = tmp_path / f"rgb-{index}.png"
        Image.new("RGB", (256, 256), (40 + index, 80, 120)).save(path)
        image_paths.append(path)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:110, 80:140] = 1
    encoded = encode_binary_mask(mask)
    rows = [
        {
            "opaque_scene_id": "S000001",
            "view_id": f"V{index:02d}",
            "image_path": image_paths[index].as_posix(),
            "terminal_failure": None,
            "objects": [
                {
                    "object_id": "P01",
                    "box_xyxy": [80, 50, 140, 110],
                    "mask_rle": encoded,
                    "p1_score": 0.8,
                    "p1_rank": 1,
                    "p1_heatmap": np.ones((16, 16)).tolist(),
                }
            ],
        }
        for index in range(2)
    ]
    evidence = FROZEN_RUNNER._scene_audit_panels(rows, review_root / "figures")
    audit_record = {"items": evidence}
    (review_root / "audit_evidence.json").write_text(
        json.dumps(audit_record),
        encoding="utf-8",
    )
    (safe_root / "run_record.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "frozen_development_pipeline",
                "image_count": 2,
                "audit_panel_count": len(evidence),
            }
        ),
        encoding="utf-8",
    )
    assert "S000001" not in json.dumps(audit_record)
    write_safe_artifact_manifest(safe_root)
    validated = FINALIZER.validate_safe_root(
        safe_root,
        expected_stage="frozen_development_pipeline",
    )
    assert validated["audit_panel_count"] == 1
    assert not (safe_root / "figures").exists()
    assert (review_root / "audit_evidence.json").is_file()
