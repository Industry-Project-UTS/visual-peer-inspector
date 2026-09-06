from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from visual_peer_inspector.tracking_safe_artifacts import (
    validate_tracking_safe_root,
)


@pytest.mark.parametrize(
    "script_name",
    (
        "run_cross_view_identity_freeze.py",
        "run_calibration_review_policy.py",
        "run_one_shot_evaluation.py",
    ),
)
def test_custom_clearml_finalizers_disable_automatic_capture(
    script_name: str,
) -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / script_name).read_text(
        encoding="utf-8"
    )
    for setting in (
        "auto_connect_frameworks=False",
        "auto_connect_arg_parser=False",
        "auto_connect_streams=False",
        "auto_resource_monitoring=False",
        "output_uri=False",
    ):
        assert setting in source


def make_safe_root(tmp_path: Path) -> Path:
    root = tmp_path / "safe"
    root.mkdir()
    (root / "summary.csv").write_text("metric,value\naccuracy,0.75\n", encoding="utf-8")
    (root / "run_record.json").write_text(
        json.dumps({"stage": "test_stage", "status": "completed", "scene_count": 20}),
        encoding="utf-8",
    )
    rows = [
        {
            "path": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
    ]
    (root / "artifact_manifest.json").write_text(
        json.dumps({"artifact_count": len(rows), "artifacts": rows}),
        encoding="utf-8",
    )
    return root


def test_complete_allowlisted_tracking_root_passes(tmp_path: Path) -> None:
    record, manifest = validate_tracking_safe_root(
        make_safe_root(tmp_path),
        expected_stage="test_stage",
        allowed_relative_paths={"summary.csv"},
    )
    assert record["scene_count"] == 20
    assert manifest["artifact_count"] == 2


def test_tracking_root_rejects_unlisted_or_modified_artifacts(tmp_path: Path) -> None:
    root = make_safe_root(tmp_path)
    (root / "unexpected.csv").write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist"):
        validate_tracking_safe_root(
            root,
            expected_stage="test_stage",
            allowed_relative_paths={"summary.csv"},
        )

    (root / "unexpected.csv").unlink()
    (root / "summary.csv").write_text("metric,value\naccuracy,0.10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        validate_tracking_safe_root(
            root,
            expected_stage="test_stage",
            allowed_relative_paths={"summary.csv"},
        )


def test_tracking_root_rejects_private_detail_and_wrong_stage(tmp_path: Path) -> None:
    root = make_safe_root(tmp_path)
    (root / "summary.csv").write_text(
        "opaque_scene_id,value\nS000001,0.75\n", encoding="utf-8"
    )
    rows = [
        {
            "path": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
        if path.name != "artifact_manifest.json"
    ]
    (root / "artifact_manifest.json").write_text(
        json.dumps({"artifact_count": len(rows), "artifacts": rows}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Protected detail"):
        validate_tracking_safe_root(
            root,
            expected_stage="test_stage",
            allowed_relative_paths={"summary.csv"},
        )

    (root / "summary.csv").write_text("metric,value\naccuracy,0.75\n", encoding="utf-8")
    rows[1]["sha256"] = sha256((root / "summary.csv").read_bytes()).hexdigest()
    rows[1]["size_bytes"] = (root / "summary.csv").stat().st_size
    (root / "artifact_manifest.json").write_text(
        json.dumps({"artifact_count": len(rows), "artifacts": rows}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different stage"):
        validate_tracking_safe_root(
            root,
            expected_stage="another_stage",
            allowed_relative_paths={"summary.csv"},
        )
