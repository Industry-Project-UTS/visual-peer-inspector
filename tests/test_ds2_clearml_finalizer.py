from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_ds2_clearml.py"
SPEC = importlib.util.spec_from_file_location("finalize_ds2_clearml", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def _safe_root(tmp_path: Path) -> Path:
    root = tmp_path / "safe"
    root.mkdir()
    (root / "summary.csv").write_text("metric,value\nmean_ap,0.75\n", encoding="utf-8")
    (root / "chart.png").write_bytes(b"aggregate chart")
    (root / "run_record.json").write_text(json.dumps({
        "schema_version": 1,
        "stage": "ds2_confirmation_and_multiview",
        "status": "completed",
        "scene_count": 300,
    }), encoding="utf-8")
    rows = []
    for path in sorted(root.iterdir()):
        rows.append({
            "path": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        })
    (root / "artifact_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
    }), encoding="utf-8")
    return root


def test_ds2_clearml_accepts_aggregate_evidence(tmp_path: Path) -> None:
    record = finalizer.validate_safe_root(
        _safe_root(tmp_path),
        expected_stage="ds2_confirmation_and_multiview",
    )
    assert record["scene_count"] == 300


def test_ds2_clearml_rejects_scene_identifiers(tmp_path: Path) -> None:
    root = _safe_root(tmp_path)
    (root / "summary.csv").write_text("opaque_scene_id,value\nS000001,0.5\n", encoding="utf-8")
    rows = []
    for path in sorted(item for item in root.iterdir() if item.name != "artifact_manifest.json"):
        rows.append({
            "path": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        })
    (root / "artifact_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Protected detail"):
        finalizer.validate_safe_root(
            root,
            expected_stage="ds2_confirmation_and_multiview",
        )


def test_ds2_clearml_extracts_only_bounded_numeric_scalars() -> None:
    values = finalizer._numeric_scalars({
        "scene_count": 300,
        "all_completion_gates_passed": True,
        "nested": {"mean_ap": 0.71, "status": "completed"},
    })
    assert values == [
        ("run.all_completion_gates_passed", 1.0),
        ("run.nested.mean_ap", 0.71),
        ("run.scene_count", 300.0),
    ]
