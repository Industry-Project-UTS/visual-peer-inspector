from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "scripts/finalize_ds2_clearml.py"
SPEC = importlib.util.spec_from_file_location("finalize_ds2_clearml", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def write_safe_root(root: Path, *, stage: str, extra: dict[str, bytes]) -> Path:
    root.mkdir(parents=True)
    (root / "run_record.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed", "stage": stage}),
        encoding="utf-8",
    )
    for relative, content in extra.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    (root / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifact_count": len(rows), "artifacts": rows}),
        encoding="utf-8",
    )
    return root


def test_development_tracking_accepts_only_the_run_record(tmp_path: Path) -> None:
    root = write_safe_root(
        tmp_path / "safe",
        stage="frozen_development_pipeline",
        extra={},
    )
    assert finalizer.validate_safe_root(
        root, expected_stage="frozen_development_pipeline"
    )["status"] == "completed"

    (root / "private_rgb.png").write_bytes(b"not a real image")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    (root / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifact_count": len(rows), "artifacts": rows}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage allowlist"):
        finalizer.validate_safe_root(
            root, expected_stage="frozen_development_pipeline"
        )


@pytest.mark.parametrize(
    "stage",
    (
        "development_identity_evaluator_release",
        "calibration_resource_release",
        "internal_lock_resource_release",
    ),
)
def test_primary_resource_tracking_is_aggregate_only(
    tmp_path: Path, stage: str
) -> None:
    root = write_safe_root(tmp_path / stage, stage=stage, extra={})
    assert finalizer.validate_safe_root(root, expected_stage=stage)["status"] == "completed"

    (root / "scene_detail.csv").write_text("metric,value\ncount,1\n", encoding="utf-8")
    rows = []
    for path in sorted(
        item for item in root.iterdir() if item.name != "artifact_manifest.json"
    ):
        rows.append(
            {
                "path": path.name,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    (root / "artifact_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "artifact_count": len(rows), "artifacts": rows}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage allowlist"):
        finalizer.validate_safe_root(root, expected_stage=stage)


def test_unregistered_tracking_stage_fails_closed(tmp_path: Path) -> None:
    root = write_safe_root(tmp_path / "safe", stage="unregistered", extra={})
    with pytest.raises(ValueError, match="no code-owned"):
        finalizer.validate_safe_root(root, expected_stage="unregistered")


def test_ds2_backfill_attestation_is_an_allowed_aggregate(tmp_path):
    root = write_safe_root(tmp_path / "safe", stage="frozen_development_pipeline",
                           extra={"backfill_attestation.json": b'{"status": "verified"}'})
    assert finalizer.validate_safe_root(root, expected_stage="frozen_development_pipeline")["status"] == "completed"
