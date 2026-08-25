from __future__ import annotations

import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_ds1_mask_evaluation_clearml.py"
SPEC = importlib.util.spec_from_file_location("finalize_ds1_mask_evaluation_clearml", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def _manifest_digest(rows):
    return sha256(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
    ).hexdigest()


def _safe_root(tmp_path: Path, *, git_dirty: bool = False) -> Path:
    root = tmp_path / "safe"
    root.mkdir()
    for name, columns in finalizer.ALLOWED_TABLES.items():
        row = {column: 0 for column in columns}
        for column, value in {
            "metric": "mean_matched_iou",
            "dimension": "view_id",
            "value": "V00",
            "route": "reference_box",
            "direction": "baseline_minus_candidate",
            "failure_type": "none",
            "gate": "proposal recall",
            "status": "pass",
        }.items():
            if column in row:
                row[column] = value
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(row)
    runtime = {
        "proposal_total_seconds": 1.0,
        "proposal_median_seconds_per_image": 0.1,
        "proposal_p95_seconds_per_image": 0.2,
        "sam_total_seconds": 2.0,
        "sam_median_seconds_per_image": 0.2,
        "sam_p95_seconds_per_image": 0.3,
        "evaluator_seconds": 3.0,
    }
    (root / "runtime_summary.json").write_text(json.dumps(runtime), encoding="utf-8")
    cohort = {
        key: getattr(DS1_SCREEN_CONTRACT, key)
        for key in (
            "cohort_id", "release_id", "role", "pilot_stage", "membership_sha256",
            "scene_count", "views_per_scene", "image_count", "asset_group_count",
        )
    }
    route = {
        "route": "reference_box",
        "detection_aware_recall_at_0_75": 0.95,
        "mean_matched_iou": 0.94,
        "detection_aware_mean_iou": 0.93,
        "mean_matched_dice": 0.96,
        "detection_aware_mean_dice": 0.95,
        "mean_matched_boundary_f1": 0.91,
        "detection_aware_mean_boundary_f1": 0.90,
    }
    record = {
        "schema_version": 1,
        "stage": "ds1_proposal_and_mask_evaluation",
        "status": "completed",
        "tracking_status": "pending",
        "cohort": cohort,
        "coverage": {
            "reference_rows": 8200,
            "decoded_sam_masks": 16000,
            "proposal_images": DS1_SCREEN_CONTRACT.image_count,
            "sam_images": DS1_SCREEN_CONTRACT.image_count,
        },
        "settings": {},
        "proposal_summary": {"micro_precision_at_0_50": 0.94, "micro_recall_at_0_50": 0.96},
        "mask_route_summary": [route, {**route, "route": "automatic_box"}],
        "component_decision": {
            "status": "proceed_with_locateanything_and_sam21",
            "all_gates_passed": True,
            "final_acceptance": False,
            "reason": "Development screen only.",
        },
        "inputs": {
            "evaluator_record_sha256": "1" * 64,
            "proposal_results_sha256": "2" * 64,
            "sam_results_sha256": "3" * 64,
            "mask_route_record_sha256": "4" * 64,
        },
        "evaluator_provenance": {
            "notebook_name": "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb",
            "execution_environment": "kaggle",
            "git_commit": "a" * 40,
            "git_dirty": git_dirty,
            "precommit_candidate": False,
            "snapshot_sha256": "b" * 64,
            "notebook_sha256": sha256(
                (ROOT / "notebooks/detection-and-segmentation/toysad8k-ds1-mask-evaluation/06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb").read_bytes()
            ).hexdigest(),
            "code_dataset_version": "12",
            "kaggle_notebook_version_06_1": "2",
        },
        "runtime": runtime,
    }
    (root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
    for name in finalizer.ALLOWED_CHARTS:
        (root / name).write_bytes(b"safe aggregate chart")
    rows = []
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        rows.append({
            "path": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        })
    manifest = {
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_content_sha256": _manifest_digest(rows),
    }
    (root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_finalizer_accepts_only_complete_aggregate_evidence(tmp_path):
    root = _safe_root(tmp_path)
    record = finalizer.validate_safe_root(root)
    assert record["status"] == "completed"
    (root / "private.json").write_text("{}", encoding="utf-8")
    try:
        finalizer.validate_safe_root(root)
    except ValueError as error:
        assert "allowlist differs" in str(error)
    else:
        raise AssertionError("Unexpected ClearML file was accepted")


def test_final_task_rejects_dirty_provenance(tmp_path):
    root = _safe_root(tmp_path, git_dirty=True)
    try:
        finalizer.validate_safe_root(root, require_clean_provenance=True)
    except ValueError as error:
        assert "clean code snapshot" in str(error)
    else:
        raise AssertionError("Dirty final evidence was accepted")


def test_final_task_rejects_precommit_provenance(tmp_path):
    root = _safe_root(tmp_path)
    record = json.loads((root / "run_record.json").read_text(encoding="utf-8"))
    record["evaluator_provenance"]["precommit_candidate"] = True
    (root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
    rows = []
    for path in sorted(item for item in root.iterdir() if item.is_file() and item.name != "artifact_manifest.json"):
        rows.append({"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size})
    (root / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifact_count": len(rows), "artifacts": rows, "manifest_content_sha256": _manifest_digest(rows)}), encoding="utf-8")
    try:
        finalizer.validate_safe_root(root, require_clean_provenance=True)
    except ValueError as error:
        assert "clean code snapshot" in str(error)
    else:
        raise AssertionError("Precommit final evidence was accepted")


def test_disabled_finalizer_writes_verified_receipt_without_credentials(tmp_path, monkeypatch):
    root = _safe_root(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--safe-root", str(root),
            "--receipt", str(receipt),
            "--mode", "disabled",
        ],
    )
    finalizer.main()
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["status"] == "disabled"
    assert value["snapshot_sha256"] == "b" * 64
    assert value["safe_manifest_sha256"] == sha256((root / "artifact_manifest.json").read_bytes()).hexdigest()


def test_tracking_install_is_skipped_for_importable_pinned_version():
    with (
        mock.patch.object(
            finalizer.importlib.metadata,
            "version",
            return_value=finalizer.PINNED_CLEARML_VERSION,
        ),
        mock.patch.object(finalizer.importlib, "import_module", return_value=object()),
        mock.patch.object(finalizer.subprocess, "run") as install,
    ):
        installed = finalizer._install_tracking_requirement()

    assert installed is False
    install.assert_not_called()


def test_tracking_install_repairs_a_version_mismatch():
    with (
        mock.patch.object(
            finalizer.importlib.metadata,
            "version",
            side_effect=["2.0.0", finalizer.PINNED_CLEARML_VERSION],
        ),
        mock.patch.object(finalizer.importlib, "import_module", return_value=object()),
        mock.patch.object(finalizer.subprocess, "run") as install,
    ):
        installed = finalizer._install_tracking_requirement()

    assert installed is True
    install.assert_called_once_with(
        [
            finalizer.sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(finalizer.CODE_ROOT / "requirements/tracking-lock.txt"),
        ],
        timeout=120,
        check=True,
    )


def test_notebook_isolated_secret_loading_and_snapshot_verification():
    notebook_path = ROOT / "notebooks" / "detection-and-segmentation" / "toysad8k-ds1-mask-evaluation" / "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "verify_code_snapshot_record" in source
    assert "finalize_ds1_mask_evaluation_clearml.py" in source
    assert '"--secret-source"' in source
    assert "UserSecretsClient" not in source
    assert "CLEARML_API_ACCESS_KEY" not in source
    assert "pip\", \"install" not in source
    assert "sys.dont_write_bytecode = True" in source
    assert "kaggle_notebook_version_06_1" in source
    assert "notebook_sha256" in source
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    metadata = json.loads((notebook_path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_internet"] is True
    assert metadata["id"] == "esreekarreddy/toysad-8k-ds1-proposal-and-mask-evaluation"
    assert metadata["dataset_sources"] == [
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-ds1-mask-evaluator",
        "esreekarreddy/vpi-toysad-ds1-locateanything-output",
    ]
    assert metadata["kernel_sources"] == [
        "esreekarreddy/toysad-8k-ds1-sam-2-1-segmentation",
    ]
