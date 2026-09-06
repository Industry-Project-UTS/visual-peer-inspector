from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "protected-evaluation"


def source_for(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebooks_are_clean_prediction_only_and_compile() -> None:
    for code, name in (
        ("09a", "09a-toysad8k-protected-calibration-inference.ipynb"),
        ("09c", "09c-toysad8k-protected-internal-lock-inference.ipynb"),
    ):
        path = NOTEBOOK_ROOT / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = source_for(path)
        assert name in source
        assert "—" not in source
        assert "run_protected_cohort_inference.py" in source
        assert '"--run-mode", RUN_PHASE' in source
        assert "VPI_RUN_PHASE" in source
        assert "VPI_MAXIMUM_NEW_SHARDS" in source
        assert "VPI_RESUME_ROOT" in source
        assert '"--recovery-output-directory"' in source
        assert "exactly one protected recovery pair" in source
        assert "not published" in source
        assert "recovery_summary.json" not in source
        assert 'OUTPUT_ROOT / "tracking/clearml_receipt.json"' in source
        assert "write_safe_artifact_manifest(tracking_root)" not in source
        assert source.index('"--run-mode", RUN_PHASE') < source.index(
            "tracking_receipt = finish_clearml"
        )
        assert "method_freeze.json" in source
        assert "application_output_contract.json" in source
        assert "prediction_lock.json" in source
        assert "multiview_prediction_manifest.jsonl" not in source or "prediction_bundle" in source
        assert "2, 3, and 5 view contracts" in source
        assert "SAM masks" in source
        assert "masked-on-black" in source
        assert "ZipInfo" in source
        assert "date_time=(1980, 1, 1, 0, 0, 0)" in source
        assert 'archive.open(member, "w")' in source
        assert "shutil.copyfileobj" in source
        assert "file_sha256(output_path)" in source
        assert "output_path.read_bytes()" not in source
        assert '"preflight"' in source
        assert "timeout=1800" in source
        assert 'inference_environment["HF_HUB_OFFLINE"] = "1"' in source
        assert source.index('"preflight"') < source.index(
            'subprocess.run(phase_command'
        )
        assert "accuracy_metrics_reported" not in source
        assert "find_evaluator" not in source
        if code == "09a":
            assert "review_policy_freeze.json" not in source
            assert "No calibration-derived review policy is applied" in source
        else:
            assert "review_policy_freeze.json" in source
            assert "Notebook 09B review policy" in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert not cell.get("outputs")
                compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")

    runner = (ROOT / "scripts/run_protected_cohort_inference.py").read_text(
        encoding="utf-8"
    )
    assert "write_recovery_tracking_root" in runner


def test_metadata_attaches_only_role_specific_rgb_and_prerequisites() -> None:
    metadata09a = json.loads((NOTEBOOK_ROOT / "kernel-metadata-09a.json").read_text())
    metadata09c = json.loads((NOTEBOOK_ROOT / "kernel-metadata-09c.json").read_text())
    for metadata in (metadata09a, metadata09c):
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is True
        joined = " ".join(metadata["dataset_sources"] + metadata["kernel_sources"]).lower()
        assert "evaluator" not in joined
        assert "label" not in joined
        assert "crosswalk" not in joined
    assert any("calibration600-model-input" in item for item in metadata09a["dataset_sources"])
    assert not any("internal-lock" in item for item in metadata09a["dataset_sources"])
    assert any("internal-lock400-model-input" in item for item in metadata09c["dataset_sources"])
    assert metadata09a["kernel_sources"] == []
    assert metadata09c["kernel_sources"] == []
    assert "esreekarreddy/vpi-toysad-development-method-freeze" in metadata09a[
        "dataset_sources"
    ]
    assert "esreekarreddy/vpi-toysad-development-method-freeze" in metadata09c[
        "dataset_sources"
    ]
    assert "esreekarreddy/vpi-toysad-calibration-review-policy" in metadata09c[
        "dataset_sources"
    ]
