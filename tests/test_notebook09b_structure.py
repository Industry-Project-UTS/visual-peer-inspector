from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "protected-evaluation"
NOTEBOOK_PATH = NOTEBOOK_ROOT / "09b-toysad8k-calibration-review-policy.ipynb"
METADATA_PATH = NOTEBOOK_ROOT / "kernel-metadata-09b.json"


def notebook_source() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def test_notebook09b_has_the_complete_calibration_boundary_and_outputs() -> None:
    notebook, source = notebook_source()
    for phrase in (
        NOTEBOOK_PATH.name,
        "verifies the complete Notebook 09A prediction lock before opening",
        "LocateAnything proposal policy",
        "SAM 2.1",
        "DINOv2-small",
        "P1 scoring",
        "cross-view identity method",
        "base asset groups remain intact",
        "result_ready",
        "review_required",
        "insufficient_evidence",
        "Raw P1 scores and margins are not probabilities",
        "risk-coverage frontier",
        "false-ready rate",
        "reliability",
        "subgroup",
        "confidence intervals",
        "Wrong `result_ready` examples",
        "capped at 16 panels",
        "all acceptable anomaly identities",
        "review_policy_freeze.json",
        "ClearML receives aggregate tables",
    ):
        assert phrase.lower() in source.lower()
    assert "No arbitrary universal accuracy threshold is used" in source
    assert "0.95" not in source
    assert "minimum accuracy 0.95" not in source.lower()
    assert "minimum recall 0.95" not in source.lower()
    assert "—" not in source
    assert "August" not in source
    assert source.index("verify_calibration_boundary(") < source.index("EVALUATOR_CORRESPONDENCE =")
    assert source.index("subprocess.run(command") < source.index("--clearml-finalize-only")
    assert len(notebook["cells"]) >= 22


def test_notebook09b_cells_are_clean_and_compile() -> None:
    notebook, _source = notebook_source()
    assert notebook["cells"][0]["cell_type"] == "markdown"
    headings = [
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines()
        if line.startswith("## ")
    ]
    assert len(headings) >= 10
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")
        compile("".join(cell.get("source", [])), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_notebook09b_metadata_consumes_09a_and_private_evaluator() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["id"] == "esreekarreddy/toysad-8k-calibration-review-policy"
    assert metadata["code_file"] == NOTEBOOK_PATH.name
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["kernel_sources"] == [
        "esreekarreddy/toysad-8k-protected-calibration-inference"
    ]
    assert "esreekarreddy/vpi-toysad-development-method-freeze" in metadata[
        "dataset_sources"
    ]
    assert "esreekarreddy/vpi-toysad-calibration600-evaluator" in metadata["dataset_sources"]


def test_runner_bounds_clearml_and_keeps_credentials_out_of_arguments() -> None:
    source = (ROOT / "scripts/run_calibration_review_policy.py").read_text(encoding="utf-8")
    notebook_source_text = notebook_source()[1]
    assert "--clearml-finalize-only" in source
    assert "evaluation_failure_receipt.json" in source
    assert "timeout=180" in notebook_source_text
    assert "CLEARML_API_ACCESS_KEY" not in source
    assert "CLEARML_API_SECRET_KEY" not in source
    assert "raw_p1_is_probability" in source
    assert "validate_tracking_safe_root" in source
    assert "allowed_relative_paths" in source
    assert "publish_frozen_resource" in source
    assert "ZipInfo" in notebook_source_text
    assert "date_time=(1980, 1, 1, 0, 0, 0)" in notebook_source_text
    assert 'archive.open(member, "w")' in notebook_source_text
    assert "shutil.copyfileobj" in notebook_source_text
    assert "file_sha256(output_path)" in notebook_source_text
    assert "output_path.read_bytes()" not in notebook_source_text
    assert "publication_receipt.json" in source
    assert "wrong_result_ready_scene_contracts.jsonl" in source
    assert "wrong_result_ready_scene_contracts.csv" in source
    assert "total_wrong_result_ready_count" in source
    assert "rendered_wrong_result_ready_count" in source
    assert "evaluator_example_count_limit" in source
    assert source.index('_write_json(safe_root / "run_record.json"') < source.index(
        "publish_frozen_resource("
    )
    assert source.index("publish_frozen_resource(") < source.index(
        'safe_root / "publication_receipt.json"'
    )


def test_workflow_runner_and_builder_compile() -> None:
    for path in (
        ROOT / "src/visual_peer_inspector/workflows/calibration_review_policy.py",
        ROOT / "scripts/run_calibration_review_policy.py",
        ROOT / "scripts/build_notebook09b.py",
    ):
        compile(path.read_text(encoding="utf-8"), path.name, "exec")
