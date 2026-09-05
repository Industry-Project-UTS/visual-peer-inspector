from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "confirmation-and-multiview"


def _source(path: Path) -> tuple[dict, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def test_notebook08_sources_are_clean_and_executable() -> None:
    notebooks = sorted(NOTEBOOK_ROOT.glob("08*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        notebook, source = _source(path)
        assert path.name in source
        assert "—" not in source
        assert all(
            cell.get("execution_count") is None and not cell.get("outputs")
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")


def test_notebook08_keeps_the_evaluator_out_of_model_inference() -> None:
    _release, release_source = _source(NOTEBOOK_ROOT / "08a-toysad8k-ds2-confirmation-release.ipynb")
    _inference, inference_source = _source(NOTEBOOK_ROOT / "08b-toysad8k-ds2-frozen-pipeline.ipynb")
    _evaluation, evaluation_source = _source(NOTEBOOK_ROOT / "08c-toysad8k-ds2-confirmation-and-multiview.ipynb")

    assert "confirm_300" in release_source
    assert "evaluator-attributes" in release_source
    assert 'command + ["--skip-archive-byte-check"]' in release_source
    assert "repeat build kept package identity" in release_source
    assert '"notebooks/*/*/versions/*"' in release_source
    assert "ATTRIBUTES_ROOT" not in inference_source
    assert "SELECTED_PROMPT" in inference_source
    assert "ds1-locateanything-prompt-freeze-verification.json" in inference_source
    assert "SHA256SUMS" in inference_source
    assert "ds2_frozen_locateanything" in inference_source
    assert "automatic-box SAM" in inference_source
    assert "automatic_box_mask_examples.png" in inference_source
    assert "proposal_examples.png" in inference_source
    assert "invalid-mask score rows" in inference_source
    assert "marked invalid_mask with no invented DINO score" in inference_source
    assert "predeclared bounded DS2 operational policy" in inference_source
    assert "production-safety policy" not in inference_source
    assert "recovery_snapshot.zip" in inference_source
    assert "per-image proposal status and discarded-box count" in inference_source
    assert "ed25f3a31f01632728cabb09d1542f84ab7b0056" in inference_source
    assert inference_source.index("run_ds2_inference.py") < inference_source.index("proposal_tracking = run_clearml_finalizer")
    assert "ATTRIBUTES_ROOT" in evaluation_source
    assert "1, 2, 3, 5, 10, and 20" in evaluation_source
    assert "proposal_summary.csv" in evaluation_source
    assert "unscoreable_summary.csv" in evaluation_source
    assert "invalid-mask candidate-view rows" in evaluation_source
    assert "Fusion uses only scored ranks" in evaluation_source
    assert "No accuracy cutoff" in evaluation_source
    assert "success_and_failure" in evaluation_source
    assert "verified_downstream_archive" in evaluation_source
    assert "Frozen configuration and confirmation status" in evaluation_source
    assert '"notebooks/*/*/versions/*"' in inference_source
    assert '"notebooks/*/*/versions/*"' in evaluation_source


def test_notebook08_metadata_is_private_and_has_the_expected_accelerator() -> None:
    metadata = {
        name: json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
        for name in ("kernel-metadata-08a.json", "kernel-metadata-08b.json", "kernel-metadata-08c.json")
    }
    assert all(value["is_private"] for value in metadata.values())
    assert metadata["kernel-metadata-08a.json"]["enable_gpu"] is False
    assert metadata["kernel-metadata-08b.json"]["enable_gpu"] is True
    assert metadata["kernel-metadata-08b.json"]["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel-metadata-08b.json"]["kernel_sources"] == [
        "esreekarreddy/toysad-8k-ds1-locateanything-prompt-freeze"
    ]
    assert metadata["kernel-metadata-08c.json"]["enable_gpu"] is False
    assert metadata["kernel-metadata-08c.json"]["kernel_sources"] == [
        "esreekarreddy/toysad-8k-ds2-frozen-pipeline"
    ]
    assert "esreekarreddy/vpi-toysad-ds2-frozen-pipeline-output" not in metadata[
        "kernel-metadata-08c.json"
    ]["dataset_sources"]
