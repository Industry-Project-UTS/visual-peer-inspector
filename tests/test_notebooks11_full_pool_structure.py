from __future__ import annotations

import ast
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks/post-freeze-sensitivity"
NOTEBOOKS = {
    "11r": NOTEBOOK_ROOT / "11r-toysad8k-full-pool-release.ipynb",
    "11a": NOTEBOOK_ROOT / "11a-toysad8k-full-pool-frozen-inference.ipynb",
    "11b": NOTEBOOK_ROOT / "11b-toysad8k-full-pool-sensitivity-evaluation.ipynb",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _load(path)["cells"])


def test_notebooks_are_clean_and_build_deterministically() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/build_notebooks11_full_pool.py")], check=True)
    first = {key: sha256(path.read_bytes()).hexdigest() for key, path in NOTEBOOKS.items()}
    subprocess.run([sys.executable, str(ROOT / "scripts/build_notebooks11_full_pool.py")], check=True)
    assert {key: sha256(path.read_bytes()).hexdigest() for key, path in NOTEBOOKS.items()} == first

    for stage, path in NOTEBOOKS.items():
        notebook = _load(path)
        source = _source(path)
        assert path.name in source
        assert "ClearML" in source
        assert "—" not in source and "–" not in source
        assert "0.95" not in source and "95%" not in source
        assert "descriptive" in source.lower()
        assert len(notebook["cells"]) >= 12
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert not cell.get("outputs")
                ast.parse("".join(cell.get("source", [])), filename=f"{stage}:{index}")


def test_stage_boundaries_and_evidence_are_explicit() -> None:
    release = _source(NOTEBOOKS["11r"])
    inference = _source(NOTEBOOKS["11a"])
    evaluation = _source(NOTEBOOKS["11b"])
    assert "4,107" in release and "82,140" in release and "2,107" in release
    assert "development, calibration and internal-lock role labels" in release
    assert "Notebook 10B" in release and "evaluation_complete.json" in release
    assert "recovery" in release.lower() and "physically separate" in release
    assert "LocateAnything, SAM 2.1, DINOv2-small" in inference
    assert "prediction-only panels" in inference
    assert "--preflight-only" in inference
    assert "timeout=1800" in inference
    assert 'environment["HF_HUB_OFFLINE"] = "1"' in inference
    assert 'environment["TRANSFORMERS_OFFLINE"] = "1"' in inference
    assert 'RUN_MODE not in {"gate", "incremental", "full"}' in inference
    assert "prediction_lock.json" in evaluation
    assert "All failures remain in the denominators" in evaluation
    assert "clustered confidence intervals" in evaluation
    assert "cannot be used to revise the method" in evaluation
    for script_name in (
        "run_full_pool_release.py",
        "run_full_pool_inference.py",
        "run_full_pool_evaluation.py",
    ):
        runner = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert '"no_retuning": True' in runner


def test_metadata_separates_model_and_evaluator_resources() -> None:
    metadata = {
        stage: _load(NOTEBOOK_ROOT / f"kernel-metadata-{stage}.json")
        for stage in ("11r", "11a", "11b")
    }
    assert metadata["11r"]["id"] == "esreekarreddy/toysad-8k-post-freeze-full-pool-release"
    assert metadata["11a"]["id"] == "esreekarreddy/toysad-8k-post-freeze-full-pool-inference"
    assert metadata["11b"]["id"] == "esreekarreddy/toysad-8k-post-freeze-full-pool-evaluation"
    assert metadata["11a"]["kernel_sources"] == []
    assert metadata["11b"]["kernel_sources"] == []
    for stage in ("11a", "11b"):
        assert "esreekarreddy/vpi-toysad-development-method-freeze" in metadata[stage][
            "dataset_sources"
        ]
        assert "esreekarreddy/vpi-toysad-calibration-review-policy" in metadata[stage][
            "dataset_sources"
        ]
    inference_inputs = " ".join(
        metadata["11a"]["dataset_sources"] + metadata["11a"]["kernel_sources"]
    ).lower()
    assert "model-input" in inference_inputs
    assert "evaluator" not in inference_inputs
    assert "label" not in inference_inputs and "crosswalk" not in inference_inputs
    evaluator_inputs = " ".join(metadata["11b"]["dataset_sources"]).lower()
    assert "predictions" in evaluator_inputs and "evaluator-only" in evaluator_inputs
    assert metadata["11a"]["enable_gpu"] is True
    assert metadata["11r"]["enable_gpu"] is False
    assert metadata["11b"]["enable_gpu"] is False
