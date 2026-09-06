from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_one_shot_evaluation_notebooks.py"
NOTEBOOKS = {
    "09d": ROOT
    / "notebooks/protected-evaluation/09d-toysad8k-internal-lock-one-shot-evaluation.ipynb",
    "10b": ROOT
    / "notebooks/official-evaluation/10b-toysad8k-official-seen-unseen-evaluation.ipynb",
}


def _sources(path: Path) -> tuple[str, str, list[dict]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    source = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    return markdown, source, cells


def test_builder_is_deterministic() -> None:
    before = {stage: path.read_bytes() for stage, path in NOTEBOOKS.items()}
    subprocess.run([sys.executable, str(BUILDER)], check=True, cwd=ROOT)
    after = {stage: path.read_bytes() for stage, path in NOTEBOOKS.items()}
    assert after == before


def test_notebooks_are_clean_and_code_cells_parse() -> None:
    for path in NOTEBOOKS.values():
        markdown, source, cells = _sources(path)
        assert path.name in markdown
        assert "—" not in markdown + source
        assert "–" not in markdown + source
        assert "0.95" not in markdown + source
        assert "CLEARML_API_ACCESS_KEY =" not in source
        assert "CLEARML_API_SECRET_KEY =" not in source
        assert '{"Publish only' not in markdown
        assert 'if stage ==' not in markdown
        assert "run_one_shot_evaluation.py" in source
        assert '"contract"' in source
        assert '"--panel-count", "16"' in source
        assert '"--bootstrap-repetitions", "10000"' in source
        assert source.index("subprocess.run(command") < source.index(
            "clearml_receipt = run_tracking_finalizer"
        )
        for cell in cells:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                ast.parse("".join(cell["source"]))


def test_09d_contains_internal_lock_receipt_and_no_threshold_gate() -> None:
    markdown, source, _cells = _sources(NOTEBOOKS["09d"])
    combined = (markdown + source).lower()
    assert "400-scene internal-lock" in combined
    assert "internal_lock_result_receipt.json" in source
    assert "universal accuracy gate" in combined
    assert "does not change the method or policy" in combined


def test_10b_reports_official_splits_separately() -> None:
    markdown, source, _cells = _sources(NOTEBOOKS["10b"])
    combined = (markdown + source).lower()
    assert "seen-test" in combined
    assert "unseen-test" in combined
    assert "pooled" in combined and "descriptive" in combined
    assert "no official result may change the method or policy" in combined


def test_clearml_finalizer_verifies_manifest_and_exact_aggregate_allowlist() -> None:
    source = (ROOT / "scripts/run_one_shot_evaluation.py").read_text(encoding="utf-8")
    assert "validate_tracking_safe_root" in source
    assert "allowed_relative_paths" in source
    assert "safe_artifact_inventory_sha256" in source


def test_kernel_metadata_uses_cpu_and_frozen_dependencies() -> None:
    metadata_09d = json.loads(
        (ROOT / "notebooks/protected-evaluation/kernel-metadata-09d.json").read_text(
            encoding="utf-8"
        )
    )
    metadata_10b = json.loads(
        (ROOT / "notebooks/official-evaluation/kernel-metadata-10b.json").read_text(
            encoding="utf-8"
        )
    )
    for metadata in (metadata_09d, metadata_10b):
        assert metadata["enable_gpu"] is False
        assert metadata["enable_tpu"] is False
        assert "esreekarreddy/vpi-toysad-development-method-freeze" in metadata[
            "dataset_sources"
        ]
        assert "esreekarreddy/vpi-toysad-calibration-review-policy" in metadata[
            "dataset_sources"
        ]
    assert metadata_09d["kernel_sources"] == [
        "esreekarreddy/09c-toysad8k-protected-internal-lock-inference"
    ]
    assert any(
        "internal-lock400-model-input" in source
        for source in metadata_09d["dataset_sources"]
    )
    assert metadata_10b["kernel_sources"] == []
    assert "esreekarreddy/vpi-toysad-internal-lock-decision" in metadata_10b[
        "dataset_sources"
    ]
    assert any(
        "official-seen-model-input" in source
        for source in metadata_10b["dataset_sources"]
    )
    assert any(
        "official-unseen-model-input" in source
        for source in metadata_10b["dataset_sources"]
    )
    assert metadata_10b["dataset_sources"] == [
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-official-seen-evaluator",
        "esreekarreddy/vpi-toysad-official-unseen-evaluator",
        "esreekarreddy/vpi-toysad-official-seen-prediction",
        "esreekarreddy/vpi-toysad-official-unseen-prediction",
        "esreekarreddy/vpi-toysad-official-prediction-summary",
        "esreekarreddy/vpi-toysad-official-seen-model-input",
        "esreekarreddy/vpi-toysad-official-unseen-model-input",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
        "esreekarreddy/vpi-toysad-internal-lock-decision",
    ]
