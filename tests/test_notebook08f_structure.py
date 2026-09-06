from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "development-freeze"
NOTEBOOK_PATH = NOTEBOOK_ROOT / "08f-toysad8k-cross-view-identity-freeze.ipynb"


def notebook_source() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    return notebook, source


def test_notebook08f_is_clean_and_states_the_prediction_first_boundary() -> None:
    notebook, source = notebook_source()

    assert NOTEBOOK_PATH.name in source
    assert "—" not in source
    assert "August" not in source
    assert "two to five uploaded views" in source
    assert "embedding-only" in source
    assert "normalized-geometry" in source
    assert "Lock all identity predictions" in source
    assert "Only then open evaluator correspondence" in source
    assert "camera metadata" in source.lower()
    assert "evaluator diagnostic" in source
    assert "No arbitrary universal accuracy threshold is used" in source
    assert "0.95" not in source
    assert "run_cross_view_identity_freeze.py" in source
    assert "--clearml-wall-time-seconds" in source
    assert "method_summary.csv" in source
    assert "clustered_intervals.csv" in source
    assert "paired_method_intervals.csv" in source
    assert "application_output_contract.json" in source
    assert "odd-one-out" in source
    assert "sam-evidence.png" in source
    assert "Human correction remains an optional review feature" in source
    assert source.index("subprocess.run(command") < source.index(
        "tracking_receipt = json.loads"
    )

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")
        compile(
            "".join(cell.get("source", [])),
            f"{NOTEBOOK_PATH}:cell-{index}",
            "exec",
        )


def test_notebook08f_metadata_is_private_cpu_and_consumes_08e() -> None:
    metadata = json.loads(
        (NOTEBOOK_ROOT / "kernel-metadata-08f.json").read_text(encoding="utf-8")
    )

    assert metadata["code_file"] == NOTEBOOK_PATH.name
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert "esreekarreddy/toysad-8k-frozen-development-pipeline" in metadata[
        "kernel_sources"
    ]
    assert "esreekarreddy/vpi-toysad-development-identity-evaluator" in metadata[
        "dataset_sources"
    ]


def test_runner_bounds_clearml_and_keeps_credentials_out_of_arguments() -> None:
    source = (ROOT / "scripts" / "run_cross_view_identity_freeze.py").read_text(
        encoding="utf-8"
    )

    assert "subprocess.run(" in source
    assert "timeout=wall_time_seconds" in source
    assert '"status": "timed_out"' in source
    assert '"status": "failed"' in source
    assert "--clearml-finalize-only" in source
    assert "validate_tracking_safe_root" in source
    assert "allowed_relative_paths" in source
    assert "CLEARML_API_ACCESS_KEY" not in source
    assert "CLEARML_API_SECRET_KEY" not in source
