from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "development-freeze"
NOTEBOOK_PATH = NOTEBOOK_ROOT / "08d-toysad8k-primary-protocol-release.ipynb"


def test_notebook08d_is_clean_and_has_the_release_boundary() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert NOTEBOOK_PATH.name in source
    assert "—" not in source
    assert "development remainder" in source
    assert "Calibration RGB stays sealed until Notebook 09A" in source
    assert "Internal-lock RGB stays sealed until Notebook 09C" in source
    assert "run_primary_protocol_release.py" in source
    assert "development_stage_membership.csv" in source
    assert "development-remainder/model-input" in source
    assert "development_rgb_contact_sheet.png" in source
    assert "private/figures" in source
    assert "Visual Peer Inspector/05 Development Freeze" in source
    assert source.index("run_primary_protocol_release.py") < source.index(
        "tracking_receipt = finish_clearml"
    )
    assert "calibration_labels.jsonl" not in source
    assert "internal_lock_labels.jsonl" not in source
    assert "ATTRIBUTES_ROOT" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
            compile(
                "".join(cell.get("source", [])),
                f"{NOTEBOOK_PATH}:cell-{index}",
                "exec",
            )


def test_notebook08d_metadata_is_private_and_cpu_only() -> None:
    metadata = json.loads(
        (NOTEBOOK_ROOT / "kernel-metadata-08d.json").read_text(encoding="utf-8")
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["code_file"] == NOTEBOOK_PATH.name
    assert "esreekarreddy/toysad8k-full-private-eda" in metadata["dataset_sources"]
