from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from visual_peer_inspector.workflows.identity_refinement import (
    BASELINE_CONFIGURATION_ID,
    PROTOCOL,
    REQUIRED_VIEWS,
    VIEW_SETS,
    configuration_id,
)


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_keeps_the_existing_split_and_model() -> None:
    assert PROTOCOL["split_change"] is False
    assert PROTOCOL["scope"] == "existing 100-scene nested development screen"
    assert PROTOCOL["encoder"] == "facebook/dinov2-small"
    assert len(PROTOCOL["encoder_revision"]) == 40
    assert PROTOCOL["route"] == "automatic"
    assert tuple(sorted({view for values in VIEW_SETS.values() for view in values})) == REQUIRED_VIEWS
    assert BASELINE_CONFIGURATION_ID == configuration_id("global_current", 0.35, 3)


def test_runner_has_bounded_prediction_and_post_compute_tracking() -> None:
    runner = (ROOT / "scripts" / "run_identity_refinement.py").read_text(encoding="utf-8")
    logger = (ROOT / "scripts" / "log_identity_refinement.py").read_text(encoding="utf-8")
    assert "Identity feature extraction needs a CUDA GPU" in runner
    assert "devices_used" in runner
    assert "model_download_start" in runner
    assert "model_cpu_ready" in runner
    assert "model_cuda_ready" in runner
    assert "ClearML" not in runner
    assert "verify_seal(root)" in logger
    assert "shareable" in logger
    assert "private" not in logger
    assert "pending-review" in logger


def test_evaluator_keeps_older_pandas_compatibility() -> None:
    workflow = (
        ROOT
        / "src"
        / "visual_peer_inspector"
        / "workflows"
        / "identity_refinement.py"
    ).read_text(encoding="utf-8")
    assert "reset_index(names=" not in workflow


def test_notebook_is_clean_and_explains_the_decision_boundary() -> None:
    notebook_path = (
        ROOT
        / "notebooks"
        / "representation-and-scoring"
        / "development-identity-refinement"
        / "08i-development-identity-refinement.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook_path.name in source
    assert "existing 100-scene development screen" in source
    assert "does not change the 2,000-scene split" in source
    assert "DINOv2-small" in source
    assert "foreground patch" in source.lower()
    assert "Lock prediction-only identity candidates" in source
    assert "Only then open development correspondence" in source
    assert "No arbitrary accuracy gate" in source
    assert "ClearML" in source
    assert "recovery" in source.lower()
    assert "find_recovery(INPUT_ROOT) if EXPECT_RECOVERY else None" in source
    assert "private gallery" in source.lower()
    assert "08i-development-identity-refinement.ipynb" in source
    assert "—" not in source
    for index, cell in enumerate(notebook["cells"]):
        cell_source = "".join(cell.get("source", []))
        expected_id = sha256(
            f"{index}:{cell['cell_type']}:{cell_source}".encode("utf-8")
        ).hexdigest()[:8]
        assert cell.get("id") == expected_id
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")
