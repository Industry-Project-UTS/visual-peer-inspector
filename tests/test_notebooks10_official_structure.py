from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks/official-evaluation"


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_official_notebooks_are_clean_compile_and_state_boundaries() -> None:
    paths = {
        "10r": NOTEBOOK_ROOT / "10r-toysad8k-official-release.ipynb",
        "10a": NOTEBOOK_ROOT / "10a-toysad8k-official-seen-frozen-inference.ipynb",
        "10a1": NOTEBOOK_ROOT / "10a1-toysad8k-official-unseen-frozen-inference.ipynb",
        "10a-finalize": NOTEBOOK_ROOT
        / "10a-finalize-toysad8k-official-prediction-summary.ipynb",
    }
    for path in paths.values():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = _source(path)
        assert path.name in source
        assert "—" not in source and "–" not in source
        assert "0.95" not in source and "95%" not in source
        assert "ClearML" in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert not cell.get("outputs")
                compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")
    release = _source(paths["10r"])
    assert "one official split at a time" in release
    assert "storage preflight" in release
    assert "seen-evaluator-only" in release and "unseen-evaluator-only" in release
    assert "VPI_MAXIMUM_NEW_SCENES" in release
    assert "VPI_RESUME_ROOT must contain exactly one official release recovery pair" in release
    assert '"--recovery-output-directory"' in release
    assert "not written before exact split coverage" in release
    assert "internal_lock_result_receipt.json" in release
    assert release.index("verify_internal_lock_decision") < release.index(
        'SOURCE_ROOT = next('
    )
    runner = (ROOT / "scripts/run_official_release.py").read_text(encoding="utf-8")
    assert "write_recovery_tracking_root" in runner
    assert 'tracking_root = args.output_root / "tracking-safe"' in runner
    assert "Visual Peer Inspector/08 Official Release" in runner
    for stage, expected_role in (("10a", "seen"), ("10a1", "unseen")):
        inference = _source(paths[stage])
        assert "VPI_RESUME_ROOT must contain exactly one recovery ZIP" in inference
        assert "--export-recovery" in inference and "--prune-after-recovery" in inference
        recovery_index = inference.index("recovery = find_recovery_pair")
        export_index = inference.index(
            'if RUN_PHASE in {"gate", "incremental"}', recovery_index
        )
        assert recovery_index < export_index
        assert "masked-on-black" in inference
        assert "accuracy metrics" in inference.lower()
        assert '"preflight"' in inference
        assert "timeout=1800" in inference
        assert 'environment["HF_HUB_OFFLINE"] = "1"' in inference
        assert inference.index('"preflight"') < inference.index(
            "subprocess.run(command, check=True, env=environment)"
        )
        assert f'VPI_OFFICIAL_COHORT", "{expected_role}"' in inference
        assert f'PINNED_OFFICIAL_COHORT = "{expected_role}"' in inference
        assert "REQUESTED_OFFICIAL_COHORT != PINNED_OFFICIAL_COHORT" in inference
        assert "OFFICIAL_COHORT = PINNED_OFFICIAL_COHORT" in inference
        assert f"official-{expected_role}-prediction" in inference
    finalizer = _source(paths["10a-finalize"])
    assert '"--run-mode", "finalize"' in finalizer
    assert "official-seen-prediction" in finalizer
    assert "official-unseen-prediction" in finalizer
    assert "official-prediction-summary" in finalizer
    inference_runner = (ROOT / "scripts/run_official_inference.py").read_text(
        encoding="utf-8"
    )
    assert "Visual Peer Inspector/09 Frozen Evaluation" in inference_runner


def test_official_metadata_uses_exact_upstreams_and_no_evaluator_on_10a() -> None:
    metadata10r = json.loads((NOTEBOOK_ROOT / "kernel-metadata-10r.json").read_text())
    metadata10a = json.loads((NOTEBOOK_ROOT / "kernel-metadata-10a.json").read_text())
    metadata10a1 = json.loads((NOTEBOOK_ROOT / "kernel-metadata-10a1.json").read_text())
    metadata_finalize = json.loads(
        (NOTEBOOK_ROOT / "kernel-metadata-10a-finalize.json").read_text()
    )
    assert metadata10r["id"] == "esreekarreddy/toysad-8k-official-release"
    assert metadata10r["enable_gpu"] is False
    assert "esreekarreddy/vpi-toysad-internal-lock-decision" in metadata10r[
        "dataset_sources"
    ]
    assert metadata10a["id"] == "esreekarreddy/toysad-8k-official-frozen-inference"
    assert metadata10a["kernel_sources"] == []
    assert "esreekarreddy/vpi-toysad-development-method-freeze" in metadata10a[
        "dataset_sources"
    ]
    assert "esreekarreddy/vpi-toysad-calibration-review-policy" in metadata10a[
        "dataset_sources"
    ]
    assert "esreekarreddy/vpi-toysad-internal-lock-decision" in metadata10a[
        "dataset_sources"
    ]
    joined = " ".join(metadata10a["dataset_sources"] + metadata10a["kernel_sources"]).lower()
    assert "evaluator" not in joined and "crosswalk" not in joined and "label" not in joined
    assert "esreekarreddy/vpi-toysad-official-seen-model-input" in metadata10a[
        "dataset_sources"
    ]
    assert "esreekarreddy/vpi-toysad-official-unseen-model-input" not in metadata10a[
        "dataset_sources"
    ]
    unseen_joined = " ".join(
        metadata10a1["dataset_sources"] + metadata10a1["kernel_sources"]
    ).lower()
    assert "evaluator" not in unseen_joined and "label" not in unseen_joined
    assert "esreekarreddy/vpi-toysad-official-unseen-model-input" in metadata10a1[
        "dataset_sources"
    ]
    assert "esreekarreddy/vpi-toysad-official-seen-model-input" not in metadata10a1[
        "dataset_sources"
    ]
    assert metadata_finalize["enable_gpu"] is False
    assert metadata_finalize["dataset_sources"] == [
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-official-seen-prediction",
        "esreekarreddy/vpi-toysad-official-unseen-prediction",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
        "esreekarreddy/vpi-toysad-internal-lock-decision",
    ]


@pytest.mark.parametrize(
    ("filename", "pinned", "conflicting"),
    (
        ("10a-toysad8k-official-seen-frozen-inference.ipynb", "seen", "unseen"),
        ("10a1-toysad8k-official-unseen-frozen-inference.ipynb", "unseen", "seen"),
    ),
)
def test_official_inference_entry_point_rejects_conflicting_cohort_environment(
    filename: str, pinned: str, conflicting: str, monkeypatch
) -> None:
    notebook = json.loads((NOTEBOOK_ROOT / filename).read_text(encoding="utf-8"))
    setup = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "PINNED_OFFICIAL_COHORT" in "".join(cell.get("source", []))
    )
    lines = setup.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("NOTEBOOK_FILE"))
    stop = next(
        index for index, line in enumerate(lines) if line.startswith("EXECUTION_ENVIRONMENT")
    )
    guard = "\n".join(lines[start:stop])

    monkeypatch.setenv("VPI_RUN_PHASE", "gate")
    monkeypatch.setenv("VPI_OFFICIAL_COHORT", pinned)
    namespace = {"os": os}
    exec(guard, namespace)  # noqa: S102 - execute the isolated generated guard
    assert namespace["OFFICIAL_COHORT"] == pinned

    monkeypatch.setenv("VPI_OFFICIAL_COHORT", conflicting)
    with pytest.raises(ValueError, match=f"is pinned to {pinned}"):
        exec(guard, {"os": os})  # noqa: S102 - exercise the generated guard
