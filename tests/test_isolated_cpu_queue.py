"""Structure tests for all clean-child fixed CPU science jobs."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_current_launch_plan import plan
from build_isolated_cpu_queue import (
    CHILD_TIMEOUT_SECONDS,
    PUBLICATION_JOBS,
    _cpu_science_jobs,
    _scientific_cells,
    build_cpu_child,
    patch_cpu_notebook,
)


def _generated_fixture(path, value, job):
    """Reproduce the launch builder's controls around canonical notebook cells."""
    notebook = nbformat.read(ROOT / job["notebook"], as_version=4)
    positions = [
        index for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "BOOTSTRAP = _runtime.bootstrap" in cell.source
    ]
    assert len(positions) == 1
    bootstrap = copy.deepcopy(notebook.cells[positions[0]])
    marker = "    BOOTSTRAP = _runtime.bootstrap"
    bootstrap.source = bootstrap.source.replace(
        marker,
        "    from current_launch import configure_runtime\n"
        f"    configure_runtime({value!r}, {job['id']!r}, CODE_ROOT)\n" + marker,
    )
    notebook.cells[positions[0]] = bootstrap
    if job.get("config", {}).get("publish_tracking"):
        notebook.cells.append(nbformat.v4.new_code_cell(
            "from current_launch import publish_completed_job\n"
            f"LAUNCH_TRACKING = publish_completed_job(LAUNCH_PLAN, {job['id']!r}, CODE_ROOT)"
        ))
    notebook.cells.append(nbformat.v4.new_code_cell(
        "from current_launch import complete_job\n"
        "JOB_COMPLETION = complete_job(LAUNCH_PLAN, " + repr(job["id"]) + ", 'a'*64)"
    ))
    nbformat.write(notebook, path)


def _jobs():
    value = plan()
    return value, {job["id"]: job for job in value["jobs"]}


def _embedded_child(wrapper):
    tree = ast.parse(wrapper)
    matches = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_child_path"
        and node.func.attr == "write_text"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    assert len(matches) == 1
    return matches[0]


def test_cpu_selector_covers_all_nonsetup_cpu_forms_only():
    value, by_id = _jobs()
    selected = {job["id"] for job in _cpu_science_jobs(value)}
    assert {
        "calibration-regions", "development-predict", "08f1", "09a", "09b",
        "09b1", "09c", "09d", "10r-seen", "10r-unseen", "10a-automatic",
        "10a1-assisted", "10a-finalize-automatic", "10b",
    } <= selected
    assert {"setup-cpu", "setup-dino", "calibration-cache-00", "calibration-regions-00"}.isdisjoint(selected)
    assert all(by_id[job_id].get("runtime", "cpu") == "cpu" for job_id in selected)
    assert all(not by_id[job_id].get("setup_job", False) for job_id in selected)


def test_every_cpu_form_has_parseable_canonical_science_and_separate_publication():
    value, _ = _jobs()
    for job in _cpu_science_jobs(value):
        cells = _scientific_cells(ROOT, job)
        assert cells
        for source in cells:
            compile(source, job["notebook"], "exec")
        combined = "\n".join(cells)
        assert "complete_job(" not in combined
        assert "publish_completed_job(" not in combined
        if job["id"] in PUBLICATION_JOBS:
            assert "publish_next_stage.py" not in combined


def test_cpu_child_keeps_science_before_completion_and_publication_separate():
    value, by_id = _jobs()
    prediction = build_cpu_child(
        value, by_id["development-predict"], _scientific_cells(ROOT, by_id["development-predict"]))
    evaluation = build_cpu_child(
        value, by_id["09b"], _scientific_cells(ROOT, by_id["09b"]))
    compile(prediction, "prediction-child.py", "exec")
    compile(evaluation, "evaluation-child.py", "exec")
    assert "Isolated CPU child can see ambient site packages" in prediction
    assert "Pinned CPU dependency resolved outside the private target" in prediction
    assert "VPI_LAUNCH_PLAN_PATH" in prediction
    assert prediction.index("science_running") < prediction.index("science_complete")
    assert prediction.index("science_complete") < prediction.index("complete_job(")
    assert "publish_completed_job(" not in prediction
    assert evaluation.index("evaluate_predictions('09b'") < evaluation.index("render_current_report(")
    assert evaluation.index("render_current_report(") < evaluation.index("science_complete")
    assert "publish_completed_job(" not in evaluation
    assert "publish_next_stage.py" not in evaluation
    assert "pending_separate_publication" in evaluation


@pytest.mark.parametrize(
    "job_id,science_marker",
    [
        ("calibration-regions", "region_merge"),
        ("08f1", "release_calibration(cfg,ROOT)"),
        ("09a", "predict_current('09a'"),
        ("09b", "evaluate_predictions('09b'"),
        ("09d", "evaluate_predictions('09d'"),
        ("10r-seen", "release_verified_cache(CONFIG, CODE_ROOT)"),
        ("10a-automatic", "predict_official(CONFIG, CODE_ROOT)"),
        ("10b", "evaluate_official(CONFIG, CODE_ROOT)"),
    ],
)
def test_generated_cpu_forms_become_one_clean_isolated_wrapper(tmp_path, job_id, science_marker):
    value, by_id = _jobs()
    job = by_id[job_id]
    path = tmp_path / f"{job_id}.ipynb"
    _generated_fixture(path, value, job)
    receipt = patch_cpu_notebook(path, ROOT, value, job)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 1
    wrapper = code[0]
    compile(wrapper, path.name, "exec")
    assert receipt["job_id"] == job_id and receipt["notebook_size_bytes"] < 1024 ** 2
    assert "prepare_runtime_environment.py" in wrapper
    assert '[sys.executable, "-S", "-B", str(_child_path)]' in wrapper
    assert f"timeout={CHILD_TIMEOUT_SECONDS}" in wrapper
    assert "VPI_RESOLVED_CODE_ROOT" in wrapper
    assert "VPI_LAUNCH_PLAN_PATH" in wrapper
    assert job["notebook"] in wrapper
    child = _embedded_child(wrapper)
    compile(child, f"{job_id}-child.py", "exec")
    assert science_marker in child
    assert not any(
        cell.outputs or cell.execution_count is not None
        for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "publish_completed_job(" not in wrapper
    if job_id in PUBLICATION_JOBS:
        assert "publish_next_stage.py" not in wrapper
        assert "pending_separate_publication" in wrapper
        assert "shareable" in wrapper and "_Image" in wrapper


def test_official_prebootstrap_constants_move_inside_child(tmp_path):
    value, by_id = _jobs()
    job = by_id["10a-automatic"]
    path = tmp_path / "10a.ipynb"
    _generated_fixture(path, value, job)
    before = nbformat.read(path, as_version=4)
    assert any(
        cell.cell_type == "code" and "STAGE = '10a'" in cell.source
        for cell in before.cells
    )
    patch_cpu_notebook(path, ROOT, value, job)
    after = nbformat.read(path, as_version=4)
    code = [cell.source for cell in after.cells if cell.cell_type == "code"]
    assert len(code) == 1
    child = _embedded_child(code[0])
    assert "STAGE = '10a'" in child
    assert code[0].index("_child_path.write_text") < code[0].index('[sys.executable, "-S", "-B"')
