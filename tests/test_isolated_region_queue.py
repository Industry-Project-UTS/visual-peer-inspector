"""Structure tests for clean-child LocateAnything/SAM region jobs."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_current_launch_plan import plan
from build_isolated_region_queue import (
    CHILD_TIMEOUT_SECONDS,
    NATIVE_REGION_NOTEBOOK,
    _region_jobs,
    _scientific_cells,
    build_region_child,
    patch_region_notebook,
)


def _generated_fixture(path, value, job):
    canonical = nbformat.read(ROOT / NATIVE_REGION_NOTEBOOK, as_version=4)
    bootstrap = copy.deepcopy(next(
        cell for cell in canonical.cells
        if cell.cell_type == "code" and "BOOTSTRAP = _runtime.bootstrap" in cell.source
    ))
    marker = "    BOOTSTRAP = _runtime.bootstrap"
    bootstrap.source = bootstrap.source.replace(
        marker,
        "    from current_launch import configure_runtime\n"
        f"    configure_runtime({value!r}, {job['id']!r}, CODE_ROOT)\n" + marker,
    )
    cells = [nbformat.v4.new_markdown_cell("# generated fixture"), bootstrap]
    cells.extend(nbformat.v4.new_code_cell(source) for source in _scientific_cells(ROOT))
    cells.append(nbformat.v4.new_code_cell(
        "from current_launch import complete_job\n"
        "JOB_COMPLETION = complete_job(LAUNCH_PLAN, " + repr(job["id"]) + ", 'a'*64)"))
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def test_region_child_keeps_pinned_gpu_boundary_and_completion_order():
    value = plan()
    job = next(row for row in value["jobs"] if row["id"] == "calibration-regions-00")
    child = build_region_child(job, _scientific_cells(ROOT))
    compile(child, "region-child.py", "exec")
    assert "Isolated region child can see ambient site packages" in child
    assert "Pinned region dependency resolved outside the private target" in child
    assert "torch.cuda.is_available()" in child
    assert "verify_installed_sam2_revision" in child
    assert "Pinned SAM-2 import resolved outside the private target" in child
    assert "Pinned SAM-2 checkpoint bytes differ" in child
    assert child.index('status="sam2_preflight_verified"') < child.index("science_running")
    assert "proposal_recovery_seeded" in child
    assert child.index("proposal_recovery_seeded") < child.index("science_running")
    assert child.index("science_running") < child.index("science_complete")
    assert child.index("science_complete") < child.index("complete_job(")


def test_generated_region_notebook_becomes_one_clean_wrapper(tmp_path):
    value = plan()
    job = next(row for row in value["jobs"] if row["id"] == "calibration-regions-00")
    path = tmp_path / Path(NATIVE_REGION_NOTEBOOK).name
    _generated_fixture(path, value, job)
    receipt = patch_region_notebook(path, ROOT, value, job)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 1
    wrapper = code[0]
    compile(wrapper, path.name, "exec")
    assert receipt["job_id"] == job["id"] and receipt["notebook_size_bytes"] < 1024 ** 2
    assert "prepare_runtime_environment.py" in wrapper
    assert '[sys.executable, "-S", "-B", str(_child_path)]' in wrapper
    assert f"timeout={CHILD_TIMEOUT_SECONDS}" in wrapper
    assert "VPI_RESOLVED_CODE_ROOT" in wrapper and "VPI_LAUNCH_PLAN_PATH" in wrapper
    assert not any(
        cell.outputs or cell.execution_count is not None
        for cell in notebook.cells if cell.cell_type == "code"
    )


def test_only_fixed_region_extract_shards_are_eligible():
    jobs = _region_jobs(plan())
    assert jobs
    assert all(job["runtime"] == "regions-gpu" for job in jobs)
    assert all(job["config"]["action"] == "extract" for job in jobs)
    assert all(job["config"]["shard_count"] >= 1 for job in jobs)
    assert all(job["config"]["maximum_new_views"] >= 1 for job in jobs)
