"""Structure tests for the clean-child compact-cache queue builder."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_current_launch_plan import plan
from build_isolated_cache_queue import (
    NATIVE_CACHE_NOTEBOOK,
    _scientific_cells,
    build_cache_child,
    patch_cache_notebook,
)


def _generated_fixture(path, value, job):
    canonical = nbformat.read(ROOT / NATIVE_CACHE_NOTEBOOK, as_version=4)
    bootstrap = next(
        cell for cell in canonical.cells
        if cell.cell_type == "code" and "BOOTSTRAP = _runtime.bootstrap" in cell.source
    )
    marker = "    BOOTSTRAP = _runtime.bootstrap"
    bootstrap.source = bootstrap.source.replace(
        marker,
        "    from current_launch import configure_runtime\n"
        f"    configure_runtime({value!r}, {job['id']!r}, CODE_ROOT)\n" + marker,
    )
    generated = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# generated fixture"),
        bootstrap,
        nbformat.v4.new_code_cell(f"LAUNCH_PLAN = {value!r}"),
        nbformat.v4.new_code_cell(
            "from post_completion_retention import begin_cache_job\n"
            "_LAUNCH_SCIENCE_REUSED = begin_cache_job(LAUNCH_PLAN, 'development-cache-00', 'a'*64)"),
        nbformat.v4.new_code_cell("if not _LAUNCH_SCIENCE_REUSED:\n    RESULT = 'science'"),
        nbformat.v4.new_code_cell(
            "from post_completion_retention import retain_completed_cache\n"
            "retain_completed_cache(LAUNCH_PLAN, 'development-cache-00', 'a'*64)"),
        nbformat.v4.new_code_cell(
            "from current_launch import complete_job\n"
            "JOB_COMPLETION = complete_job(LAUNCH_PLAN, 'development-cache-00', 'a'*64)"),
    ])
    nbformat.write(generated, path)


def test_child_preserves_cache_gate_order_and_clean_runtime_boundary():
    value = plan()
    job = next(row for row in value["jobs"] if row["id"] == "development-cache-00")
    child = build_cache_child(value, job, _scientific_cells(ROOT))
    ast.parse(child)
    assert child.index("begin_cache_job(") < child.index("if not _LAUNCH_SCIENCE_REUSED:")
    assert child.index("if not _LAUNCH_SCIENCE_REUSED:") < child.index("retain_completed_cache(")
    assert child.index("retain_completed_cache(") < child.index("complete_job(")
    assert 'Path("/kaggle/working/runtime-receipts")' in child
    assert '"scientific_run_started": False' in child
    assert 'scientific_run_started=True' in child
    assert 'status="runtime_imports_verified"' in child
    assert 'find_spec("torchvision")' in child
    assert "Dinov2Model" in child


def test_generated_cache_notebook_becomes_one_clean_wrapper(tmp_path):
    value = plan()
    job = copy.deepcopy(next(row for row in value["jobs"] if row["id"] == "development-cache-00"))
    path = tmp_path / Path(NATIVE_CACHE_NOTEBOOK).name
    _generated_fixture(path, value, job)
    receipt = patch_cache_notebook(path, ROOT, value, job)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 1
    wrapper = code[0]
    compile(wrapper, path.name, "exec")
    assert receipt["job_id"] == job["id"] and receipt["notebook_size_bytes"] < 1024 ** 2
    assert "prepare_runtime_environment.py" in wrapper
    assert 'Path("/kaggle/working/runtime-profiles")' in wrapper
    assert "VPI_RESOLVED_CODE_ROOT" in wrapper and '"PYTHONPATH"' in wrapper
    assert '[sys.executable, "-S", "-B", str(_child_path)]' in wrapper
    assert '_environment["PYTHONUNBUFFERED"] = "1"' in wrapper
    assert '_environment["MPLBACKEND"] = "Agg"' in wrapper
    assert '"MPLCONFIGDIR"' in wrapper
    assert "isolated_child_failed" in wrapper and "scientific_run_started" in wrapper
    assert not any(cell.outputs or cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")


def test_only_compact_prepare_shards_are_eligible():
    value = plan()
    matching = [
        job for job in value["jobs"]
        if job.get("notebook") == NATIVE_CACHE_NOTEBOOK
        and job.get("compact_checkpoints") is True
        and job.get("config", {}).get("action") == "prepare_shard"
    ]
    assert len(matching) == 53
    assert all(job["config"]["maximum_output_bytes"] == 15 * 1024 ** 3 for job in matching)
