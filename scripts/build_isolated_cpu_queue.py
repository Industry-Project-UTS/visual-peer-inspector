#!/usr/bin/env python3
"""Run every uncompleted fixed CPU science job in a clean child interpreter.

Kaggle may import NumPy and related packages before the first notebook cell.
Those ambient imports cannot be replaced safely in that process.  This builder
copies an existing queue, replaces each uncompleted non-setup CPU notebook with
a one-cell wrapper, and runs its canonical science in a fresh ``python -S``
child using the already sealed CPU wheelhouse.

Only terminally failed CPU jobs may be removed from queue state.  Completed
dependencies and their receipts are preserved byte-for-byte.  Aggregate
ClearML publication remains a separate post-seal operation.  This script does
not call Kaggle or change the scientific plan.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
import uuid

import nbformat

from build_isolated_cache_queue import _source


SCHEMA = "vpi-isolated-cpu-queue/v2"
CHILD_TIMEOUT_SECONDS = 14_400
PUBLICATION_JOBS = frozenset({"development-evaluate", "09b", "09d", "10b"})
EVALUATION_JOBS = PUBLICATION_JOBS


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _phase_write():
    return "_runtime.atomic_json(_PHASE_PATH, _PHASE)"


def _cpu_science_jobs(plan):
    """Return fixed CPU jobs while excluding setup and DINO/GPU workloads."""
    return [
        job for job in plan["jobs"]
        if job.get("runtime", "cpu") == "cpu" and not job.get("setup_job", False)
    ]


def _scientific_cells(source_root, job):
    """Read this job's canonical code cells without launch/publication control.

    Some official notebooks define their stage constants before the bootstrap,
    so all non-bootstrap code cells are retained in original order.  The 10B
    optional ClearML cell is deliberately omitted: publication is retried from
    sealed aggregates and must never control scientific completion.
    """
    path = Path(source_root) / job["notebook"]
    notebook = json.loads(path.read_text())
    positions = [
        index for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
        and "BOOTSTRAP = _runtime.bootstrap" in _source(cell)
    ]
    if len(positions) != 1:
        raise ValueError(f"Canonical CPU notebook {job['id']} must have one verified bootstrap")
    cells = []
    removed_publication = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code" or index == positions[0]:
            continue
        source = _source(cell)
        if job["id"] in PUBLICATION_JOBS and "publish_next_stage.py" in source:
            removed_publication += 1
            continue
        if any(marker in source for marker in (
                "publish_completed_job(", "complete_job(", "configure_runtime(", "LAUNCH_PLAN")):
            raise ValueError(f"Canonical CPU science for {job['id']} contains launch control")
        ast.parse(source)
        cells.append(source)
    if not cells:
        raise ValueError(f"Canonical CPU notebook {job['id']} has no scientific cells")
    if job["id"] == "10b" and removed_publication != 1:
        raise ValueError("10B must have exactly one separately bounded publication cell")
    if job["id"] in PUBLICATION_JOBS and any(
            "publish_next_stage.py" in source or "publish_completed_job(" in source
            for source in cells):
        raise ValueError(f"Publication leaked into isolated science for {job['id']}")
    return cells


def build_cpu_child(plan, job, scientific_cells):
    """Return canonical fixed science plus completion gates for a clean child."""
    science = textwrap.indent("\n\n".join(scientific_cells), "    ")
    program = f'''from pathlib import Path
import importlib
import json
import os
import sys
import types

CODE_ROOT = Path(os.environ["VPI_RESOLVED_CODE_ROOT"]).resolve(strict=True)
NOTEBOOK_RELATIVE = {job['notebook']!r}
RUNTIME_PROFILE = None
sys.path.insert(0, str(CODE_ROOT / "scripts"))
sys.path.insert(0, str(CODE_ROOT / "src"))
import current_notebook_runtime as _runtime
BOOTSTRAP = _runtime.bootstrap(CODE_ROOT, NOTEBOOK_RELATIVE, runtime_profile=RUNTIME_PROFILE)
CODE_PROVENANCE = BOOTSTRAP["provenance"]
ROOT = CODE_ROOT

def display(value):
    print(json.dumps(value, sort_keys=True, default=str), flush=True)

class Image:
    def __init__(self, filename):
        self.filename = str(filename)
    def __str__(self):
        return self.filename

# Canonical report cells use IPython only to construct display objects.  Keep
# the child dependency target closed by supplying that tiny display surface.
_ipython = types.ModuleType("IPython")
_ipython.__path__ = []
_ipython.get_ipython = lambda: None
_ipython.version_info = (0, 0)
_ipython_display = types.ModuleType("IPython.display")
_ipython_display.display = display
_ipython_display.Image = Image
_ipython.display = _ipython_display
sys.modules.setdefault("IPython", _ipython)
sys.modules.setdefault("IPython.display", _ipython_display)

LAUNCH_PLAN = json.loads(Path(os.environ["VPI_LAUNCH_PLAN_PATH"]).read_text())
_JOB_ID = {job['id']!r}
_PHASE_PATH = Path("/kaggle/working/runtime-receipts") / (_JOB_ID + "-isolated-child.json")
_PHASE = {{"schema": "vpi-isolated-cpu-child/v1", "job_id": _JOB_ID,
          "status": "bootstrap_verified", "scientific_run_started": False}}
{_phase_write()}
try:
    _dependency_target = Path(os.environ["VPI_PACKAGE_TARGET"]).resolve(strict=True)
    _ambient_packages = [
        str(Path(item).resolve()) for item in sys.path
        if item and ("site-packages" in item or "dist-packages" in item)
        and not Path(item).resolve().is_relative_to(_dependency_target)
    ]
    if _ambient_packages:
        raise RuntimeError("Isolated CPU child can see ambient site packages")
    import numpy
    import pandas as pd
    import scipy
    import matplotlib
    import PIL
    for _module in (numpy, pd, scipy, matplotlib, PIL):
        if not Path(_module.__file__).resolve().is_relative_to(_dependency_target):
            raise RuntimeError("Pinned CPU dependency resolved outside the private target")
    _PHASE.update(status="runtime_imports_verified", numpy=str(numpy.__version__),
                  pandas=str(pd.__version__), scipy=str(scipy.__version__))
    {_phase_write()}
    _PHASE.update(status="science_running", scientific_run_started=True)
    {_phase_write()}
{science}
    _PHASE.update(status="science_complete")
    {_phase_write()}
    from current_launch import complete_job
    JOB_COMPLETION = complete_job(
        LAUNCH_PLAN, _JOB_ID, CODE_PROVENANCE["snapshot_sha256"])
    print(JOB_COMPLETION, flush=True)
    _PHASE.update(status="complete", tracking_status={('pending_separate_publication' if job['id'] in PUBLICATION_JOBS else 'not_applicable')!r})
    {_phase_write()}
except BaseException as _child_error:
    _PHASE.update(status="failed", error_type=type(_child_error).__name__,
                  scientific_results_preserved=True)
    {_phase_write()}
    raise
'''
    ast.parse(program)
    assert program.index("science_running") < program.index("science_complete")
    assert program.index("science_complete") < program.index("complete_job(")
    if job["id"] in PUBLICATION_JOBS:
        assert "publish_completed_job(" not in program
        assert "publish_next_stage.py" not in program
    return program


def _wrapper_cell(existing, plan, job, child):
    source = _source(existing)
    marker = "    from current_launch import configure_runtime\n"
    if source.count(marker) != 1 or source.count("except BaseException as _bootstrap_error:") != 1:
        raise ValueError(f"Unexpected generated bootstrap cell for {job['id']}")
    prefix = source.split(marker, 1)[0]
    figure_display = ""
    if job["id"] in EVALUATION_JOBS:
        figure_display = '''    from IPython.display import display as _display, Image as _Image
    _report_root = Path(json.loads((Path("/kaggle/working/launch-config") / (_JOB_ID + ".json")).read_text())["report_root"])
    for _figure in sorted((_report_root / "shareable" / "figures").glob("*.png")):
        _display(_Image(filename=str(_figure)))
'''
    plan_json = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    body = f'''    from current_launch import configure_runtime, resolve_job
    LAUNCH_PLAN = json.loads({plan_json!r})
    _JOB_ID = {job['id']!r}
    configure_runtime(LAUNCH_PLAN, _JOB_ID, CODE_ROOT)
    LAUNCH_BINDING = resolve_job(LAUNCH_PLAN, _JOB_ID)
    import subprocess
    import uuid
    _profile_relative = "requirements/current-evaluation-runtime.json"
    _attempt = uuid.uuid4().hex[:12]
    _target = _runtime.runtime_target("/tmp/vpi-runtime-" + _JOB_ID + "-" + _attempt)
    os.environ["VPI_PACKAGE_TARGET"] = str(_target)
    _isolated_profile = Path("/kaggle/working/runtime-profiles") / (_JOB_ID + "-" + _attempt)
    _prepare_command = [sys.executable, str(CODE_ROOT / "scripts/prepare_runtime_environment.py"),
                        "--profile", str(CODE_ROOT / _profile_relative),
                        "--wheelhouse", os.environ["VPI_WHEELHOUSE"],
                        "--target", str(_target), "--output-root", str(_isolated_profile)]
    subprocess.run(_prepare_command, check=True, timeout=1020)
    _base_profile = json.loads((CODE_ROOT / _profile_relative).read_text())
    _base_contract = _base_profile.get("runtime_contract", _base_profile)
    _runtime.observed_contract(_isolated_profile, CODE_ROOT / _profile_relative, _base_contract)
    _dependency_receipt = json.loads((_target / "dependency_receipt.json").read_text())
    if _dependency_receipt.get("runtime_contract") != _base_contract:
        raise ValueError("Fresh isolated dependency target differs from the registered profile")
    _plan_path = Path("/tmp") / ("vpi-current-plan-" + _JOB_ID + "-" + _attempt + ".json")
    _plan_path.write_text(json.dumps(LAUNCH_PLAN, sort_keys=True, separators=(",", ":")))
    _child_path = Path("/tmp") / ("vpi-current-cpu-" + _JOB_ID + "-" + _attempt + ".py")
    _child_path.write_text({child!r})
    _environment = dict(os.environ)
    _environment["VPI_RESOLVED_CODE_ROOT"] = str(CODE_ROOT)
    _environment["VPI_LAUNCH_PLAN_PATH"] = str(_plan_path)
    _environment["PYTHONUNBUFFERED"] = "1"
    _environment["PYTHONDONTWRITEBYTECODE"] = "1"
    _environment["MPLBACKEND"] = "Agg"
    _environment["MPLCONFIGDIR"] = "/tmp/vpi-matplotlib-" + _attempt
    _environment["PYTHONPATH"] = os.pathsep.join(
        [str(_target), str(CODE_ROOT / "scripts"), str(CODE_ROOT / "src")])
    _child_invoked = True
    _completed = subprocess.run(
        [sys.executable, "-S", "-B", str(_child_path)], env=_environment,
        check=True, timeout={CHILD_TIMEOUT_SECONDS})
{figure_display}    _completion_path = Path("/kaggle/working/job_completion.json")
    _completion = json.loads(_completion_path.read_text())
    if (_completion.get("schema") != "vpi-current-job-completion/v1"
            or _completion.get("job_id") != _JOB_ID
            or _completion.get("status") != "outputs_complete"):
        raise ValueError("Isolated CPU child did not write this job's completion receipt")
    print(json.dumps({{"launch_binding": LAUNCH_BINDING,
                      "child_returncode": _completed.returncode,
                      "completion": _completion}}, sort_keys=True, default=str))
'''
    failure = f'''except BaseException as _bootstrap_error:
    _failure_root = Path('/kaggle/working/runtime-receipts') if Path('/kaggle').is_dir() else Path.cwd()/'.local/runtime-receipts'
    _failure_root.mkdir(parents=True, exist_ok=True)
    _phase_path = _failure_root / ({job['id']!r} + '-isolated-child.json')
    try:
        _phase = json.loads(_phase_path.read_text()) if _phase_path.is_file() else {{}}
    except Exception:
        _phase = {{}}
    _child_invoked_value = bool(locals().get('_child_invoked', False))
    _failure = {{'status': 'isolated_child_failed' if _child_invoked_value else 'bootstrap_failed',
                'error_type': _phase.get('error_type', type(_bootstrap_error).__name__),
                'scientific_run_started': _phase.get('scientific_run_started', False),
                'scientific_results_preserved': _phase.get('scientific_results_preserved', _child_invoked_value)}}
    _temporary = _failure_root/'bootstrap-failure.json.partial'
    _temporary.write_text(json.dumps(_failure, sort_keys=True)+'\\n')
    os.replace(_temporary, _failure_root/'bootstrap-failure.json')
    raise
'''
    wrapper = prefix + body + failure
    ast.parse(wrapper)
    return wrapper


def patch_cpu_notebook(path, source_root, plan, job, scientific_cells=None):
    path = Path(path)
    notebook = json.loads(path.read_text())
    positions = [
        index for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code" and "BOOTSTRAP = _runtime.bootstrap" in _source(cell)
    ]
    if len(positions) != 1:
        raise ValueError(f"Generated CPU notebook {job['id']} must have one bootstrap")
    combined = "\n".join(_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    if "JOB_COMPLETION = complete_job(" not in combined:
        raise ValueError(f"Generated CPU notebook {job['id']} has no completion gate")
    if bool(job.get("config", {}).get("publish_tracking")) != ("publish_completed_job(" in combined):
        raise ValueError(f"Generated CPU notebook {job['id']} changed publication policy")
    cells = scientific_cells or _scientific_cells(source_root, job)
    child = build_cpu_child(plan, job, cells)
    index = positions[0]
    wrapper = copy.deepcopy(notebook["cells"][index])
    wrapper["source"] = _wrapper_cell(wrapper, plan, job, child)
    wrapper["outputs"] = []
    wrapper["execution_count"] = None
    # Even official notebooks with a small pre-bootstrap stage-constant cell
    # must execute all Python in the isolated child, never in Kaggle's kernel.
    prefix_cells = [
        cell for cell in notebook["cells"][:index]
        if cell.get("cell_type") != "code"
    ]
    notebook["cells"] = [*prefix_cells, wrapper]
    parsed = nbformat.from_dict(notebook)
    nbformat.validate(parsed)
    for cell in parsed.cells:
        if cell.cell_type == "code":
            compile(cell.source, path.name, "exec")
            if cell.outputs or cell.execution_count is not None:
                raise ValueError("Isolated CPU queue notebooks must remain clean")
    encoded = (json.dumps(notebook, indent=1) + "\n").encode()
    if len(encoded) >= 1024 ** 2:
        raise ValueError("Isolated CPU notebook exceeds Kaggle's reviewed 1 MiB bound")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return {"job_id": job["id"], "notebook_sha256": hashlib.sha256(encoded).hexdigest(),
            "notebook_size_bytes": len(encoded)}


def build_queue(source_queue, output_queue, source_root, retry_job=None):
    source_queue, output_queue, source_root = map(Path, (source_queue, output_queue, source_root))
    if output_queue.exists():
        raise FileExistsError("Choose a fresh output queue")
    sys.path.insert(0, str(source_root / "scripts"))
    import current_kaggle_queue as queue
    from current_notebook_runtime import atomic_json, verify_source

    source = verify_source(source_root)
    temporary = output_queue.with_name(output_queue.name + ".incomplete-" + uuid.uuid4().hex)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with queue._queue_lock(source_queue):
            _, plan, state, _ = queue.open_queue(source_queue)
            jobs = _cpu_science_jobs(plan)
            if not jobs:
                raise ValueError("Launch plan has no fixed CPU science jobs")
            target_ids = {job["id"] for job in jobs}
            tracked = {name: copy.deepcopy(row) for name, row in state["jobs"].items()
                       if name in target_ids}
            active = {name: row.get("status") for name, row in tracked.items()
                      if row.get("status") not in {"failed", queue.COMPLETE}}
            if active:
                raise ValueError("Cannot patch active CPU jobs: " + json.dumps(active, sort_keys=True))
            failed = {name: row for name, row in tracked.items() if row.get("status") == "failed"}
            if retry_job is not None and retry_job not in failed:
                raise ValueError("Requested retry job is not a terminally failed CPU science job")
            if state.get("source_snapshot_sha256") != source["snapshot_sha256"]:
                raise ValueError("Queue and prepared source snapshot differ")
            shutil.copytree(source_queue, temporary, ignore=shutil.ignore_patterns(".queue-state.lock"))
        for name in failed:
            state["jobs"].pop(name)
        pending = [job for job in jobs if job["id"] not in state["jobs"]]
        if not pending:
            raise ValueError("No uncompleted CPU science jobs require isolation")
        patched = []
        for job in pending:
            target = temporary / job["id"] / Path(job["notebook"]).name
            patched.append(patch_cpu_notebook(
                target, source_root, plan, job, _scientific_cells(source_root, job)))
        state["prepared_jobs"] = {
            job["id"]: queue._job_files(temporary, plan, job) for job in plan["jobs"]
        }
        atomic_json(temporary / "queue-state.json", state)
        queue.open_queue(temporary)
        receipt = {
            "schema": SCHEMA,
            "source_queue": str(source_queue),
            "source_snapshot_sha256": source["snapshot_sha256"],
            "plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
            "tracked_jobs_preserved": sorted(state["jobs"]),
            "completed_cpu_jobs_preserved": sorted(
                name for name, row in tracked.items() if row.get("status") == queue.COMPLETE),
            "superseded_failed_jobs": failed,
            "patched_job_count": len(patched),
            "patched_jobs": patched,
            "tracking_jobs_pending": sorted(
                job["id"] for job in pending if job["id"] in PUBLICATION_JOBS),
            "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
            "patched_jobs_submitted": False,
            "kaggle_commands_run": False,
        }
        atomic_json(temporary.parent / (temporary.name + "-receipt.json"), receipt)
        os.replace(temporary, output_queue)
        atomic_json(output_queue.parent / "isolated-cpu-queue-receipt.json", receipt)
        return receipt
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--retry-job", default=None,
                        help="Optional terminally failed CPU job that must be present in the source state")
    args = parser.parse_args()
    print(json.dumps(build_queue(args.source_queue, args.output_queue, args.source_root, args.retry_job),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
