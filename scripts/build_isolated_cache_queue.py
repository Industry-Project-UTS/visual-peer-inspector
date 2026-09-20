#!/usr/bin/env python3
"""Copy a prepared queue and isolate compact DINO cache jobs in clean children.

Kaggle imports NumPy in the notebook kernel before the first user cell.  The
registered wheelhouse therefore cannot replace NumPy safely in that process.
This local-only queue builder keeps launch binding in the notebook process,
installs the sealed runtime into a fresh /tmp target through a clean helper,
then runs the cache science, retention, and completion gates in a clean child.
The child starts with ``-S`` so Kaggle's ambient site packages cannot mix with
the exact private target (for example, system torchvision with private torch).

The input queue is read under its queue lock and is never modified.  The output
queue must be new.  No Kaggle command is executed by this script.
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


NATIVE_CACHE_NOTEBOOK = "notebooks/input-preparation/current-native-regions-to-object-cache.ipynb"
SCHEMA = "vpi-isolated-cache-queue/v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _compact_cache_jobs(plan):
    return [
        job
        for job in plan["jobs"]
        if job.get("notebook") == NATIVE_CACHE_NOTEBOOK
        and job.get("compact_checkpoints") is True
        and job.get("config", {}).get("action") == "prepare_shard"
    ]


def _scientific_cells(source_root):
    path = Path(source_root) / NATIVE_CACHE_NOTEBOOK
    notebook = json.loads(path.read_text())
    positions = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code" and "BOOTSTRAP = _runtime.bootstrap" in _source(cell)
    ]
    if len(positions) != 1:
        raise ValueError("Canonical native-cache notebook must have one verified bootstrap")
    cells = [
        _source(cell)
        for cell in notebook["cells"][positions[0] + 1 :]
        if cell.get("cell_type") == "code"
    ]
    if not cells:
        raise ValueError("Canonical native-cache notebook has no scientific cells")
    joined = "\n".join(cells)
    forbidden = ("begin_cache_job(", "retain_completed_cache(", "complete_job(", "LAUNCH_PLAN")
    if any(marker in joined for marker in forbidden):
        raise ValueError("Canonical science unexpectedly contains generated launch control")
    for source in cells:
        ast.parse(source)
    return cells


def _phase_write():
    return "_runtime.atomic_json(_PHASE_PATH, _PHASE)"


def build_cache_child(plan, job, scientific_cells):
    """Return the clean child program, with checkpoint gates in original order."""
    science = "\n\n".join(scientific_cells)
    indented_science = textwrap.indent(science, "        ")
    program = f'''from pathlib import Path
import json
import os
import sys
import importlib.util

CODE_ROOT = Path(os.environ["VPI_RESOLVED_CODE_ROOT"]).resolve(strict=True)
NOTEBOOK_RELATIVE = {NATIVE_CACHE_NOTEBOOK!r}
RUNTIME_PROFILE = None
sys.path.insert(0, str(CODE_ROOT / "scripts"))
sys.path.insert(0, str(CODE_ROOT / "src"))
import current_notebook_runtime as _runtime
BOOTSTRAP = _runtime.bootstrap(CODE_ROOT, NOTEBOOK_RELATIVE, runtime_profile=RUNTIME_PROFILE)
CODE_PROVENANCE = BOOTSTRAP["provenance"]
ROOT = CODE_ROOT

def display(value):
    print(json.dumps(value, sort_keys=True, default=str))

LAUNCH_PLAN = {plan!r}
_JOB_ID = {job['id']!r}
_PHASE_PATH = Path("/kaggle/working/runtime-receipts") / (_JOB_ID + "-isolated-child.json")
_PHASE = {{"schema": "vpi-isolated-cache-child/v1", "job_id": _JOB_ID,
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
        raise RuntimeError("Isolated DINO child can see ambient site packages")
    if importlib.util.find_spec("torchvision") is not None:
        raise RuntimeError("Unpinned torchvision is visible in the isolated DINO child")
    import torch
    import transformers
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
    for _module in (torch, transformers):
        if not Path(_module.__file__).resolve().is_relative_to(_dependency_target):
            raise RuntimeError("Pinned DINO dependency resolved outside the private target")
    _PHASE.update(status="runtime_imports_verified", torch=str(torch.__version__),
                  transformers=str(transformers.__version__), torchvision_visible=False)
    {_phase_write()}
    print({{"status": "runtime_imports_verified", "torch": str(torch.__version__),
           "transformers": str(transformers.__version__), "torchvision_visible": False}}, flush=True)
    from post_completion_retention import begin_cache_job
    _LAUNCH_SCIENCE_REUSED = begin_cache_job(
        LAUNCH_PLAN, _JOB_ID, CODE_PROVENANCE["snapshot_sha256"])
    _PHASE.update(status="reuse_verified" if _LAUNCH_SCIENCE_REUSED else "science_ready",
                  completed_science_reused=bool(_LAUNCH_SCIENCE_REUSED))
    {_phase_write()}
    print({{"verified_completed_science_reused": _LAUNCH_SCIENCE_REUSED}})
    if not _LAUNCH_SCIENCE_REUSED:
        _PHASE.update(status="science_running", scientific_run_started=True)
        {_phase_write()}
{indented_science}
        _PHASE.update(status="science_complete")
        {_phase_write()}
    from post_completion_retention import retain_completed_cache
    LAUNCH_RETENTION = retain_completed_cache(
        LAUNCH_PLAN, _JOB_ID, CODE_PROVENANCE["snapshot_sha256"])
    display(LAUNCH_RETENTION)
    _PHASE.update(status="retention_complete")
    {_phase_write()}
    from current_launch import complete_job
    JOB_COMPLETION = complete_job(
        LAUNCH_PLAN, _JOB_ID, CODE_PROVENANCE["snapshot_sha256"])
    print(JOB_COMPLETION)
    _PHASE.update(status="complete")
    {_phase_write()}
except BaseException as _child_error:
    _PHASE.update(status="failed", error_type=type(_child_error).__name__,
                  scientific_results_preserved=True)
    {_phase_write()}
    raise
'''
    ast.parse(program)
    required = [
        "begin_cache_job(",
        "if not _LAUNCH_SCIENCE_REUSED:",
        "retain_completed_cache(",
        "complete_job(",
    ]
    offsets = [program.index(marker) for marker in required]
    if offsets != sorted(offsets):
        raise AssertionError("Cache reuse, science, retention, and completion order changed")
    return program


def _wrapper_cell(existing, plan, job, child):
    source = _source(existing)
    marker = "    from current_launch import configure_runtime\n"
    if source.count(marker) != 1 or source.count("except BaseException as _bootstrap_error:") != 1:
        raise ValueError(f"Unexpected generated bootstrap cell for {job['id']}")
    prefix = source.split(marker, 1)[0]
    body = f'''    from current_launch import configure_runtime, resolve_job
    LAUNCH_PLAN = {plan!r}
    configure_runtime(LAUNCH_PLAN, {job['id']!r}, CODE_ROOT)
    LAUNCH_BINDING = resolve_job(LAUNCH_PLAN, {job['id']!r})
    import subprocess
    import uuid
    _profile_relative = os.environ.get("VPI_BASE_RUNTIME_PROFILE") or "requirements/current-evaluation-runtime.json"
    _attempt = uuid.uuid4().hex[:12]
    _target = _runtime.runtime_target("/tmp/vpi-runtime-" + {job['id']!r} + "-" + _attempt)
    os.environ["VPI_PACKAGE_TARGET"] = str(_target)
    _isolated_profile = Path("/kaggle/working/runtime-profiles") / ({job['id']!r} + "-" + _attempt)
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
    _child_path = Path("/tmp") / ("vpi-current-cache-" + {job['id']!r} + "-" + _attempt + ".py")
    _child_path.write_text({child!r})
    _environment = dict(os.environ)
    _environment["VPI_RESOLVED_CODE_ROOT"] = str(CODE_ROOT)
    _environment["PYTHONUNBUFFERED"] = "1"
    _environment["PYTHONDONTWRITEBYTECODE"] = "1"
    _environment["MPLBACKEND"] = "Agg"
    _environment["MPLCONFIGDIR"] = "/tmp/vpi-matplotlib-" + _attempt
    _environment["PYTHONPATH"] = os.pathsep.join(
        [str(_target), str(CODE_ROOT / "scripts"), str(CODE_ROOT / "src")])
    _child_invoked = True
    _completed = subprocess.run(
        [sys.executable, "-S", "-B", str(_child_path)], env=_environment,
        check=True, timeout=7200)
    _completion_path = Path("/kaggle/working/job_completion.json")
    _completion = json.loads(_completion_path.read_text())
    if (_completion.get("schema") != "vpi-current-job-completion/v1"
            or _completion.get("job_id") != {job['id']!r}
            or _completion.get("status") != "outputs_complete"):
        raise ValueError("Isolated cache child did not write this job's completion receipt")
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


def patch_cache_notebook(path, source_root, plan, job, scientific_cells=None):
    path = Path(path)
    notebook = json.loads(path.read_text())
    code = [(index, _source(cell)) for index, cell in enumerate(notebook["cells"])
            if cell.get("cell_type") == "code"]
    positions = [index for index, source in code if "BOOTSTRAP = _runtime.bootstrap" in source]
    if len(positions) != 1:
        raise ValueError(f"Generated cache notebook {job['id']} must have one bootstrap")
    combined = "\n".join(source for _, source in code)
    generated = [
        "begin_cache_job(", "if not _LAUNCH_SCIENCE_REUSED:",
        "retain_completed_cache(", "JOB_COMPLETION = complete_job(",
    ]
    offsets = [combined.index(marker) for marker in generated]
    if offsets != sorted(offsets):
        raise ValueError(f"Generated checkpoint order changed for {job['id']}")
    scientific_cells = scientific_cells or _scientific_cells(source_root)
    child = build_cache_child(plan, job, scientific_cells)
    index = positions[0]
    wrapper = copy.deepcopy(notebook["cells"][index])
    wrapper["source"] = _wrapper_cell(wrapper, plan, job, child)
    wrapper["outputs"] = []
    wrapper["execution_count"] = None
    notebook["cells"] = [*notebook["cells"][:index], wrapper]
    parsed = nbformat.from_dict(notebook)
    nbformat.validate(parsed)
    for cell in parsed.cells:
        if cell.cell_type == "code":
            compile(cell.source, path.name, "exec")
            if cell.outputs or cell.execution_count is not None:
                raise ValueError("Isolated queue notebooks must remain clean")
    encoded = (json.dumps(notebook, indent=1) + "\n").encode()
    if len(encoded) >= 1024 ** 2:
        raise ValueError("Isolated cache notebook exceeds Kaggle's reviewed 1 MiB bound")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return {"job_id": job["id"], "notebook_sha256": hashlib.sha256(encoded).hexdigest(),
            "notebook_size_bytes": len(encoded)}


def build_queue(source_queue, output_queue, source_root):
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
            jobs = _compact_cache_jobs(plan)
            if not jobs:
                raise ValueError("Launch plan has no compact native-cache shard jobs")
            started = sorted(job["id"] for job in jobs if job["id"] in state["jobs"])
            if started:
                raise ValueError("Cannot change already tracked cache jobs: " + ", ".join(started))
            if state.get("source_snapshot_sha256") != source["snapshot_sha256"]:
                raise ValueError("Queue and prepared source snapshot differ")
            shutil.copytree(source_queue, temporary, ignore=shutil.ignore_patterns(".queue-state.lock"))
        scientific = _scientific_cells(source_root)
        patched = []
        for job in jobs:
            target = temporary / job["id"] / Path(job["notebook"]).name
            patched.append(patch_cache_notebook(target, source_root, plan, job, scientific))
        prepared = {job["id"]: queue._job_files(temporary, plan, job) for job in plan["jobs"]}
        state["prepared_jobs"] = prepared
        atomic_json(temporary / "queue-state.json", state)
        queue.open_queue(temporary)
        receipt = {
            "schema": SCHEMA,
            "source_queue": str(source_queue),
            "source_snapshot_sha256": source["snapshot_sha256"],
            "plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
            "tracked_jobs_preserved": sorted(state["jobs"]),
            "patched_job_count": len(patched),
            "patched_jobs": patched,
            "patched_jobs_submitted": False,
            "kaggle_commands_run": False,
        }
        atomic_json(temporary.parent / (temporary.name + "-receipt.json"), receipt)
        os.replace(temporary, output_queue)
        final_receipt = output_queue.parent / "isolated-cache-queue-receipt.json"
        atomic_json(final_receipt, receipt)
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
    args = parser.parse_args()
    receipt = build_queue(args.source_queue, args.output_queue, args.source_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
