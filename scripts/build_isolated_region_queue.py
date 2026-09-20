#!/usr/bin/env python3
"""Retry fixed LocateAnything/SAM jobs in clean pinned GPU children.

Kaggle imports NumPy in the notebook process before the first user cell.  The
sealed region wheelhouse therefore cannot replace that process safely.  This
builder copies a prepared queue, replaces every uncompleted region-extraction
notebook with a one-cell wrapper, installs the unchanged GPU runtime in a fresh
private target, and runs the unchanged region science in ``python -S``.

Only terminally failed region jobs are removed from queue state.  Their records
remain in the builder receipt.  Completed jobs and unrelated queue state are
preserved byte-for-byte.  This script never calls Kaggle.
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


NATIVE_REGION_NOTEBOOK = "notebooks/input-preparation/current-native-region-inference.ipynb"
SCHEMA = "vpi-isolated-region-queue/v1"
CHILD_TIMEOUT_SECONDS = 39_600


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _region_jobs(plan):
    return [
        job for job in plan["jobs"]
        if job.get("notebook") == NATIVE_REGION_NOTEBOOK
        and job.get("runtime") == "regions-gpu"
        and job.get("config", {}).get("action") == "extract"
    ]


def _scientific_cells(source_root):
    path = Path(source_root) / NATIVE_REGION_NOTEBOOK
    notebook = json.loads(path.read_text())
    positions = [
        index for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
        and "BOOTSTRAP = _runtime.bootstrap" in _source(cell)
    ]
    if len(positions) != 1:
        raise ValueError("Canonical native-region notebook must have one verified bootstrap")
    cells = [
        _source(cell) for cell in notebook["cells"][positions[0] + 1:]
        if cell.get("cell_type") == "code"
    ]
    if not cells:
        raise ValueError("Canonical native-region notebook has no scientific cell")
    joined = "\n".join(cells)
    if any(marker in joined for marker in ("complete_job(", "LAUNCH_PLAN", "configure_runtime(")):
        raise ValueError("Canonical region science unexpectedly contains launch control")
    for source in cells:
        ast.parse(source)
    return cells


def _phase_write():
    return "_runtime.atomic_json(_PHASE_PATH, _PHASE)"


def build_region_child(job, scientific_cells):
    """Return unchanged region science plus completion gates for a clean child."""
    science = textwrap.indent("\n\n".join(scientific_cells), "    ")
    program = f'''from pathlib import Path
import importlib
import json
import os
import sys

CODE_ROOT = Path(os.environ["VPI_RESOLVED_CODE_ROOT"]).resolve(strict=True)
NOTEBOOK_RELATIVE = {NATIVE_REGION_NOTEBOOK!r}
RUNTIME_PROFILE = "requirements/profiles/linux-regions-cuda126.json"
sys.path.insert(0, str(CODE_ROOT / "scripts"))
sys.path.insert(0, str(CODE_ROOT / "src"))
import current_notebook_runtime as _runtime
BOOTSTRAP = _runtime.bootstrap(CODE_ROOT, NOTEBOOK_RELATIVE, runtime_profile=RUNTIME_PROFILE)
CODE_PROVENANCE = BOOTSTRAP["provenance"]
ROOT = CODE_ROOT

def display(value):
    print(json.dumps(value, sort_keys=True, default=str), flush=True)

LAUNCH_PLAN = json.loads(Path(os.environ["VPI_LAUNCH_PLAN_PATH"]).read_text())
_JOB_ID = {job['id']!r}
_PHASE_PATH = Path("/kaggle/working/runtime-receipts") / (_JOB_ID + "-isolated-child.json")
_PHASE = {{"schema": "vpi-isolated-region-child/v1", "job_id": _JOB_ID,
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
        raise RuntimeError("Isolated region child can see ambient site packages")
    import numpy
    import pandas
    import PIL
    import torch
    import torchvision
    import transformers
    for _module in (numpy, pandas, PIL, torch, torchvision, transformers):
        if not Path(_module.__file__).resolve().is_relative_to(_dependency_target):
            raise RuntimeError("Pinned region dependency resolved outside the private target")
    if not torch.cuda.is_available():
        raise RuntimeError("Pinned region child has no CUDA device")
    _PHASE.update(status="runtime_imports_verified", torch=str(torch.__version__),
                  torchvision=str(torchvision.__version__), transformers=str(transformers.__version__),
                  cuda_device=str(torch.cuda.get_device_name(0)))
    {_phase_write()}
    print({{"status": "runtime_imports_verified", "torch": str(torch.__version__),
           "torchvision": str(torchvision.__version__), "transformers": str(transformers.__version__),
           "cuda_device": str(torch.cuda.get_device_name(0))}}, flush=True)
    # Fail before LocateAnything if the separately built SAM source wheel, imports,
    # packaged Hydra config or checkpoint bytes differ from the fixed method.
    _region_config = _runtime.resolve_config("native-regions", CODE_ROOT)
    from visual_peer_inspector.data.model_safe_package import file_sha256 as _file_sha256
    from visual_peer_inspector.models.sam2 import (
        Sam2Configuration as _Sam2Configuration,
        verify_installed_sam2_revision as _verify_installed_sam2_revision,
    )
    _sam_configuration = _Sam2Configuration()
    _sam_provenance = _verify_installed_sam2_revision(_sam_configuration.code_revision)
    if (_sam_provenance.get("distribution") != "SAM-2"
            or _sam_provenance.get("version") != "1.0"):
        raise RuntimeError("Pinned SAM-2 distribution name or version differs")
    import sam2.build_sam as _sam_build_module
    import sam2.sam2_image_predictor as _sam_predictor_module
    if (not callable(getattr(_sam_build_module, "build_sam2", None))
            or not isinstance(getattr(_sam_predictor_module, "SAM2ImagePredictor", None), type)):
        raise RuntimeError("Pinned SAM-2 inference imports are incomplete")
    for _sam_module in (_sam_build_module, _sam_predictor_module):
        if not Path(_sam_module.__file__).resolve().is_relative_to(_dependency_target):
            raise RuntimeError("Pinned SAM-2 import resolved outside the private target")
    _sam_package_root = Path(_sam_build_module.__file__).resolve().parent
    _sam_config_path = _sam_package_root / _sam_configuration.model_config
    if (not _sam_config_path.is_file() or _sam_config_path.is_symlink()
            or not _sam_config_path.resolve().is_relative_to(_dependency_target)):
        raise RuntimeError("Pinned SAM-2 package omits its fixed model configuration")
    _sam_checkpoint = Path(_region_config["sam_checkpoint"]).resolve(strict=True)
    if (_sam_checkpoint.is_symlink()
            or _sam_checkpoint.stat().st_size != _sam_configuration.checkpoint_size_bytes
            or _file_sha256(_sam_checkpoint) != _sam_configuration.checkpoint_sha256):
        raise RuntimeError("Pinned SAM-2 checkpoint bytes differ")
    _PHASE.update(status="sam2_preflight_verified", sam2_distribution=_sam_provenance,
                  sam2_config=str(_sam_configuration.model_config),
                  sam2_checkpoint_sha256=_sam_configuration.checkpoint_sha256)
    {_phase_write()}
    print({{"status": "sam2_preflight_verified", "sam2_distribution": _sam_provenance,
           "sam2_config": _sam_configuration.model_config,
           "sam2_checkpoint_sha256": _sam_configuration.checkpoint_sha256}}, flush=True)
    _recovery_fields = {{"proposal_recovery_root", "proposal_recovery_lock_sha256",
                        "proposal_recovery_binding_sha256"}}
    _present_recovery = _recovery_fields & set(_region_config)
    if _present_recovery and _present_recovery != _recovery_fields:
        raise ValueError("Proposal recovery configuration is incomplete")
    if _present_recovery:
        from region_proposal_recovery import expected_binding as _expected_binding, seed_recovery as _seed_recovery
        from vpi_object_bank.storage import digest as _binding_digest
        _expected_recovery_binding = _expected_binding(_region_config, CODE_ROOT)
        if _binding_digest(_expected_recovery_binding) != _region_config["proposal_recovery_binding_sha256"]:
            raise ValueError("Proposal recovery expected binding differs from the continuation plan")
        _recovery_receipt = _seed_recovery(
            _region_config["proposal_recovery_root"], _region_config["output_root"],
            _region_config["proposal_recovery_lock_sha256"], _expected_recovery_binding)
        _PHASE.update(status="proposal_recovery_seeded", proposal_recovery=_recovery_receipt)
        {_phase_write()}
        print({{"status": "proposal_recovery_seeded", "receipt": _recovery_receipt}}, flush=True)
    _PHASE.update(status="science_running", scientific_run_started=True)
    {_phase_write()}
{science}
    _PHASE.update(status="science_complete")
    {_phase_write()}
    from current_launch import complete_job
    JOB_COMPLETION = complete_job(
        LAUNCH_PLAN, _JOB_ID, CODE_PROVENANCE["snapshot_sha256"])
    print(JOB_COMPLETION, flush=True)
    _PHASE.update(status="complete")
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
    return program


def _wrapper_cell(existing, plan, job, child):
    source = _source(existing)
    marker = "    from current_launch import configure_runtime\n"
    if source.count(marker) != 1 or source.count("except BaseException as _bootstrap_error:") != 1:
        raise ValueError(f"Unexpected generated bootstrap cell for {job['id']}")
    prefix = source.split(marker, 1)[0]
    plan_json = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    body = f'''    from current_launch import configure_runtime, resolve_job
    LAUNCH_PLAN = json.loads({plan_json!r})
    _JOB_ID = {job['id']!r}
    configure_runtime(LAUNCH_PLAN, _JOB_ID, CODE_ROOT)
    LAUNCH_BINDING = resolve_job(LAUNCH_PLAN, _JOB_ID)
    import subprocess
    import uuid
    _profile_relative = "requirements/profiles/linux-regions-cuda126.json"
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
        raise ValueError("Fresh isolated dependency target differs from the registered region profile")
    _plan_path = Path("/tmp") / ("vpi-current-plan-" + _JOB_ID + "-" + _attempt + ".json")
    _plan_path.write_text(json.dumps(LAUNCH_PLAN, sort_keys=True, separators=(",", ":")))
    _child_path = Path("/tmp") / ("vpi-current-regions-" + _JOB_ID + "-" + _attempt + ".py")
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
    _completion_path = Path("/kaggle/working/job_completion.json")
    _completion = json.loads(_completion_path.read_text())
    if (_completion.get("schema") != "vpi-current-job-completion/v1"
            or _completion.get("job_id") != _JOB_ID
            or _completion.get("status") != "outputs_complete"):
        raise ValueError("Isolated region child did not write this job's completion receipt")
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


def patch_region_notebook(path, source_root, plan, job, scientific_cells=None):
    path = Path(path)
    notebook = json.loads(path.read_text())
    positions = [
        index for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
        and "BOOTSTRAP = _runtime.bootstrap" in _source(cell)
    ]
    if len(positions) != 1:
        raise ValueError(f"Generated region notebook {job['id']} must have one bootstrap")
    combined = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    if "JOB_COMPLETION = complete_job(" not in combined:
        raise ValueError(f"Generated region notebook {job['id']} has no completion gate")
    cells = scientific_cells or _scientific_cells(source_root)
    child = build_region_child(job, cells)
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
                raise ValueError("Isolated region queue notebooks must remain clean")
    encoded = (json.dumps(notebook, indent=1) + "\n").encode()
    if len(encoded) >= 1024 ** 2:
        raise ValueError("Isolated region notebook exceeds the reviewed 1 MiB bound")
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
            jobs = _region_jobs(plan)
            if not jobs:
                raise ValueError("Launch plan has no fixed GPU region jobs")
            target_ids = {job["id"] for job in jobs}
            tracked = {name: copy.deepcopy(row) for name, row in state["jobs"].items()
                       if name in target_ids}
            invalid = {name: row.get("status") for name, row in tracked.items()
                       if row.get("status") != "failed"}
            if invalid:
                raise ValueError("Cannot patch non-failed tracked region jobs: " + json.dumps(invalid, sort_keys=True))
            if state.get("source_snapshot_sha256") != source["snapshot_sha256"]:
                raise ValueError("Queue and prepared source snapshot differ")
            shutil.copytree(source_queue, temporary, ignore=shutil.ignore_patterns(".queue-state.lock"))
        for name in tracked:
            state["jobs"].pop(name)
        scientific = _scientific_cells(source_root)
        patched = []
        for job in jobs:
            target = temporary / job["id"] / Path(job["notebook"]).name
            patched.append(patch_region_notebook(target, source_root, plan, job, scientific))
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
            "superseded_failed_jobs": tracked,
            "patched_job_count": len(patched),
            "patched_jobs": patched,
            "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
            "patched_jobs_submitted": False,
            "kaggle_commands_run": False,
        }
        atomic_json(temporary.parent / (temporary.name + "-receipt.json"), receipt)
        os.replace(temporary, output_queue)
        atomic_json(output_queue.parent / "isolated-region-queue-receipt.json", receipt)
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
    print(json.dumps(build_queue(args.source_queue, args.output_queue, args.source_root),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
