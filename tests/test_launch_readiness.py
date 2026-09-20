"""Read-only launch-readiness policy tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_launch_readiness import audit_queue


def _write_job(root, job, source, *, internet=False, gpu=False):
    folder = root / job["id"]
    folder.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbformat.write(notebook, folder / Path(job["notebook"]).name)
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "enable_internet": internet,
        "enable_gpu": gpu,
        "dataset_sources": [],
        "kernel_sources": [],
    }))


def _fixture(tmp_path):
    jobs = [
        {"id": "cpu", "runtime": "cpu", "notebook": "cpu.ipynb", "config": {"action": "predict"}},
        {"id": "cache", "runtime": "dino-cpu", "notebook": "cache.ipynb", "config": {"action": "prepare_shard"}},
        {"id": "regions", "runtime": "regions-gpu", "notebook": "regions.ipynb", "config": {"action": "extract"}},
        {"id": "evaluate", "runtime": "cpu", "notebook": "evaluate.ipynb", "config": {"action": "evaluate", "publish_tracking": True}},
    ]
    root = tmp_path / "queue"
    root.mkdir()
    plan = {"jobs": jobs, "resources": {}}
    (root / "launch-plan.json").write_text(json.dumps(plan))
    (root / "queue-state.json").write_text(json.dumps({"jobs": {}}))
    _write_job(root, jobs[0], "value = 1")
    _write_job(root, jobs[1], 'SCHEMA="vpi-isolated-cache-child/v1"\nargs=[sys.executable, "-S", "-B"]\nruntime_imports_verified=True')
    _write_job(root, jobs[2], 'SCHEMA="vpi-isolated-region-child/v1"\nargs=[sys.executable, "-S", "-B"]', gpu=True)
    _write_job(root, jobs[3], "publish_completed_job(plan, job, root)")
    return root


def test_known_runtime_hazards_fail_closed(tmp_path):
    report = audit_queue(_fixture(tmp_path))
    codes = {(item["job_id"], item["code"]) for item in report["blockers"]}
    assert report["status"] == "blocked"
    assert ("cpu", "cpu_not_clean_child") in codes
    assert ("regions", "region_sam_code_preflight_missing") in codes
    assert ("evaluate", "tracking_inline_without_internet") in codes
    assert ("cache", "dino_cache_not_clean_child") not in codes


def test_completed_jobs_do_not_block_a_recovery_queue(tmp_path):
    root = _fixture(tmp_path)
    state = {"jobs": {job: {"status": "kaggle_complete_with_receipt"}
                       for job in ("cpu", "cache", "regions", "evaluate")}}
    (root / "queue-state.json").write_text(json.dumps(state))
    report = audit_queue(root)
    assert report["status"] == "ready"
    assert not report["blockers"]


def test_explicit_sam_resource_and_preflight_make_region_ready(tmp_path):
    root = _fixture(tmp_path)
    plan = json.loads((root / "launch-plan.json").read_text())
    revision = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
    plan["resources"]["sam2-code"] = {
        "kind": "dataset", "ref": "owner/sam2-code", "marker": "sam2_code_receipt.json", "suffixes": [""]}
    plan["runtime_readiness"] = {"sam2_code": {
        "resource": "sam2-code", "distribution": "SAM-2",
        "repository": "https://github.com/facebookresearch/sam2", "revision": revision,
        "preflight_marker": "verify_region_model_runtime",
    }}
    region = next(job for job in plan["jobs"] if job["id"] == "regions")
    region["config"]["sam_code_root"] = {"$resource": "sam2-code"}
    (root / "launch-plan.json").write_text(json.dumps(plan))
    path = root / "regions" / "regions.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.cells[0].source += "\nverify_region_model_runtime()"
    nbformat.write(notebook, path)
    report = audit_queue(root)
    codes = {(item["job_id"], item["code"]) for item in report["blockers"]}
    assert ("regions", "region_sam_code_preflight_missing") not in codes


def test_materialized_queue_is_fully_classified():
    queue = ROOT.parent / ".local" / "kaggle-launch-kit-v11" / "queue"
    if not queue.is_dir():
        return
    report = audit_queue(queue)
    assert report["job_count"] == 152
    assert sum(row["jobs"] for row in report["families"].values()) == 152
