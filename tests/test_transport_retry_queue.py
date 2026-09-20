"""Fail-closed tests for the transport-boundary retry queue."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_continuation_queue as continuation
import build_transport_retry_queue as retry
from current_notebook_runtime import atomic_json, canonical, verify_source


def _source(root, *, include_diagnostic, changed_science=False):
    files = {
        continuation.REGION_PROFILE: "same profile\n",
        continuation.NATIVE_REGIONS: "changed\n" if changed_science else "same science\n",
        "src/vpi_next_evaluation/contracts.py": "same contract\n",
        "scripts/current_notebook_runtime.py": "# fixture bootstrap\n",
        "requirements/current-evaluation-runtime.json": "{}\n",
    }
    if include_diagnostic:
        files[retry.REMOVED_LOCAL_DIAGNOSTIC] = "completed diagnostic\n"
    inventory = {}
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "schema": "vpi-current-notebook-source/v1",
        "source_state": "working_directory_snapshot",
        "files": inventory,
        "snapshot_sha256": hashlib.sha256(canonical(inventory)).hexdigest(),
    }
    atomic_json(root / "prepared_source_manifest.json", record)
    return record


def test_transport_transition_removes_only_local_gpu_diagnostic(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    _source(old, include_diagnostic=True)
    expected = _source(new, include_diagnostic=False)
    _, checked = retry._verify_source_transition(old, new, verify_source)
    assert checked == expected

    changed = tmp_path / "changed"; changed.mkdir()
    _source(changed, include_diagnostic=False, changed_science=True)
    with pytest.raises(ValueError, match="scientific source bytes"):
        retry._verify_source_transition(old, changed, verify_source)


def test_old_external_evidence_requires_exact_full_plan_projection():
    full = {"jobs": [{"id": "done"}, {"id": "pending"}]}
    old = {
        "jobs": [{"id": "pending"}],
        "external_completed_jobs": {"done": {
            "plan_sha256": "a" * 64,
            "source_snapshot_sha256": "b" * 64,
            "completion_sha256": "c" * 64,
            "outputs": {"resource": "d" * 64},
            "kernel_ref": "owner/done",
        }},
    }
    assert set(retry._old_external_evidence(old, full)) == {"done"}
    old["jobs"].append({"id": "extra"})
    with pytest.raises(ValueError, match="exact projection"):
        retry._old_external_evidence(old, full)


def test_retry_builder_has_no_kaggle_submission_command():
    source = (ROOT / "scripts/build_transport_retry_queue.py").read_text()
    assert "kaggle kernels push" not in source
    assert "subprocess" not in source
