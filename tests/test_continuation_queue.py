"""Fail-closed tests for the local-only continuation queue transformer."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_continuation_queue as continuation
from build_current_launch_plan import plan
from current_notebook_runtime import atomic_json, canonical, verify_source
from current_launch import validate_plan


def test_continuation_recovery_verifier_does_not_import_scientific_packages():
    source = (ROOT / "scripts/build_continuation_queue.py").read_text()
    assert "from vpi_object_bank.storage import" not in source
    assert "from visual_peer_inspector" not in source


def _source(root, *, sam2, native="unchanged native regions"):
    profile = {
        "runtime_contract": {"packages": {"SAM-2": "1.0"} if sam2 else {}},
        "source_packages": ({"SAM-2": {
            "version": "1.0",
            "repository": "https://github.com/facebookresearch/sam2",
            "revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        }} if sam2 else {}),
    }
    files = {
        continuation.REGION_PROFILE: json.dumps(profile, sort_keys=True),
        continuation.NATIVE_REGIONS: native,
        "scripts/current_notebook_runtime.py": "# fixture bootstrap\n",
        "requirements/current-evaluation-runtime.json": "{}\n",
    }
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


def test_source_transition_requires_exact_sam_pin_and_unchanged_science(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    _source(old, sam2=False); current = _source(new, sam2=True)
    checked_old, checked_new = continuation._verify_source_transition(old, new, verify_source)
    assert checked_new == current
    assert checked_old["snapshot_sha256"] != checked_new["snapshot_sha256"]

    changed = tmp_path / "changed"; changed.mkdir()
    _source(changed, sam2=True, native="changed science")
    with pytest.raises(ValueError, match="native-regions source change"):
        continuation._verify_source_transition(old, changed, verify_source)


def test_completed_setup_regions_is_retained_while_other_completions_leave_plan():
    value = plan()
    completed = {'setup-cpu', 'setup-regions'}
    evidence = {}
    for name in completed - continuation.RERUN_COMPLETED:
        resources = {
            key: hashlib.sha256(key.encode()).hexdigest()
            for key, spec in value['resources'].items()
            if spec.get('producer_job') == name
        }
        refs = {value['resources'][key]['ref'] for key in resources}
        evidence[name] = {
            'plan_sha256': 'a' * 64,
            'source_snapshot_sha256': 'b' * 64,
            'completion_sha256': 'c' * 64,
            'outputs': resources,
            'kernel_ref': refs.pop(),
        }
    result = continuation._externalize(
        value, completed - continuation.RERUN_COMPLETED, evidence)
    ids = {job['id'] for job in result['jobs']}
    assert 'setup-cpu' not in ids and 'setup-regions' in ids
    assert validate_plan(result)['status'] == 'valid'


def test_recoveries_attach_only_to_both_fixed_calibration_shards(monkeypatch):
    value = plan()
    source = {'files': {continuation.NATIVE_REGIONS: 'f' * 64}}
    registry = {
        'schema': continuation.RECOVERY_SCHEMA,
        'jobs': {
            name: {
                'package_root': f'/private/{name}',
                'resource_name': f'recovery-{name}',
                'resource_ref': f'esreekarreddy/recovery-{name}',
            }
            for name in continuation.RECOVERY_JOBS
        },
    }

    def verified(_, record, job, __):
        return {
            'resource_name': record['resource_name'],
            'resource_spec': {
                'kind': 'dataset', 'ref': record['resource_ref'],
                'marker': 'proposal_recovery_manifest.json', 'suffixes': [''],
                'required_sha256': {'proposal_recovery_lock.json': 'a' * 64},
            },
            'lock_sha256': 'a' * 64,
            'binding_sha256': hashlib.sha256(job.encode()).hexdigest(),
            'package_root': record['package_root'],
        }

    monkeypatch.setattr(continuation, '_verify_recovery', verified)
    attached = continuation._attach_recoveries(value, registry, source)
    assert set(attached) == set(continuation.RECOVERY_JOBS)
    for name in continuation.RECOVERY_JOBS:
        job = next(row for row in value['jobs'] if row['id'] == name)
        assert job['config']['proposal_recovery_root'] == {
            '$resource': f'recovery-{name}'
        }
        assert job['config']['proposal_recovery_lock_sha256'] == 'a' * 64
    validate_plan(value)

    incomplete = copy.deepcopy(registry)
    incomplete['jobs'].pop('calibration-regions-00')
    with pytest.raises(ValueError, match='Both calibration proposal recoveries'):
        continuation._attach_recoveries(plan(), incomplete, source)


def test_completion_evidence_rejects_receipt_or_state_byte_mismatch(tmp_path):
    old_plan = {'resources': {}, 'jobs': []}
    root = tmp_path / 'old'; root.mkdir()
    receipt = {
        'job_id': 'done', 'plan_sha256': continuation._digest(old_plan),
        'source_snapshot_sha256': 'b' * 64, 'outputs': {'out': 'c' * 64},
    }
    path = root / 'receipt.json'; atomic_json(path, receipt)
    state = {'source_snapshot_sha256': 'b' * 64, 'jobs': {'done': {
        'completion_receipt': 'receipt.json',
        'completion_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'kernel_ref': 'owner/done',
    }}}
    assert continuation._completion_evidence(root, old_plan, state, 'done')['outputs'] == {'out': 'c' * 64}
    state['jobs']['done']['completion_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='bytes differ'):
        continuation._completion_evidence(root, old_plan, state, 'done')


def test_active_old_job_requires_exact_not_found_reconciliation(tmp_path):
    record = {
        'status': 'submitted', 'kernel_ref': 'owner/job',
        'notebook_sha256': 'a' * 64, 'metadata_sha256': 'b' * 64,
        'last_sync_error_type': 'RuntimeError',
    }
    with pytest.raises(ValueError, match='active or uncertain'):
        continuation._verify_not_found_reconciliations({'job': record}, [])

    value = {
        'schema': continuation.RECONCILIATION_SCHEMA,
        'job_id': 'job', 'kernel_ref': 'owner/job',
        'checked_at_utc': '2026-09-18T23:21:11Z',
        'command': ['kaggle', 'kernels', 'status', 'owner/job'],
        'exit_code': 1, 'result': 'http_404_not_found',
        'remote_session_exists': False,
        'old_state_record_sha256': continuation._digest(record),
        'safe_continuation_action': 'return_job_to_untracked_in_fresh_queue_only',
        'old_queue_mutated': False,
    }
    path = tmp_path / 'reconciliation.json'; atomic_json(path, value)
    checked = continuation._verify_not_found_reconciliations({'job': record}, [path])
    assert checked['job']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()

    forged = copy.deepcopy(value); forged['remote_session_exists'] = True
    atomic_json(path, forged)
    with pytest.raises(ValueError, match='evidence differs'):
        continuation._verify_not_found_reconciliations({'job': record}, [path])


def test_running_old_job_cannot_be_reset_by_not_found_receipt(tmp_path):
    record = {'status': 'running', 'kernel_ref': 'owner/job'}
    value = {
        'schema': continuation.RECONCILIATION_SCHEMA,
        'job_id': 'job', 'kernel_ref': 'owner/job',
        'checked_at_utc': '2026-09-18T23:21:11Z',
        'command': ['kaggle', 'kernels', 'status', 'owner/job'],
        'exit_code': 1, 'result': 'http_404_not_found',
        'remote_session_exists': False,
        'old_state_record_sha256': continuation._digest(record),
        'safe_continuation_action': 'return_job_to_untracked_in_fresh_queue_only',
        'old_queue_mutated': False,
    }
    path = tmp_path / 'reconciliation.json'; atomic_json(path, value)
    with pytest.raises(ValueError, match='evidence differs'):
        continuation._verify_not_found_reconciliations({'job': record}, [path])
