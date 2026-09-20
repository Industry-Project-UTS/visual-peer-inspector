#!/usr/bin/env python3
"""Build a fresh queue from verified completions and proposal recoveries.

The old queue, its state, and its source package are read only.  Completed jobs
become immutable external prerequisites in a newly generated plan.  The region
runtime setup is deliberately rerun because the current profile adds the pinned
SAM-2 source distribution that is absent from the old wheelhouse.

This command performs no Kaggle operation.  It only accepts a separately
materialized, unstarted full queue and writes to a fresh destination.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


SCHEMA = "vpi-current-continuation-queue/v1"
RECOVERY_SCHEMA = "vpi-current-proposal-recovery-registry/v1"
RERUN_COMPLETED = frozenset({"setup-regions"})
REGION_PROFILE = "requirements/profiles/linux-regions-cuda126.json"
NATIVE_REGIONS = "src/vpi_next_evaluation/native_regions.py"
RECOVERY_JOBS = ("calibration-regions-00", "calibration-regions-01")
RECONCILIATION_SCHEMA = "vpi-kaggle-not-found-reconciliation/v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _verify_seal(root, *, name):
    """Verify a recovery seal without importing the scientific package tree."""
    root = Path(root).resolve(strict=True)
    lock = _read_json(root / name)
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Proposal recovery has an empty artifact inventory")
    for relative, expected in files.items():
        path = (root / relative).resolve(strict=True)
        if (not path.is_relative_to(root) or not path.is_file()
                or _file_digest(path) != expected):
            raise ValueError(f"Proposal recovery checksum differs: {relative}")
    return lock


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _same_scientific_contract(old, new):
    keys = ("models_fixed", "original_roles", "official_roles", "methods",
            "core_views", "diagnostic_views", "schedule_count")
    if any(old.get(key) != new.get(key) for key in keys):
        raise ValueError("Continuation changes the frozen scientific contract")
    old_jobs = {job["id"]: job for job in old["jobs"]}
    new_jobs = {job["id"]: job for job in new["jobs"]}
    if set(old_jobs) != set(new_jobs):
        raise ValueError("Continuation full plan job inventory differs")
    # Runtime transport may change, but experiment configurations and graph do not.
    for name in old_jobs:
        for key in ("notebook", "runtime", "setup_job", "depends_on", "config",
                    "input_boundary", "compact_checkpoints"):
            if old_jobs[name].get(key) != new_jobs[name].get(key):
                raise ValueError(f"Continuation changes job contract: {name}/{key}")
    old_resources, new_resources = old["resources"], new["resources"]
    if set(old_resources) != set(new_resources):
        raise ValueError("Continuation full plan resource inventory differs")
    for name in old_resources:
        if old_resources[name] != new_resources[name]:
            raise ValueError(f"Continuation changes resource contract: {name}")


def _verify_source_transition(old_source, new_source, verify_source):
    old = verify_source(Path(old_source))
    new = verify_source(Path(new_source))
    old_files, new_files = old["files"], new["files"]
    for relative in (REGION_PROFILE, NATIVE_REGIONS):
        if relative not in old_files or relative not in new_files:
            raise ValueError("Source package omits a continuation contract file")
    if old_files[NATIVE_REGIONS] != new_files[NATIVE_REGIONS]:
        raise ValueError("Proposal recovery cannot cross a native-regions source change")
    if old_files[REGION_PROFILE] == new_files[REGION_PROFILE]:
        raise ValueError("Region setup rerun requires a changed runtime profile")
    profile = json.loads((Path(new_source) / REGION_PROFILE).read_text())
    package = profile.get("source_packages", {}).get("SAM-2", {})
    if (profile.get("runtime_contract", {}).get("packages", {}).get("SAM-2") != "1.0"
            or package.get("version") != "1.0"
            or package.get("repository") != "https://github.com/facebookresearch/sam2"
            or package.get("revision") != "2b90b9f5ceec907a1c18123530e92e794ad901a4"):
        raise ValueError("New region profile does not contain the approved SAM-2 source pin")
    return old, new


def _verify_fresh_materialization(root, new_source, queue, new_source_record):
    root = Path(root)
    plan = json.loads((root / "launch-plan.json").read_text())
    queue.validate_plan(plan)
    state_path = root / "queue-state.json"
    if state_path.is_file() and json.loads(state_path.read_text()).get("jobs"):
        raise ValueError("Continuation requires an unstarted full materialization")
    expected = {job["id"]: queue._job_files(root, plan, job) for job in plan["jobs"]}
    if len(expected) != len(plan["jobs"]):
        raise ValueError("Materialized full queue has duplicate jobs")
    source_resource = plan.get("resources", {}).get("source", {})
    if (source_resource.get("marker") != "prepared_source_manifest.json"
            or source_resource.get("ref") is None):
        raise ValueError("Materialized full queue has no prepared source resource")
    if not _sha(new_source_record.get("snapshot_sha256")):
        raise ValueError("New source snapshot is invalid")
    return plan


def _completion_evidence(old_root, old_plan, old_state, completed_job):
    record = old_state["jobs"][completed_job]
    receipt_path = Path(old_root) / record["completion_receipt"]
    receipt = json.loads(receipt_path.read_text())
    if record.get("completion_sha256") != _file_digest(receipt_path):
        raise ValueError("Old completion receipt bytes differ")
    if (receipt.get("job_id") != completed_job
            or receipt.get("plan_sha256") != _digest(old_plan)
            or receipt.get("source_snapshot_sha256") != old_state["source_snapshot_sha256"]
            or receipt.get("outputs") is None):
        raise ValueError("Old completion receipt binding differs")
    return {
        "plan_sha256": receipt["plan_sha256"],
        "source_snapshot_sha256": receipt["source_snapshot_sha256"],
        "completion_sha256": record["completion_sha256"],
        "outputs": copy.deepcopy(receipt["outputs"]),
        "kernel_ref": record["kernel_ref"],
    }


def _verify_not_found_reconciliations(active, receipt_paths):
    """Require exact local evidence before an uncertain remote job is reset."""
    receipts = {}
    expected_fields = {
        "schema", "job_id", "kernel_ref", "checked_at_utc", "command",
        "exit_code", "result", "remote_session_exists",
        "old_state_record_sha256", "safe_continuation_action", "old_queue_mutated",
    }
    for path_value in receipt_paths or ():
        path = Path(path_value).resolve(strict=True)
        if path.is_symlink():
            raise ValueError("Reconciliation receipt cannot be a symlink")
        value = json.loads(path.read_text())
        if set(value) != expected_fields or value.get("schema") != RECONCILIATION_SCHEMA:
            raise ValueError("Not-found reconciliation receipt schema differs")
        job_id = value.get("job_id")
        if not isinstance(job_id, str) or job_id in receipts:
            raise ValueError("Duplicate or invalid reconciled job")
        record = active.get(job_id)
        if record is None:
            raise ValueError("Reconciliation receipt is not for an active old job")
        kernel_ref = record.get("kernel_ref")
        if (record.get("status") not in {"submitted", "submission_uncertain"}
                or value.get("kernel_ref") != kernel_ref
                or value.get("command") != ["kaggle", "kernels", "status", kernel_ref]
                or value.get("exit_code") != 1
                or value.get("result") != "http_404_not_found"
                or value.get("remote_session_exists") is not False
                or value.get("safe_continuation_action") != "return_job_to_untracked_in_fresh_queue_only"
                or value.get("old_queue_mutated") is not False
                or value.get("old_state_record_sha256") != _digest(record)
                or not isinstance(value.get("checked_at_utc"), str)
                or not value["checked_at_utc"].endswith("Z")):
            raise ValueError("Not-found reconciliation evidence differs from old queue state")
        receipts[job_id] = {"path": str(path), "sha256": _file_digest(path)}
    missing = set(active) - set(receipts)
    if missing:
        raise ValueError("Old queue still has active or uncertain jobs: " + ", ".join(sorted(missing)))
    return receipts


def _externalize(plan, completed, evidence):
    output = copy.deepcopy(plan)
    output["jobs"] = [job for job in output["jobs"] if job["id"] not in completed]
    output["external_completed_jobs"] = {
        name: copy.deepcopy(evidence[name]) for name in sorted(completed)
    }
    for name, record in output["external_completed_jobs"].items():
        for resource, marker_digest in record["outputs"].items():
            spec = output["resources"].get(resource)
            if not spec or spec.get("producer_job") != name:
                raise ValueError("Completed output no longer matches the full plan")
            if spec.get("ref") != record["kernel_ref"]:
                raise ValueError("Completed output kernel reference differs from the full plan")
            spec.pop("producer_job")
            spec["availability"] = "existing_kaggle"
            spec["external_source_job"] = name
            required = dict(spec.get("required_sha256", {}))
            if spec["marker"] in required and required[spec["marker"]] != marker_digest:
                raise ValueError("Completed output marker pin conflicts with the full plan")
            required[spec["marker"]] = marker_digest
            spec["required_sha256"] = required
    output["scientific_runs_started"] = True
    return output


def _verify_recovery(package_root, record, job, new_source_record):
    root = Path(package_root).resolve(strict=True)
    if root.is_symlink():
        raise ValueError("Proposal recovery package cannot be a symlink")
    lock = _verify_seal(root, name="proposal_recovery_lock.json")
    manifest = _read_json(root / "proposal_recovery_manifest.json")
    binding = _read_json(root / "run_binding.json")
    expected_keys = {"package_root", "resource_name", "resource_ref"}
    if set(record) != expected_keys:
        raise ValueError("Proposal recovery registry fields differ")
    if (manifest.get("schema") != "vpi-current-proposal-recovery/v1"
            or manifest.get("completed_proposal_views") != 2000
            or manifest.get("expected_proposal_views") != 2000
            or manifest.get("source_binding_sha256") != _digest(binding)
            or manifest.get("source_native_regions_sha256") != new_source_record["files"][NATIVE_REGIONS]
            or binding.get("role") != "calibration"
            or binding.get("partition", {}).get("count") != 6
            or binding.get("partition", {}).get("index") != int(job.rsplit("-", 1)[1])):
        raise ValueError("Proposal recovery binding or complete view inventory differs")
    lock_digest = _file_digest(root / "proposal_recovery_lock.json")
    if lock.get("metadata") != {
        "schema": "vpi-current-proposal-recovery/v1",
        "source_binding_sha256": manifest["source_binding_sha256"],
    }:
        raise ValueError("Proposal recovery lock metadata differs")
    ref, resource = record["resource_ref"], record["resource_name"]
    if (not isinstance(resource, str) or not resource
            or not isinstance(ref, str) or ref.count("/") != 1):
        raise ValueError("Proposal recovery Kaggle resource identity differs")
    return {
        "resource_name": resource,
        "resource_spec": {
            "kind": "dataset",
            "ref": ref,
            "marker": "proposal_recovery_manifest.json",
            "suffixes": [""],
            "availability": "prepared_local_upload",
            "required_sha256": {"proposal_recovery_lock.json": lock_digest},
        },
        "lock_sha256": lock_digest,
        "binding_sha256": manifest["source_binding_sha256"],
        "package_root": str(root),
    }


def _attach_recoveries(plan, registry, new_source_record):
    if registry.get("schema") != RECOVERY_SCHEMA or set(registry) != {"schema", "jobs"}:
        raise ValueError("Proposal recovery registry schema differs")
    if set(registry["jobs"]) != set(RECOVERY_JOBS):
        raise ValueError("Both calibration proposal recoveries are required")
    jobs = {job["id"]: job for job in plan["jobs"]}
    attached = {}
    for name in RECOVERY_JOBS:
        if name not in jobs:
            raise ValueError("Proposal recovery target is absent from continuation")
        checked = _verify_recovery(
            registry["jobs"][name]["package_root"], registry["jobs"][name],
            name, new_source_record)
        resource = checked["resource_name"]
        if resource in plan["resources"]:
            raise ValueError("Proposal recovery resource name collides")
        plan["resources"][resource] = checked["resource_spec"]
        jobs[name]["config"].update(
            proposal_recovery_root={"$resource": resource},
            proposal_recovery_lock_sha256=checked["lock_sha256"],
            proposal_recovery_binding_sha256=checked["binding_sha256"],
        )
        attached[name] = checked
    return attached


def continuation_plan(old_queue, old_source, materialized_full_queue, new_source,
                      recovery_registry, reconciliation_receipts=()):
    """Return a validated continuation plan and evidence without writing a queue."""
    for path in (Path(old_source), Path(new_source)):
        sys.path.insert(0, str(path / "scripts"))
        sys.path.insert(0, str(path / "src"))
    import current_kaggle_queue as queue
    from current_notebook_runtime import verify_source

    old_source_record, new_source_record = _verify_source_transition(
        old_source, new_source, verify_source)
    with queue._queue_lock(old_queue):
        old_root, old_plan, old_state, _ = queue.open_queue(old_queue)
        if old_state.get("source_snapshot_sha256") != old_source_record["snapshot_sha256"]:
            raise ValueError("Old queue and old source snapshot differ")
        active = {
            name: copy.deepcopy(record) for name, record in old_state["jobs"].items()
            if record.get("status") in queue.ACTIVE
        }
        reconciled = _verify_not_found_reconciliations(active, reconciliation_receipts)
        completed_all = {
            name for name, record in old_state["jobs"].items()
            if record.get("status") == queue.COMPLETE
        }
        if not RERUN_COMPLETED <= completed_all:
            raise ValueError("Old queue does not contain the invalidated region setup completion")
        completed = completed_all - RERUN_COMPLETED
        evidence = {
            name: _completion_evidence(old_root, old_plan, old_state, name)
            for name in completed
        }
    new_plan = _verify_fresh_materialization(
        materialized_full_queue, new_source, queue, new_source_record)
    _same_scientific_contract(old_plan, new_plan)
    plan = _externalize(new_plan, completed, evidence)
    attached = _attach_recoveries(plan, recovery_registry, new_source_record)
    plan["continuation"] = {
        "schema": SCHEMA,
        "old_plan_sha256": _digest(old_plan),
        "old_source_snapshot_sha256": old_source_record["snapshot_sha256"],
        "new_full_plan_sha256": _digest(new_plan),
        "new_source_snapshot_sha256": new_source_record["snapshot_sha256"],
        "rerun_completed_jobs": sorted(RERUN_COMPLETED),
        "external_completed_job_count": len(completed),
        "proposal_recovery_jobs": list(RECOVERY_JOBS),
        "not_found_reconciliations": {
            name: row["sha256"] for name, row in sorted(reconciled.items())
        },
    }
    queue.validate_plan(plan)
    return plan, attached, new_source_record


def build_queue(old_queue, old_source, materialized_full_queue, new_source,
                recovery_registry, output_queue, reconciliation_receipts=()):
    """Generate and isolate a fresh continuation queue; never submit it."""
    destination = Path(output_queue)
    if destination.exists():
        raise FileExistsError("Choose a fresh continuation queue destination")
    registry = json.loads(Path(recovery_registry).read_text())
    plan, attached, source = continuation_plan(
        Path(old_queue), Path(old_source), Path(materialized_full_queue),
        Path(new_source), registry, reconciliation_receipts)

    sys.path.insert(0, str(Path(new_source) / "scripts"))
    import current_kaggle_queue as queue
    from current_launch import build_kernels
    from current_notebook_runtime import atomic_json

    build_root = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    base_queue = build_root / "base"
    cpu_queue = build_root / "cpu"
    cache_queue = build_root / "cache"
    region_queue = build_root / "region"
    build_root.parent.mkdir(parents=True, exist_ok=True)
    moved = False
    try:
        build_root.mkdir()
        build_kernels(plan, new_source, base_queue)
        prepared = {job["id"]: queue._job_files(base_queue, plan, job) for job in plan["jobs"]}
        state = {
            "schema": queue.SCHEMA,
            "plan_sha256": _digest(plan),
            "source_snapshot_sha256": source["snapshot_sha256"],
            "prepared_jobs": prepared,
            "jobs": {},
        }
        atomic_json(base_queue / "queue-state.json", state)
        queue.open_queue(base_queue)

        from build_isolated_cpu_queue import build_queue as isolate_cpu
        from build_isolated_cache_queue import build_queue as isolate_cache
        from build_isolated_region_queue import build_queue as isolate_regions
        cpu_receipt = isolate_cpu(base_queue, cpu_queue, new_source, retry_job=None)
        cache_receipt = isolate_cache(cpu_queue, cache_queue, new_source)
        region_receipt = isolate_regions(cache_queue, region_queue, new_source)
        queue.open_queue(region_queue)

        from validate_launch_readiness import audit_queue
        readiness = audit_queue(region_queue)
        atomic_json(build_root / "launch-readiness.json", readiness)
        if readiness.get("status") != "ready" or readiness.get("blockers"):
            raise ValueError("Continuation queue did not pass launch readiness")
        receipt = {
            "schema": SCHEMA,
            "old_queue": str(Path(old_queue)),
            "materialized_full_queue": str(Path(materialized_full_queue)),
            "new_source_snapshot_sha256": source["snapshot_sha256"],
            "continuation_plan_sha256": _digest(plan),
            "remaining_job_count": len(plan["jobs"]),
            "external_completed_jobs": sorted(plan["external_completed_jobs"]),
            "rerun_completed_jobs": sorted(RERUN_COMPLETED),
            "not_found_reconciliations": copy.deepcopy(
                plan["continuation"]["not_found_reconciliations"]),
            "proposal_recoveries": {
                name: {key: value for key, value in row.items() if key != "package_root"}
                for name, row in attached.items()
            },
            "isolation_receipts": {
                "cpu": {"sha256": _file_digest(build_root / "isolated-cpu-queue-receipt.json"),
                        "patched_job_count": cpu_receipt["patched_job_count"]},
                "cache": {"sha256": _file_digest(build_root / "isolated-cache-queue-receipt.json"),
                          "patched_job_count": cache_receipt["patched_job_count"]},
                "regions": {"sha256": _file_digest(build_root / "isolated-region-queue-receipt.json"),
                            "patched_job_count": region_receipt["patched_job_count"]},
            },
            "launch_readiness_status": readiness["status"],
            "launch_readiness_sha256": _file_digest(build_root / "launch-readiness.json"),
            "kaggle_commands_run": False,
        }
        atomic_json(build_root / "continuation-queue-receipt.json", receipt)
        os.replace(region_queue, destination)
        moved = True
        for name in ("isolated-cpu-queue-receipt.json", "isolated-cache-queue-receipt.json",
                     "isolated-region-queue-receipt.json", "launch-readiness.json"):
            shutil.copyfile(build_root / name, destination / name)
        atomic_json(destination.parent / "continuation-queue-receipt.json", receipt)
        shutil.rmtree(build_root)
        return receipt
    except BaseException:
        if moved and destination.exists():
            shutil.rmtree(destination)
        if build_root.exists():
            failure = {
                "schema": SCHEMA,
                "status": "failed_closed",
                "error": "Continuation build stopped before launch readiness",
                "kaggle_commands_run": False,
            }
            atomic_json(build_root.parent / (build_root.name + "-failure.json"), failure)
            shutil.rmtree(build_root)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-queue", required=True, type=Path)
    parser.add_argument("--old-source", required=True, type=Path)
    parser.add_argument("--materialized-full-queue", required=True, type=Path)
    parser.add_argument("--new-source", required=True, type=Path)
    parser.add_argument("--proposal-recovery-registry", required=True, type=Path)
    parser.add_argument("--not-found-reconciliation", action="append", default=[], type=Path)
    parser.add_argument("--output-queue", required=True, type=Path)
    args = parser.parse_args()
    result = build_queue(
        args.old_queue, args.old_source, args.materialized_full_queue,
        args.new_source, args.proposal_recovery_registry, args.output_queue,
        args.not_found_reconciliation)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
