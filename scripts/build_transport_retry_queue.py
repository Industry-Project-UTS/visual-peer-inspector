#!/usr/bin/env python3
"""Rebuild a fresh queue after a source-transport boundary correction.

The old queue and all completed Kaggle outputs remain immutable.  This builder
accepts only one scientific-tree transition: removal of the completed local
GPU compatibility diagnostic.  Every other ``src`` byte, including the frozen
region implementation and runtime profile, must remain identical.  Completed
jobs become external prerequisites and failed jobs return to untracked state in
the new queue.  No Kaggle command is executed here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import build_continuation_queue as continuation


SCHEMA = "vpi-current-transport-retry-queue/v1"
REMOVED_LOCAL_DIAGNOSTIC = "src/vpi_next_evaluation/gpu_compatibility.py"


def _verify_source_transition(old_source, new_source, verify_source):
    old = verify_source(Path(old_source))
    new = verify_source(Path(new_source))
    old_files, new_files = old["files"], new["files"]
    old_src = {name: value for name, value in old_files.items() if name.startswith("src/")}
    new_src = {name: value for name, value in new_files.items() if name.startswith("src/")}
    if REMOVED_LOCAL_DIAGNOSTIC not in old_src or REMOVED_LOCAL_DIAGNOSTIC in new_src:
        raise ValueError("Transport retry must remove the completed local GPU diagnostic")
    expected = dict(old_src)
    expected.pop(REMOVED_LOCAL_DIAGNOSTIC)
    if new_src != expected:
        raise ValueError("Transport retry changes frozen scientific source bytes")
    for relative in (continuation.REGION_PROFILE, continuation.NATIVE_REGIONS):
        if old_files.get(relative) != new_files.get(relative):
            raise ValueError("Transport retry changes the frozen region implementation or profile")
    return old, new


def _old_external_evidence(old_plan, old_full_plan):
    external = old_plan.get("external_completed_jobs")
    if not isinstance(external, dict):
        raise ValueError("Old continuation plan omits external completion evidence")
    full_jobs = {job["id"] for job in old_full_plan["jobs"]}
    current_jobs = {job["id"] for job in old_plan["jobs"]}
    if current_jobs | set(external) != full_jobs or current_jobs & set(external):
        raise ValueError("Old continuation is not an exact projection of its full plan")
    evidence = copy.deepcopy(external)
    for name, record in evidence.items():
        if set(record) != {
            "plan_sha256", "source_snapshot_sha256", "completion_sha256",
            "outputs", "kernel_ref",
        }:
            raise ValueError(f"External completion evidence differs: {name}")
        if not isinstance(record["outputs"], dict) or not record["outputs"]:
            raise ValueError(f"External completion output inventory is empty: {name}")
    return evidence


def retry_plan(old_queue, old_source, old_full_queue, new_full_queue, new_source,
               recovery_registry):
    for path in (Path(old_source), Path(new_source)):
        sys.path.insert(0, str(path / "scripts"))
        sys.path.insert(0, str(path / "src"))
    import current_kaggle_queue as queue
    from current_notebook_runtime import verify_source

    old_source_record, new_source_record = _verify_source_transition(
        old_source, new_source, verify_source)
    old_full_plan = json.loads((Path(old_full_queue) / "launch-plan.json").read_text())
    new_full_plan = continuation._verify_fresh_materialization(
        new_full_queue, new_source, queue, new_source_record)
    continuation._same_scientific_contract(old_full_plan, new_full_plan)

    with queue._queue_lock(old_queue):
        old_root, old_plan, old_state, _ = queue.open_queue(old_queue)
        if old_state.get("source_snapshot_sha256") != old_source_record["snapshot_sha256"]:
            raise ValueError("Old queue and old source snapshot differ")
        active = {
            name for name, record in old_state["jobs"].items()
            if record.get("status") in queue.ACTIVE
        }
        if active:
            raise ValueError("Synchronize active jobs before building a transport retry")
        if old_plan.get("continuation", {}).get("new_full_plan_sha256") != continuation._digest(old_full_plan):
            raise ValueError("Old continuation does not name the supplied full plan")
        evidence = _old_external_evidence(old_plan, old_full_plan)
        newly_completed = {
            name for name, record in old_state["jobs"].items()
            if record.get("status") == queue.COMPLETE
        }
        for name in newly_completed:
            if name in evidence:
                raise ValueError("Completed job appears twice in continuation evidence")
            evidence[name] = continuation._completion_evidence(
                old_root, old_plan, old_state, name)

    completed = set(evidence)
    result = continuation._externalize(new_full_plan, completed, evidence)
    registry = json.loads(Path(recovery_registry).read_text())
    attached = continuation._attach_recoveries(result, registry, new_source_record)
    result["continuation"] = {
        "schema": SCHEMA,
        "old_plan_sha256": continuation._digest(old_plan),
        "old_full_plan_sha256": continuation._digest(old_full_plan),
        "old_source_snapshot_sha256": old_source_record["snapshot_sha256"],
        "new_full_plan_sha256": continuation._digest(new_full_plan),
        "new_source_snapshot_sha256": new_source_record["snapshot_sha256"],
        "external_completed_job_count": len(completed),
        "reset_failed_jobs": sorted(
            name for name, record in old_state["jobs"].items()
            if record.get("status") == "failed"
        ),
        "removed_local_diagnostic": REMOVED_LOCAL_DIAGNOSTIC,
        "proposal_recovery_jobs": list(continuation.RECOVERY_JOBS),
    }
    queue.validate_plan(result)
    return result, attached, new_source_record


def build_queue(old_queue, old_source, old_full_queue, new_full_queue, new_source,
                recovery_registry, output_queue):
    destination = Path(output_queue)
    if destination.exists():
        raise FileExistsError("Choose a fresh retry queue destination")
    plan, attached, source = retry_plan(
        Path(old_queue), Path(old_source), Path(old_full_queue),
        Path(new_full_queue), Path(new_source), Path(recovery_registry))

    sys.path.insert(0, str(Path(new_source) / "scripts"))
    import current_kaggle_queue as queue
    from current_launch import build_kernels
    from current_notebook_runtime import atomic_json
    from build_isolated_cpu_queue import build_queue as isolate_cpu
    from build_isolated_cache_queue import build_queue as isolate_cache
    from build_isolated_region_queue import build_queue as isolate_regions
    from validate_launch_readiness import audit_queue

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
        atomic_json(base_queue / "queue-state.json", {
            "schema": queue.SCHEMA,
            "plan_sha256": continuation._digest(plan),
            "source_snapshot_sha256": source["snapshot_sha256"],
            "prepared_jobs": prepared,
            "jobs": {},
        })
        queue.open_queue(base_queue)
        cpu_receipt = isolate_cpu(base_queue, cpu_queue, new_source, retry_job=None)
        cache_receipt = isolate_cache(cpu_queue, cache_queue, new_source)
        region_receipt = isolate_regions(cache_queue, region_queue, new_source)
        queue.open_queue(region_queue)
        readiness = audit_queue(region_queue)
        atomic_json(build_root / "launch-readiness.json", readiness)
        if readiness.get("status") != "ready" or readiness.get("blockers"):
            raise ValueError("Transport retry queue did not pass launch readiness")
        receipt = {
            "schema": SCHEMA,
            "old_queue": str(Path(old_queue)),
            "new_source_snapshot_sha256": source["snapshot_sha256"],
            "continuation_plan_sha256": continuation._digest(plan),
            "remaining_job_count": len(plan["jobs"]),
            "external_completed_jobs": sorted(plan["external_completed_jobs"]),
            "reset_failed_jobs": plan["continuation"]["reset_failed_jobs"],
            "proposal_recoveries": {
                name: {key: value for key, value in row.items() if key != "package_root"}
                for name, row in attached.items()
            },
            "isolation_receipts": {
                "cpu": {"patched_job_count": cpu_receipt["patched_job_count"]},
                "cache": {"patched_job_count": cache_receipt["patched_job_count"]},
                "regions": {"patched_job_count": region_receipt["patched_job_count"]},
            },
            "launch_readiness_status": readiness["status"],
            "launch_readiness_sha256": continuation._file_digest(
                build_root / "launch-readiness.json"),
            "kaggle_commands_run": False,
        }
        atomic_json(build_root / "transport-retry-queue-receipt.json", receipt)
        os.replace(region_queue, destination)
        moved = True
        for name in (
            "isolated-cpu-queue-receipt.json", "isolated-cache-queue-receipt.json",
            "isolated-region-queue-receipt.json", "launch-readiness.json",
        ):
            shutil.copyfile(build_root / name, destination / name)
        atomic_json(destination.parent / "transport-retry-queue-receipt.json", receipt)
        shutil.rmtree(build_root)
        return receipt
    except BaseException:
        if moved and destination.exists():
            shutil.rmtree(destination)
        if build_root.exists():
            shutil.rmtree(build_root)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-queue", required=True, type=Path)
    parser.add_argument("--old-source", required=True, type=Path)
    parser.add_argument("--old-full-queue", required=True, type=Path)
    parser.add_argument("--new-full-queue", required=True, type=Path)
    parser.add_argument("--new-source", required=True, type=Path)
    parser.add_argument("--proposal-recovery-registry", required=True, type=Path)
    parser.add_argument("--output-queue", required=True, type=Path)
    args = parser.parse_args()
    result = build_queue(
        args.old_queue, args.old_source, args.old_full_queue, args.new_full_queue,
        args.new_source, args.proposal_recovery_registry, args.output_queue)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
