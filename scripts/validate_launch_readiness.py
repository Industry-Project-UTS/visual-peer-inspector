#!/usr/bin/env python3
"""Fail closed when a prepared Kaggle queue still has known runtime hazards.

This validator is read-only.  It checks the materialized notebooks and metadata,
not merely the logical launch plan, because queue builders may add the isolated
child process after the base notebooks are generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "vpi-kaggle-launch-readiness/v1"
COMPLETE_STATUSES = frozenset({"kaggle_complete_with_receipt"})
SETUP_ACTION = "prepare"


def _source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _notebook_text(queue_root, job):
    path = Path(queue_root) / job["id"] / Path(job["notebook"]).name
    notebook = json.loads(path.read_text())
    return "\n".join(
        _source(cell) for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _metadata(queue_root, job):
    return json.loads((Path(queue_root) / job["id"] / "kernel-metadata.json").read_text())


def _resource_names(value):
    if isinstance(value, dict):
        if "$resource" in value:
            return {value["$resource"]}
        result = set()
        for item in value.values():
            result.update(_resource_names(item))
        return result
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(_resource_names(item))
        return result
    return set()


def _sam_contract(plan):
    """Return an explicit SAM code contract or None.

    Model checkpoint bytes are not the SAM Python package.  The plan must name
    both the immutable code resource and the preflight marker executed by each
    region notebook before proposal inference.
    """
    value = plan.get("runtime_readiness", {}).get("sam2_code")
    if not isinstance(value, dict):
        return None
    required = {"resource", "distribution", "repository", "revision", "preflight_marker"}
    if set(value) != required:
        return None
    revision = value.get("revision")
    if (value.get("distribution") != "SAM-2"
            or value.get("repository") != "https://github.com/facebookresearch/sam2"
            or not isinstance(revision, str) or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
            or not isinstance(value.get("preflight_marker"), str)
            or not value["preflight_marker"]):
        return None
    if value["resource"] not in plan.get("resources", {}):
        return None
    return value


def audit_queue(queue_root):
    """Classify every job and return blockers without changing the queue."""
    queue_root = Path(queue_root)
    plan = json.loads((queue_root / "launch-plan.json").read_text())
    state_path = queue_root / "queue-state.json"
    state = json.loads(state_path.read_text()).get("jobs", {}) if state_path.is_file() else {}
    sam_contract = _sam_contract(plan)
    rows = []
    blockers = []
    warnings = []

    for job in plan["jobs"]:
        job_id = job["id"]
        runtime = job.get("runtime", "cpu")
        action = job.get("config", {}).get("action")
        status = state.get(job_id, {}).get("status", "untracked")
        complete = status in COMPLETE_STATUSES
        metadata = _metadata(queue_root, job)
        text = _notebook_text(queue_root, job)
        sources = len(metadata.get("dataset_sources", [])) + len(metadata.get("kernel_sources", []))
        family = (
            "setup" if job.get("setup_job") else
            "regions-gpu" if runtime == "regions-gpu" else
            "dino-cache" if runtime.startswith("dino-") else
            "cpu-science"
        )
        job_blockers = []
        job_warnings = []

        if not complete and family == "cpu-science":
            clean_child = (
                "vpi-isolated-cpu-child/v1" in text
                and '[sys.executable, "-S", "-B"' in text
            )
            if not clean_child:
                job_blockers.append("cpu_not_clean_child")

        if not complete and family == "dino-cache":
            clean_child = (
                "vpi-isolated-cache-child/v1" in text
                and '[sys.executable, "-S", "-B"' in text
                and "runtime_imports_verified" in text
            )
            if not clean_child:
                job_blockers.append("dino_cache_not_clean_child")

        if not complete and family == "regions-gpu":
            clean_child = (
                "vpi-isolated-region-child/v1" in text
                and '[sys.executable, "-S", "-B"' in text
                and metadata.get("enable_gpu") is True
                and metadata.get("enable_internet") is False
            )
            if not clean_child:
                job_blockers.append("region_not_clean_gpu_child")
            refs = _resource_names(job.get("config", {})) | set(job.get("attach_resources", []))
            # Runtime wheelhouses are implicit Kaggle inputs for their runtime.
            refs.add("wheels-regions")
            if (sam_contract is None
                    or sam_contract["resource"] not in refs
                    or sam_contract["preflight_marker"] not in text):
                job_blockers.append("region_sam_code_preflight_missing")

        if not complete and job.get("config", {}).get("publish_tracking"):
            if "publish_completed_job(" in text and metadata.get("enable_internet") is not True:
                job_blockers.append("tracking_inline_without_internet")

        if not job.get("setup_job") and metadata.get("enable_internet") is True:
            job_blockers.append("scientific_job_has_internet")
        if job.get("setup_job") and metadata.get("enable_internet") is not True:
            job_blockers.append("setup_job_has_no_internet")
        if sources > 20:
            job_warnings.append("large_kaggle_input_fan_in")

        row = {
            "job_id": job_id,
            "family": family,
            "runtime": runtime,
            "action": action,
            "status": status,
            "complete": complete,
            "enable_gpu": bool(metadata.get("enable_gpu")),
            "enable_internet": bool(metadata.get("enable_internet")),
            "input_source_count": sources,
            "blockers": job_blockers,
            "warnings": job_warnings,
        }
        rows.append(row)
        blockers.extend({"job_id": job_id, "code": code} for code in job_blockers)
        warnings.extend({"job_id": job_id, "code": code} for code in job_warnings)

    families = {}
    for name in ("setup", "cpu-science", "dino-cache", "regions-gpu"):
        selected = [row for row in rows if row["family"] == name]
        families[name] = {
            "jobs": len(selected),
            "complete": sum(row["complete"] for row in selected),
            "blocked": sum(bool(row["blockers"]) for row in selected),
            "warning": sum(bool(row["warnings"]) for row in selected),
        }
    return {
        "schema": SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "job_count": len(rows),
        "families": families,
        "blockers": blockers,
        "warnings": warnings,
        "jobs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_queue(args.queue_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".partial")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    raise SystemExit(0 if report["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
