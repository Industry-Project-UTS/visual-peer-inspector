#!/usr/bin/env python3
"""Build a fresh two-job queue for setup and DINO FP32 CUDA compatibility.

The existing CPU queue and its active state are read only.  This builder emits
one setup-only job and one label-free development compatibility job; it never
submits either job and it contains no production GPU cache shards.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from build_current_launch_plan import plan as current_plan
from current_launch import build_kernels, referenced_resources, validate_plan


NOTEBOOK = "notebooks/input-preparation/current-dino-gpu-compatibility.ipynb"


def plan():
    base = current_plan()
    jobs = {job["id"]: job for job in base["jobs"]}
    cache = copy.deepcopy(jobs["development-cache-00"]["config"])
    cache.update(
        action="gpu_compatibility",
        reference_feature_root={"$resource": "reuse100"},
        output_root={"$output": "dino-gpu-compatibility"},
    )
    for key in (
        "feature_root", "feature_roots", "neutral_root", "shard_count", "shard_index",
        "max_new_units", "max_new_scenes", "maximum_output_bytes",
    ):
        cache.pop(key, None)

    needed = referenced_resources(cache) | {"source"}
    resources = {}
    for name in needed:
        resource = copy.deepcopy(base["resources"][name])
        resource.pop("producer_job", None)
        resources[name] = resource
    resources.update({
        "runtime-dino-gpu": {
            "kind": "kernel",
            "ref": "esreekarreddy/vpi-current-setup-dino-gpu",
            "producer_job": "setup-dino-gpu",
            "marker": "runtime_profile_lock.json",
            "suffixes": ["setup-dino-gpu/runtime-profile"],
        },
        "wheels-dino-gpu": {
            "kind": "kernel",
            "ref": "esreekarreddy/vpi-current-setup-dino-gpu",
            "producer_job": "setup-dino-gpu",
            "marker": "wheelhouse_manifest.json",
            "suffixes": ["setup-dino-gpu/wheelhouse"],
        },
        "dino-gpu-compatibility": {
            "kind": "kernel",
            "ref": "esreekarreddy/vpi-current-dino-gpu-compatibility",
            "producer_job": "dino-gpu-compatibility",
            "marker": "artifact_lock.json",
            "suffixes": ["dino-gpu-compatibility"],
        },
    })
    setup = {
        "id": "setup-dino-gpu",
        "purpose": "Setup only: resolve exact DINO CUDA FP32 dependencies and observe package metadata; no model or dataset execution",
        "notebook": "notebooks/input-preparation/current-kaggle-runtime-setup.ipynb",
        "runtime": "cpu",
        "setup_job": True,
        "depends_on": [],
        "attach_resources": [],
        "config": {
            "profile": "dino-gpu",
            "action": "prepare",
            "build_wheelhouse": True,
            "download_models": False,
            "wheelhouse": "/kaggle/working/setup-dino-gpu/wheelhouse",
            "target": "/tmp/vpi-runtime-setup-dino-gpu",
            "output_root": "/kaggle/working/setup-dino-gpu",
        },
    }
    compatibility = {
        "id": "dino-gpu-compatibility",
        "purpose": "Label-free strict-FP32 CUDA comparison with the sealed 100-scene CPU DINO cache",
        "notebook": NOTEBOOK,
        "runtime": "dino-gpu",
        "input_boundary": "model_only",
        "depends_on": ["setup-dino-gpu"],
        "config": cache,
    }
    value = {
        "schema": "vpi-dino-gpu-compatibility-launch-plan/v1",
        "models_fixed": True,
        "raw_dataset_reupload_required": False,
        "scientific_runs_started": False,
        "queue_scope": "setup plus one label-free development compatibility job; no production feature cache",
        "resources": resources,
        "jobs": [setup, compatibility],
    }
    validate_plan(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_kernels(plan(), args.source_root, args.output_root))


if __name__ == "__main__":
    main()
