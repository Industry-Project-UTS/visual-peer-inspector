#!/usr/bin/env python3
"""Run one real LocateAnything image inside a disposable GPU process."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT  # noqa: E402
from visual_peer_inspector.data.model_safe_package import verify_model_input_package  # noqa: E402
from visual_peer_inspector.models.locateanything import (  # noqa: E402
    LocateAnythingConfiguration,
    LocateAnythingWorker,
    build_proposal_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def write_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    configuration = LocateAnythingConfiguration()
    started = time.perf_counter()
    checkpoints: list[dict] = []

    def report(stage: str, details: dict) -> None:
        row = {
            "stage": stage,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            **dict(details),
        }
        checkpoints.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    receipt = {
        "schema_version": 1,
        "stage": "ds1_locateanything_disposable_preflight",
        "status": "failed",
        "model_id": configuration.model_id,
        "model_revision": configuration.model_revision,
        "configuration_sha256": configuration.sha256,
        "proposal_count": 0,
        "runtime_seconds": 0.0,
        "load_seconds": 0.0,
        "checkpoints": checkpoints,
    }
    try:
        audit = verify_model_input_package(
            args.model_input_root,
            contract=DS1_SCREEN_CONTRACT,
        )
        jobs = build_proposal_jobs(
            args.model_input_root,
            DS1_SCREEN_CONTRACT.cohort_id,
        )
        report("input_verified", {"images": audit.image_count})
        load_started = time.perf_counter()
        worker = LocateAnythingWorker(configuration, progress_callback=report)
        receipt["load_seconds"] = round(time.perf_counter() - load_started, 3)
        result = worker.run_job(jobs[0], attempt=1)
        receipt.update(
            {
                "status": "passed" if result["status"] == "ok" else "failed",
                "proposal_count": int(result["proposal_count"]),
                "runtime_seconds": float(result["runtime_seconds"]),
                "error_type": result["error_type"],
                "python_version": platform.python_version(),
                "torch_version": importlib.metadata.version("torch"),
                "transformers_version": importlib.metadata.version("transformers"),
            }
        )
        import torch

        receipt.update(
            {
                "cuda_version": str(torch.version.cuda),
                "cudnn_version": str(torch.backends.cudnn.version()),
                "device_name": torch.cuda.get_device_name(0),
                "peak_gpu_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 4
                ),
                "peak_gpu_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / (1024**3), 4
                ),
            }
        )
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        receipt["error_message"] = str(error)[:300]
    receipt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_receipt(args.receipt, receipt)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
