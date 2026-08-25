#!/usr/bin/env python3
"""Run the DS1 SAM 2.1 workflow without network tracking."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.sam2 import Sam2Configuration
from visual_peer_inspector.workflows.sam_ds1 import run_ds1_sam_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-mode",
        choices=("contract_smoke", "model_inference"),
        required=True,
    )
    parser.add_argument(
        "--inference-stage",
        choices=("one_image", "pilot", "full"),
        default="full",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--model-load-json", type=Path)
    return parser.parse_args()


def _read_json(path: Path | None, default: Any) -> Any:
    if path is None:
        return default
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_evidence() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "device_count": 0}
    available = bool(torch.cuda.is_available())
    return {
        "cuda_available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    configuration = Sam2Configuration()
    checkpoint_sha256 = args.checkpoint_sha256
    if checkpoint is not None:
        observed = file_sha256(checkpoint)
        if checkpoint_sha256 is not None and checkpoint_sha256 != observed:
            raise ValueError("SAM checkpoint hash differs from the expected value")
        checkpoint_sha256 = observed
        if checkpoint_sha256 != configuration.checkpoint_sha256:
            raise ValueError("SAM checkpoint hash differs from the pinned configuration")

    def progress(stage: str, details: dict[str, Any]) -> None:
        print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)

    record = run_ds1_sam_workflow(
        prompt_root=args.prompt_root,
        media_root=args.media_root,
        output_root=args.output_root,
        configuration=configuration,
        run_mode=args.run_mode,
        inference_stage=args.inference_stage,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        provenance=_read_json(args.provenance_json, {}),
        platform_versions={
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pillow": _package_version("pillow"),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
        },
        gpu_evidence=_gpu_evidence(),
        model_load_rows=_read_json(args.model_load_json, []),
        progress_callback=progress,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
