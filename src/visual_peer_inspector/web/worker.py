"""Entry point run only by the isolated inference Python environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path


def check_installation():
    """Import/metadata checks only: no CUDA initialization or model downloads."""
    for module in (
        "torch",
        "torchvision",
        "transformers",
        "sam2",
        "scipy",
        "peft",
        "decord",
    ):
        if importlib.util.find_spec(module) is None:
            raise ModuleNotFoundError(module)
        importlib.import_module(module)
    from visual_peer_inspector.models.sam2 import (
        SAM2_CODE_REVISION,
        verify_installed_sam2_revision,
    )

    from .model_runtime import runtime_versions

    verify_installed_sam2_revision(SAM2_CODE_REVISION)
    return runtime_versions()


def main():
    parser = argparse.ArgumentParser(description="Visual Peer Inspector model worker")
    parser.add_argument("--job", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check installed packages without loading weights",
    )
    args = parser.parse_args()
    channel = sys.stdout

    def emit(value):
        print(json.dumps(value), file=channel, flush=True)

    if args.check:
        emit({"dependencies": check_installation(), "models_loaded": False})
        return 0
    if args.job is None:
        parser.error("--job or --check is required")
    models = None
    try:
        with redirect_stdout(sys.stderr):
            from .model_runtime import GPUModels, runtime_versions
            # Cached decisions need only NumPy/SciPy and never initialise CUDA,
            # SAM, Transformers or checkpoint downloads. Neural stages validate
            # their own dependencies lazily when missing evidence requires them.
            versions = runtime_versions()
            from .pipeline import UploadPipeline
            from .transport import read_request, write_result

            models = GPUModels(
                allow_downloads=os.environ.get("VPI_ALLOW_MODEL_DOWNLOADS") == "1"
            )
            try:
                pipeline = UploadPipeline(
                    models, review_policy_path=os.environ.get("VPI_REVIEW_POLICY")
                )
            except (ValueError, OSError, KeyError):
                emit({"error": "policy"})
                return 1
            request = read_request(args.job)
            result = pipeline.inspect(request, lambda stage: emit({"stage": str(stage), "detail": getattr(stage, 'detail', '')}))
            result.provenance["runtime_versions"] = versions
            write_result(args.job, result)
        emit({"complete": True})
        return 0
    except Exception as error:  # noqa: BLE001 - isolate model failures at the process boundary
        from .model_runtime import RuntimeUnavailable

        if isinstance(error, RuntimeUnavailable):
            code = str(error)
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            code = "dependencies"
        elif "out of memory" in str(error).lower():
            code = "out_of_memory"
        else:
            code = "inference_failed"
        traceback.print_exc(file=sys.stderr)
        emit({"error": code})
        return 1
    finally:
        if models is not None:
            models.release()


if __name__ == "__main__":
    raise SystemExit(main())
