"""Run foreground patch association and its development-only evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "predict", "evaluate", "verify"))
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--contract-encoder", action="store_true")
    args = parser.parse_args()

    from visual_peer_inspector.workflows.comparison_io import verify_seal
    from visual_peer_inspector.workflows.identity_patch_association_study import (
        MODEL_REVISION,
        evaluate,
        predict,
        preflight,
    )

    if args.stage == "preflight":
        print(preflight(args.model_input, args.scoring))
        return
    if args.stage == "verify":
        lock_name = "prediction_lock.json" if (args.output / "prediction_lock.json").is_file() else "artifact_lock.json"
        print(verify_seal(args.output, name=lock_name))
        return
    if args.stage == "evaluate":
        if args.predictions is None or args.evaluator is None:
            parser.error("Evaluation requires --predictions and --evaluator")
        summary = evaluate(
            args.model_input,
            args.scoring,
            args.evaluator,
            args.predictions,
            args.output,
        )
        print(summary.to_string(index=False))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output.parent).free
    if free_bytes < 2 * 1024**3:
        raise OSError("Less than 2 GiB is available in the output volume")
    if args.contract_encoder:
        from visual_peer_inspector.models.visual_encoder import ContractVisualEncoder

        encoder_factory = ContractVisualEncoder
        producer = "contract-encoder"
    else:
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Patch correspondence feature extraction needs a CUDA GPU")
        torch.manual_seed(32_040)
        print(
            {
                "event": "runtime_ready",
                "gpu": torch.cuda.get_device_name(0),
                "visible_gpus": torch.cuda.device_count(),
                "devices_used": 1,
                "output_free_bytes": free_bytes,
            },
            flush=True,
        )

        def encoder_factory():
            from huggingface_hub import snapshot_download
            from visual_peer_inspector.models.visual_encoder import FrozenVisualEncoder, specification_by_key

            specification = specification_by_key("dinov2_small")
            print({"event": "model_download_start", "revision": MODEL_REVISION}, flush=True)
            snapshot = snapshot_download(repo_id=specification.model_id, revision=MODEL_REVISION)
            print({"event": "model_download_complete", "snapshot": Path(snapshot).name}, flush=True)
            encoder = FrozenVisualEncoder(
                specification,
                revision=MODEL_REVISION,
                device="cpu",
                model_source=snapshot,
            )
            print({"event": "model_cpu_ready"}, flush=True)
            encoder.model.to("cuda:0")
            encoder.device = "cuda:0"
            torch.cuda.reset_peak_memory_stats(0)
            print({"event": "model_cuda_ready", "allocated_bytes": torch.cuda.memory_allocated(0)}, flush=True)
            return encoder

        producer = "dinov2-small-foreground-patch-correspondence"
    predict(
        args.model_input,
        args.scoring,
        args.output,
        SOURCE,
        encoder_factory,
        producer=producer,
        recovery_bundle=args.recovery_bundle,
        recovery_manifest=args.recovery_manifest,
    )
    if not args.contract_encoder:
        import torch

        print({"event": "gpu_summary", "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)}, flush=True)


if __name__ == "__main__":
    main()
