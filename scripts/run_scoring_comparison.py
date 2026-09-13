"""Run the prediction or evaluation stage from explicit package paths."""

import argparse
import os
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "predict", "evaluate", "verify"])
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--recovery-bundle", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--cohort", choices=["smoke20", "screen100"], default="smoke20")
    args = parser.parse_args()
    from visual_peer_inspector.workflows.scoring_comparison import REVISION, preflight, predict
    from visual_peer_inspector.workflows.comparison_io import verify_seal
    if args.stage == "preflight":
        print(preflight(args.model_input))
    elif args.stage == "predict":
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This notebook needs a CUDA GPU; do not silently start CPU inference")
        torch.manual_seed(32040)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(args.output.parent).free
        if free < 2 * 1024**3:
            raise OSError("Less than 2 GiB is available in the output volume")
        device = torch.cuda.get_device_properties(0)
        print({"event": "runtime_ready", "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": device.total_memory, "torch": torch.__version__, "visible_gpus": torch.cuda.device_count(), "devices_used": 1, "output_free_bytes": free}, flush=True)
        from visual_peer_inspector.models.visual_encoder import FrozenVisualEncoder, specification_by_key
        def load_encoder():
            from huggingface_hub import snapshot_download

            specification = specification_by_key("dinov2_small")
            print({"event": "model_download_start", "model": specification.model_id, "revision": REVISION}, flush=True)
            snapshot = snapshot_download(repo_id=specification.model_id, revision=REVISION)
            print({"event": "model_download_complete", "snapshot": Path(snapshot).name}, flush=True)
            encoder = FrozenVisualEncoder(specification, revision=REVISION, device="cpu", model_source=snapshot)
            print({"event": "model_cpu_ready"}, flush=True)
            encoder.model.to("cuda:0")
            encoder.device = "cuda:0"
            torch.cuda.reset_peak_memory_stats(0)
            print({"event": "model_cuda_ready", "allocated_bytes": torch.cuda.memory_allocated(0)}, flush=True)
            return encoder
        predict(
            args.model_input,
            args.output,
            SOURCE,
            load_encoder,
            cohort=args.cohort,
            recovery_bundle=args.recovery_bundle,
            recovery_manifest=args.recovery_manifest,
        )
        print({"event": "gpu_summary", "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)}, flush=True)
    elif args.stage == "evaluate":
        if args.evaluator is None or args.predictions is None:
            parser.error("Evaluation requires --predictions and --evaluator")
        from visual_peer_inspector.workflows.comparison_evaluation import evaluate
        print(evaluate(args.model_input, args.predictions, args.evaluator, args.output).to_string(index=False))
    else:
        print(verify_seal(args.output, name="prediction_lock.json" if (args.output / "prediction_lock.json").exists() else "artifact_lock.json"))


if __name__ == "__main__":
    main()
