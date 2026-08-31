#!/usr/bin/env python3
"""Run Notebook 07.1 as a resumable command-line workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from visual_peer_inspector.workflows.representation_ds1 import (
    DEFAULT_ENCODER_BATCH_SIZE,
    DEFAULT_ENCODERS,
    run_ds1_representation_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--mask-route-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encoder", action="append", dest="encoders")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ENCODER_BATCH_SIZE)
    parser.add_argument(
        "--run-mode",
        choices=("model_evidence", "contract_test"),
        default="model_evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoders = tuple(args.encoders or DEFAULT_ENCODERS)
    if args.run_mode == "contract_test" and args.encoders is None:
        encoders = ("contract_encoder",)
    result = run_ds1_representation_pipeline(
        rgb_root=args.rgb_root,
        mask_route_root=args.mask_route_root,
        evaluator_root=args.evaluator_root,
        output_root=args.output_root,
        encoder_keys=encoders,
        device=args.device,
        hf_token=os.environ.get("HF_TOKEN"),
        run_mode=args.run_mode,
        encoder_batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
