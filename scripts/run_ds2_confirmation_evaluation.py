#!/usr/bin/env python3
"""Run protected DS2 confirmation and multi-view evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.workflows.multiview_confirmation_ds2 import (
    run_ds2_protected_evaluation,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--attributes-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    record = run_ds2_protected_evaluation(
        rgb_root=args.rgb_root,
        geometry_root=args.geometry_root,
        attributes_root=args.attributes_root,
        sam_results_path=args.inference_root / "sam21" / "reference-box" / "private" / "sam_results.jsonl",
        automatic_sam_results_path=args.inference_root / "sam21" / "automatic-box" / "private" / "sam_results.jsonl",
        representation_root=args.inference_root / "representation",
        output_root=args.output_root,
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
