#!/usr/bin/env python3
"""Build the private DS1 SAM prompt-to-reference crosswalk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.evaluation.sam_crosswalk_ds1 import build_ds1_sam_evaluator_crosswalk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--proposal-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_ds1_sam_evaluator_crosswalk(
        prompt_root=args.prompt_root,
        geometry_root=args.geometry_root,
        proposal_results_path=args.proposal_results,
        output_root=args.output_root,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
