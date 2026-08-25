#!/usr/bin/env python3
"""Run Notebook 06.1's evaluator workflow outside Jupyter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.workflows.mask_evaluation_ds1 import run_ds1_mask_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--proposal-output-root", type=Path, required=True)
    parser.add_argument("--sam-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=32_040)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = json.loads(args.provenance_json.read_text(encoding="utf-8"))
    record = run_ds1_mask_evaluation(
        evaluator_root=args.evaluator_root,
        proposal_output_root=args.proposal_output_root,
        sam_output_root=args.sam_output_root,
        output_root=args.output_root,
        evaluator_provenance=provenance,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
