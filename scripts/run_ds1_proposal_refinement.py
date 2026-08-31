#!/usr/bin/env python3
"""Run the bounded DS1 proposal-refinement experiment outside Jupyter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from visual_peer_inspector.workflows.proposal_refinement_ds1 import (
    run_ds1_proposal_refinement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=32_040)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = run_ds1_proposal_refinement(
        proposal_root=args.proposal_root,
        evaluator_root=args.evaluator_root,
        rgb_root=args.rgb_root,
        output_root=args.output_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
