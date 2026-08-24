#!/usr/bin/env python3
"""Build the protected SAM evaluator bridge for Notebook 06."""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_peer_inspector.evaluation.sam_crosswalk import (
    build_sam_evaluator_crosswalk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--proposal-results", type=Path, required=True)
    parser.add_argument("--proposal-matches", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_sam_evaluator_crosswalk(
        prompt_root=args.prompt_root,
        geometry_manifest_path=args.geometry_manifest,
        proposal_results_path=args.proposal_results,
        proposal_matches_path=args.proposal_matches,
        output_root=args.output_root,
    )
    print(
        "Protected SAM evaluator bridge verified: "
        f"{record['row_count']} prompt records"
    )


if __name__ == "__main__":
    main()
