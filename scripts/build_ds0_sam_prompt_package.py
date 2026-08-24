#!/usr/bin/env python3
"""Build the model-safe DS0 prompt bridge for Notebook 05."""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_peer_inspector.data.sam_prompt_package import build_sam_prompt_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--proposal-results", type=Path, required=True)
    parser.add_argument("--proposal-decision-record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_sam_prompt_package(
        model_manifest_path=args.model_manifest,
        media_root=args.media_root,
        geometry_manifest_path=args.geometry_manifest,
        proposal_results_path=args.proposal_results,
        proposal_decision_record_path=args.proposal_decision_record,
        output_root=args.output_root,
    )
    print(
        "DS0 SAM prompt package verified: "
        f"{record['image_count']} images, "
        f"{record['reference_prompt_count']} reference prompts, "
        f"{record['automatic_prompt_count']} automatic prompts"
    )


if __name__ == "__main__":
    main()
