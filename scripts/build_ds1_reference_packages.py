#!/usr/bin/env python3
"""Build DS1 trusted geometry and protected evaluator attributes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.evaluation.proposal_reference_ds1 import build_ds1_reference_packages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--development-source-map", type=Path, required=True)
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--required-source-members", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--geometry-output-root", type=Path, required=True)
    parser.add_argument("--attributes-output-root", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_ds1_reference_packages(
        model_manifest_path=args.model_manifest,
        development_source_map_path=args.development_source_map,
        development_labels_path=args.development_labels,
        required_source_members_path=args.required_source_members,
        source_archive_path=args.source_archive,
        archive_index_path=args.archive_index,
        geometry_output_root=args.geometry_output_root,
        attributes_output_root=args.attributes_output_root,
        mask_threshold=args.mask_threshold,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
