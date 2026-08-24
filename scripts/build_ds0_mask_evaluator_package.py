#!/usr/bin/env python3
"""Build the private DS0 package used to evaluate SAM masks."""

from __future__ import annotations

import argparse
from pathlib import Path

from visual_peer_inspector.evaluation.sam_mask_package import (
    build_ds0_mask_evaluator_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an evaluator-only DS0 package containing supplied masks, the "
            "private SAM crosswalk and protected scene attributes."
        )
    )
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--crosswalk-root", type=Path, required=True)
    parser.add_argument("--attributes-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_ds0_mask_evaluator_package(
        geometry_root=args.geometry_root,
        crosswalk_root=args.crosswalk_root,
        attributes_root=args.attributes_root,
        source_archive_path=args.source_archive,
        archive_index_path=args.archive_index,
        output_root=args.output_root,
    )
    print(
        "Private DS0 mask evaluator package verified: "
        f"{record['scene_count']} scenes, {record['image_count']} views, "
        f"{record['visible_mask_count']} supplied masks and "
        f"{record['crosswalk_row_count']} crosswalk rows"
    )
    print("This package is evaluator-only and must not be used as model input.")


if __name__ == "__main__":
    main()
