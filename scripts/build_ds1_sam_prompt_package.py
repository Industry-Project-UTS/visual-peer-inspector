#!/usr/bin/env python3
"""Build the frozen DS1 prompt package used by Notebook 05.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.data.sam_prompt_package_ds1 import (
    build_ds1_sam_prompt_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-package-record", type=Path, required=True)
    parser.add_argument("--proposal-results", type=Path, required=True)
    parser.add_argument("--proposal-run-record", type=Path, required=True)
    parser.add_argument("--proposal-artifact-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = build_ds1_sam_prompt_package(
        model_input_root=args.model_input_root,
        geometry_manifest_path=args.geometry_manifest,
        geometry_package_record_path=args.geometry_package_record,
        proposal_results_path=args.proposal_results,
        proposal_run_record_path=args.proposal_run_record,
        proposal_artifact_manifest_path=args.proposal_artifact_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
