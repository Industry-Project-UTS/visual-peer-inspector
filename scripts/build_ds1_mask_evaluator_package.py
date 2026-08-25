#!/usr/bin/env python3
"""Build the private DS1 proposal and mask evaluator package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.evaluation.ds1_evaluator_package import build_ds1_evaluator_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--protected-attributes", type=Path, required=True)
    parser.add_argument("--evaluator-crosswalk", type=Path, required=True)
    parser.add_argument("--crosswalk-package-record", type=Path, required=True)
    parser.add_argument("--source-mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = json.loads(args.provenance_json.read_text(encoding="utf-8"))
    record = build_ds1_evaluator_package(
        reference_manifest_path=args.reference_manifest,
        protected_attributes_path=args.protected_attributes,
        evaluator_crosswalk_path=args.evaluator_crosswalk,
        crosswalk_package_record_path=args.crosswalk_package_record,
        source_mask_root=args.source_mask_root,
        output_root=args.output_root,
        provenance=provenance,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
