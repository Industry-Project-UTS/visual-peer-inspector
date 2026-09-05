#!/usr/bin/env python3
"""Build the separated DS2 RGB, geometry, and evaluator packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    build_cohort_rgb_package,
    verify_model_input_package,
)
from visual_peer_inspector.evaluation.confirmation_reference import (
    build_ds2_reference_packages,
    verify_ds2_attributes_package,
    verify_ds2_geometry_package,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--administrative-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-archive-byte-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_root = args.output_root.expanduser().resolve()
    rgb_root = output_root / "model-input"
    geometry_root = output_root / "oracle-geometry"
    attributes_root = output_root / "evaluator-attributes"

    rgb_audit, build_audit = build_cohort_rgb_package(
        selection_root=args.selection_root,
        development_root=args.development_root,
        administrative_root=args.administrative_root,
        source_root=args.source_root,
        output_root=output_root,
        contract=DS2_CONFIRM_CONTRACT,
        verify_archive_bytes=not args.skip_archive_byte_check,
    )
    if geometry_root.exists() or attributes_root.exists():
        geometry_record = verify_ds2_geometry_package(geometry_root)
        attributes_record = verify_ds2_attributes_package(attributes_root)
    else:
        records = build_ds2_reference_packages(
            model_manifest_path=rgb_root / "neutral_corpus_manifest.jsonl",
            development_source_map_path=(
                args.development_root / "development_source_map.jsonl"
            ),
            development_labels_path=args.development_root / "development_labels.jsonl",
            required_source_members_path=(
                args.administrative_root / "required_source_members.jsonl"
            ),
            source_archive_path=args.source_root / "toysAD8K.zip",
            archive_index_path=args.source_root / "archive_index.sqlite3",
            geometry_output_root=geometry_root,
            attributes_output_root=attributes_root,
        )
        geometry_record = records["geometry"]
        attributes_record = records["attributes"]

    verified_rgb = verify_model_input_package(
        rgb_root,
        contract=DS2_CONFIRM_CONTRACT,
    )
    record = {
        "schema_version": 1,
        "stage": "ds2_confirmation_input_release",
        "status": "completed",
        "cohort": {
            key: getattr(DS2_CONFIRM_CONTRACT, key)
            for key in (
                "cohort_id",
                "release_id",
                "role",
                "pilot_stage",
                "membership_sha256",
                "scene_count",
                "views_per_scene",
                "image_count",
                "asset_group_count",
            )
        },
        "rgb_package": verified_rgb.to_dict(),
        "rgb_build": build_audit,
        "geometry_package": geometry_record,
        "attributes_package": attributes_record,
        "boundary": {
            "model_input_contains_labels": False,
            "geometry_contains_anomaly_labels": False,
            "evaluator_attributes_model_facing": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
