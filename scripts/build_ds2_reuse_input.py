#!/usr/bin/env python3
"""Build the private DS2 reuse input consumed by Notebook 08E."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.confirmation_mask_routes import (
    build_ds2_mask_route_package,
)
from visual_peer_inspector.data.ds2_reuse_input import (
    PACKAGE_KIND,
    checksum_inventory,
    verify_ds2_reuse_input,
)
from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.sam2 import Sam2Configuration
from visual_peer_inspector.workflows.locateanything_confirmation_ds2 import (
    load_frozen_ds1_configuration,
    resolve_ds2_operational_policy,
)

FROZEN_POLICY = "area_nms_075_cap8"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--selected-prompt", type=Path, required=True)
    parser.add_argument("--locate-run-record", type=Path, required=True)
    parser.add_argument("--automatic-sam-run-record", type=Path, required=True)
    parser.add_argument("--reference-sam-run-record", type=Path, required=True)
    parser.add_argument("--automatic-sam-results", type=Path, required=True)
    parser.add_argument("--reference-sam-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_record(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def write_record(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_fields(record: dict, fields: set[str], name: str) -> None:
    missing = sorted(fields.difference(record))
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")


def validate_source_records(args: argparse.Namespace) -> dict:
    selected_prompt = args.selected_prompt.expanduser().resolve()
    locate_path = args.locate_run_record.expanduser().resolve()
    automatic_record_path = args.automatic_sam_run_record.expanduser().resolve()
    reference_record_path = args.reference_sam_run_record.expanduser().resolve()
    automatic_results = args.automatic_sam_results.expanduser().resolve()
    reference_results = args.reference_sam_results.expanduser().resolve()

    frozen_configuration, policy, selected_prompt_sha256 = (
        load_frozen_ds1_configuration(selected_prompt)
    )
    effective_configuration, _execution_policy = resolve_ds2_operational_policy(
        frozen_configuration
    )
    if policy != FROZEN_POLICY:
        raise ValueError(f"DS2 reuse requires the frozen {FROZEN_POLICY} policy")

    locate = read_record(locate_path)
    automatic = read_record(automatic_record_path)
    reference = read_record(reference_record_path)
    require_fields(
        locate,
        {
            "schema_version",
            "stage",
            "status",
            "cohort_id",
            "image_count",
            "ds1_freeze_sha256",
            "ds1_generation_configuration_sha256",
            "effective_generation_configuration_sha256",
            "proposal_package_sha256",
        },
        "LocateAnything run record",
    )
    sam_fields = {
        "schema_version",
        "stage",
        "status",
        "route",
        "cohort_id",
        "image_count",
        "configuration_sha256",
        "prompt_package_sha256",
        "sam_results_sha256",
    }
    require_fields(automatic, sam_fields, "automatic SAM run record")
    require_fields(reference, sam_fields, "reference SAM run record")

    cohort = DS2_CONFIRM_CONTRACT.cohort_id
    sam_configuration_sha256 = Sam2Configuration(
        device="cuda", precision="float16"
    ).sha256
    if (
        locate["schema_version"] != 1
        or locate["stage"] != "ds2_frozen_locateanything"
        or locate["status"] != "completed"
        or locate["cohort_id"] != cohort
        or int(locate["image_count"]) != DS2_CONFIRM_CONTRACT.image_count
        or locate["ds1_freeze_sha256"] != selected_prompt_sha256
        or locate["ds1_generation_configuration_sha256"] != frozen_configuration.sha256
        or locate["effective_generation_configuration_sha256"]
        != effective_configuration.sha256
    ):
        raise ValueError(
            "LocateAnything run identity differs from the frozen DS2 route"
        )
    for record, route in ((automatic, "automatic_box"), (reference, "reference_box")):
        if (
            record["schema_version"] != 1
            or record["status"] != "completed"
            or record["route"] != route
            or record["cohort_id"] != cohort
            or int(record["image_count"]) != DS2_CONFIRM_CONTRACT.image_count
            or record["configuration_sha256"] != sam_configuration_sha256
        ):
            raise ValueError(f"{route} SAM identity differs from the frozen DS2 route")
    if automatic["prompt_package_sha256"] != locate["proposal_package_sha256"]:
        raise ValueError(
            "Automatic SAM does not reference the verified proposal package"
        )
    if automatic["sam_results_sha256"] != file_sha256(automatic_results):
        raise ValueError("Automatic SAM results differ from their recorded digest")
    if reference["sam_results_sha256"] != file_sha256(reference_results):
        raise ValueError("Reference SAM results differ from their recorded digest")

    return {
        "selected_prompt_sha256": selected_prompt_sha256,
        "locate_run_record_sha256": file_sha256(locate_path),
        "automatic_sam_run_record_sha256": file_sha256(automatic_record_path),
        "reference_sam_run_record_sha256": file_sha256(reference_record_path),
        "automatic_sam_results_sha256": file_sha256(automatic_results),
        "reference_sam_results_sha256": file_sha256(reference_results),
        "proposal_package_sha256": locate["proposal_package_sha256"],
        "locate_configuration_sha256": effective_configuration.sha256,
        "sam_configuration_sha256": sam_configuration_sha256,
        "model_id": effective_configuration.model_id,
        "model_revision": effective_configuration.model_revision,
        "prompt": effective_configuration.prompt,
        "postprocessing_policy": policy,
    }


def main() -> None:
    args = parse_arguments()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"DS2 reuse output already exists: {output_root}")
    identity = validate_source_records(args)

    mask_root = output_root / "mask-routes"
    mask_record = build_ds2_mask_route_package(
        rgb_root=args.rgb_root,
        geometry_root=args.geometry_root,
        sam_results_path=args.reference_sam_results,
        automatic_sam_results_path=args.automatic_sam_results,
        output_root=mask_root,
        provenance={
            "selected_prompt_sha256": identity["selected_prompt_sha256"],
            "locate_run_record_sha256": identity["locate_run_record_sha256"],
            "automatic_sam_run_record_sha256": identity[
                "automatic_sam_run_record_sha256"
            ],
            "reference_sam_run_record_sha256": identity[
                "reference_sam_run_record_sha256"
            ],
            "automatic_sam_results_sha256": identity["automatic_sam_results_sha256"],
            "reference_sam_results_sha256": identity["reference_sam_results_sha256"],
            "proposal_package_sha256": identity["proposal_package_sha256"],
            "geometry_record_sha256": file_sha256(
                args.geometry_root / "package_record.json"
            ),
            "rgb_record_sha256": file_sha256(args.rgb_root / "package_record.json"),
        },
    )
    copies = {
        args.selected_prompt: output_root / "selected_prompt.json",
        args.locate_run_record: output_root / "locateanything_run_record.json",
        args.automatic_sam_run_record: output_root / "automatic_sam_run_record.json",
        args.reference_sam_run_record: output_root / "reference_sam_run_record.json",
    }
    for source, destination in copies.items():
        shutil.copyfile(source.expanduser().resolve(), destination)

    checksum_path = output_root / "content_checksums.sha256"
    checksum_path.write_text(
        "\n".join(checksum_inventory(output_root)) + "\n", encoding="utf-8"
    )
    package_record = {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "release_id": DS2_CONFIRM_CONTRACT.release_id,
        "membership_sha256": DS2_CONFIRM_CONTRACT.membership_sha256,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "image_count": DS2_CONFIRM_CONTRACT.image_count,
        "views_per_scene": DS2_CONFIRM_CONTRACT.views_per_scene,
        **identity,
        "mask_route_package_record_sha256": file_sha256(
            mask_root / "package_record.json"
        ),
        "mask_routes_sha256": mask_record["mask_routes_sha256"],
        "content_checksums_sha256": file_sha256(checksum_path),
        "all_checks_passed": True,
    }
    write_record(output_root / "package_record.json", package_record)
    print(json.dumps(verify_ds2_reuse_input(output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
