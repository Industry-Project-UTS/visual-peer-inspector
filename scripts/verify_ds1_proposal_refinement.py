#!/usr/bin/env python3
"""Verify a complete local or downloaded Notebook 04.2 output directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from visual_peer_inspector.workflows.proposal_refinement_ds1 import (
    verify_artifact_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def verify_output(root: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    manifests = {
        name: verify_artifact_manifest(root / name)
        for name in ("safe", "private", "model_input")
    }
    run_record = json.loads(
        (root / "safe" / "run_record.json").read_text(encoding="utf-8")
    )
    if run_record.get("stage") != "ds1_proposal_refinement":
        raise ValueError("Output is not a DS1 proposal-refinement run")
    if run_record.get("status") != "completed":
        raise ValueError("Proposal-refinement run is not complete")
    if not run_record.get("all_completion_gates_passed"):
        raise ValueError("Proposal-refinement completion gates did not all pass")
    package = json.loads(
        (root / "model_input" / "package_record.json").read_text(encoding="utf-8")
    )
    if package.get("scene_count") != 100 or package.get("image_count") != 2_000:
        raise ValueError("Proposal-refinement package does not contain complete DS1 coverage")
    if package.get("policy_image_row_count") != 10_000:
        raise ValueError("Proposal-refinement package has incomplete policy coverage")
    gates = list(
        csv.DictReader(
            (root / "safe" / "completion_gates.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    if not gates or any(str(row["passed"]).lower() != "true" for row in gates):
        raise ValueError("Proposal-refinement completion table contains a failed gate")
    charts = sorted((root / "safe" / "figures").glob("*.png"))
    if len(charts) != 3:
        raise ValueError("Proposal-refinement safe output must contain three charts")
    for path in charts:
        with Image.open(path) as image:
            image.verify()
        if path.stat().st_size == 0:
            raise ValueError("Proposal-refinement chart is empty")
    private_panel = root / "private" / "figures" / "baseline_vs_selected_boxes.png"
    with Image.open(private_panel) as image:
        image.verify()
    return {
        "status": "verified",
        "scene_count": package["scene_count"],
        "image_count": package["image_count"],
        "policy_count": package["policy_count"],
        "selected_diagnostic_policy": run_record["selected_diagnostic_policy"],
        "automatic_proposal_route_accepted": run_record[
            "automatic_proposal_route_accepted"
        ],
        "artifact_counts": {
            name: record["artifact_count"] for name, record in manifests.items()
        },
        "safe_chart_count": len(charts),
    }


def main() -> None:
    result = verify_output(parse_args().output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
