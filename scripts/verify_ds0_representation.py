#!/usr/bin/env python3
"""Verify a complete local or downloaded Notebook 07 output directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from visual_peer_inspector.workflows.representation_ds0 import (
    verify_artifact_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--require-model-evidence", action="store_true"
    )
    return parser.parse_args()


def verify_output(root: Path, *, require_model_evidence: bool) -> dict:
    root = root.expanduser().resolve(strict=True)
    manifests = {
        name: verify_artifact_manifest(root / name)
        for name in ("safe", "private", "model_input", "cache")
    }
    run_record = json.loads(
        (root / "safe" / "run_record.json").read_text(encoding="utf-8")
    )
    if run_record.get("status") != "completed":
        raise ValueError("Representation run is not complete")
    if not run_record.get("all_completion_gates_passed"):
        raise ValueError("Representation completion gates did not all pass")
    if require_model_evidence and run_record.get("run_mode") != "model_evidence":
        raise ValueError("Downloaded output is not a model-evidence run")
    package = json.loads(
        (root / "model_input" / "package_record.json").read_text(encoding="utf-8")
    )
    if package.get("scene_count") != 20 or package.get("image_count") != 400:
        raise ValueError("Representation package does not contain complete DS0 coverage")
    figures = sorted((root / "safe" / "figures").glob("*.png"))
    if len(figures) < 8:
        raise ValueError("Safe output is missing aggregate charts")
    for path in figures:
        with Image.open(path) as image:
            image.verify()
        if path.stat().st_size == 0:
            raise ValueError("Aggregate chart is empty")
    gates = list(
        csv.DictReader(
            (root / "safe" / "completion_gates.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    if not gates or any(str(row["passed"]).lower() != "true" for row in gates):
        raise ValueError("Completion gate table contains a failed gate")
    return {
        "status": "verified",
        "run_mode": run_record["run_mode"],
        "available_encoders": package["available_encoders"],
        "score_row_count": package["score_row_count"],
        "evaluation_row_count": run_record["evaluation_row_count"],
        "artifact_counts": {
            name: record["artifact_count"] for name, record in manifests.items()
        },
        "safe_chart_count": len(figures),
    }


def main() -> None:
    args = parse_args()
    result = verify_output(
        args.output_root,
        require_model_evidence=args.require_model_evidence,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
