#!/usr/bin/env python3
"""Upload aggregate evidence to ClearML after scientific outputs are sealed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ALLOWED_FILES = {
    "method_summary.csv",
    "view_schedule_sensitivity.csv",
    "fold_summary.csv",
    "category_summary.csv",
    "paired_asset_intervals.csv",
    "risk_coverage.csv",
    "failure_waterfall.csv",
    "method_registry.csv",
    "feature_schema.csv",
    "hybrid_coefficients.csv",
    "development_decision.json",
    "evaluation_receipt.json",
    "preparation_receipt.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shareable-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    args = parser.parse_args()
    receipt = {"status": "disabled"}
    has_environment_keys = bool(
        os.environ.get("CLEARML_API_ACCESS_KEY") and os.environ.get("CLEARML_API_SECRET_KEY")
    )
    has_local_config = (Path.home() / "clearml.conf").is_file()
    if has_environment_keys or has_local_config:
        try:
            from clearml import Task

            present = {path.name for path in args.shareable_root.iterdir() if path.is_file()}
            unexpected = sorted(present - ALLOWED_FILES)
            if unexpected:
                raise ValueError(f"Unexpected aggregate file for ClearML: {unexpected}")
            task = Task.init(
                project_name="Visual Peer Inspector/Development Scoring",
                task_name=args.task_name,
                reuse_last_task_id=False,
                auto_connect_frameworks=False,
                auto_connect_arg_parser=False,
            )
            logger = task.get_logger()
            for path in sorted(args.shareable_root.glob("*.csv")):
                frame = pd.read_csv(path)
                logger.report_table(title=path.stem, series="verified aggregate", table_plot=frame)
                task.upload_artifact(path.stem, artifact_object=frame)
            for path in sorted((args.shareable_root / "figures").glob("*.png")):
                logger.report_image(title=path.stem, series="verified figure", iteration=0, local_path=str(path))
            decision_path = args.shareable_root / "development_decision.json"
            if decision_path.is_file():
                task.connect(json.loads(decision_path.read_text()), name="development decision")
            task.mark_completed()
            task.close()
            receipt = {"status": "completed", "task_id": task.id, "task_name": args.task_name}
        except Exception as error:
            receipt = {"status": "failed", "error_type": type(error).__name__, "message": str(error)[:500]}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    print(json.dumps({"event": "clearml_finalization", **receipt}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
