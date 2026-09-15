"""Upload verified aggregate 08J evidence after computation has completed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    from clearml import Task
    import pandas as pd

    from visual_peer_inspector.workflows.comparison_io import atomic_json, read_json, verify_seal

    root = args.evidence.resolve()
    verify_seal(root)
    receipt = read_json(root / "shareable" / "evaluation_receipt.json")
    decision = read_json(root / "shareable" / "development_decision.json")
    safe_files = sorted((root / "shareable").glob("*.csv"))
    safe_files += sorted((root / "shareable").glob("*.json"))
    safe_files += sorted((root / "shareable" / "figures").glob("*.png"))
    if not safe_files:
        raise FileNotFoundError("No verified shareable evidence was found")

    task = Task.init(
        project_name="Visual Peer Inspector/Development Identity",
        task_name="bounded-association-study-08j",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_connect_streams=False,
        auto_resource_monitoring=False,
    )
    task.set_tags(["development-only", "dinov2-small", "identity-association", decision["status"]])
    task.connect({"receipt": receipt, "decision": decision}, name="verified_evidence")
    logger = task.get_logger()
    for path in safe_files:
        if path.suffix == ".csv":
            table = pd.read_csv(path)
            logger.report_table(title=path.stem, series="development", table_plot=table, iteration=0)
            if path.name == "primary_candidate_ranking.csv":
                for row in table.to_dict("records"):
                    for metric in ("ap", "top1", "recall_at_2", "identity_link_f1"):
                        logger.report_scalar(title=metric, series=str(row["configuration_id"]), value=float(row[metric]), iteration=0)
        elif path.suffix == ".png":
            logger.report_image(title="Identity association", series=path.stem, local_path=str(path), iteration=0)
        task.upload_artifact(name=path.name, artifact_object=str(path), wait_on_upload=True)
    logger.flush()
    task_id = task.id
    task.close()
    atomic_json(
        root / "tracking_receipt.json",
        {
            "status": "sdk_upload_completed",
            "task_id": task_id,
            "files": [path.relative_to(root).as_posix() for path in safe_files],
            "remote_readback": "required before final tagging",
        },
    )
    print(f"ClearML upload completed for task {task_id}")


if __name__ == "__main__":
    main()
