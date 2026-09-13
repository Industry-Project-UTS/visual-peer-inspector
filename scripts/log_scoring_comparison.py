"""Upload only approved aggregate files after computation has finished."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_peer_inspector.workflows.comparison_io import atomic_json, digest_file, read_json, verify_seal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--notebook", choices=["08G", "08H"], required=True)
    args = parser.parse_args()
    from clearml import Task
    import pandas as pd
    source = args.evidence
    if args.notebook == "08G":
        verify_seal(source, name="prediction_lock.json")
        record = read_json(source / "prediction_lock.json")
        public = {"scene_count": len(record["metadata"]["scenes"]), "prediction_lock_sha256": digest_file(source / "prediction_lock.json"), "producer": record["metadata"]["producer"], "scope": "development comparison"}
        verify_seal(source, name="preview_lock.json")
        safe_files = [source / "shareable/prediction_runtime.csv", source / "shareable/prediction_runtime.png"]
    else:
        verify_seal(source)
        public = read_json(source / "shareable/evaluation_receipt.json")
        safe_files = sorted((source / "shareable").glob("*.csv")) + sorted((source / "shareable").glob("*.png"))
    task = Task.init(project_name="Visual Peer Inspector/Development Scoring", task_name=f"{args.notebook.lower()}-scoring-comparison", task_type=Task.TaskTypes.testing, reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False, auto_connect_streams=False, auto_resource_monitoring=False)
    task.set_tags([args.notebook, "development-only", "pending-review"])
    task.connect(public, name="verified_evidence")
    logger = task.get_logger()
    for path in safe_files:
        if path.suffix == ".csv":
            table = pd.read_csv(path)
            logger.report_table(title=path.stem, series="development", table_plot=table, iteration=0)
            if path.stem == "ranking_summary":
                for row in table.to_dict("records"):
                    series = f"{row['route']}/{row['identity']}/{row['view_count']}views/{row['method']}"
                    for metric in ("ap", "top1", "recall_at_2", "auroc"):
                        logger.report_scalar(title=metric, series=series, value=row[metric], iteration=0)
        else:
            logger.report_image(title="Aggregate comparison", series=path.stem, local_path=str(path), iteration=0)
        task.upload_artifact(name=path.name, artifact_object=str(path), wait_on_upload=True)
    logger.flush()
    task_id = task.id
    task.close()
    atomic_json(source / "tracking_receipt.json", {"status": "sdk_upload_completed", "task_id": task_id, "notebook": args.notebook, "files": [p.name for p in safe_files], "remote_readback": "required before final tag"})
    print("ClearML upload completed. Verify artifacts in the project before tagging the run final.")


if __name__ == "__main__":
    main()
