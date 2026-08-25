#!/usr/bin/env python3
"""Build the clean Notebook 06.1 source from reviewed cells."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "detection-and-segmentation" / "toysad8k-ds1-mask-evaluation"
NOTEBOOK_PATH = NOTEBOOK_DIR / "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K DS1 Proposal and Mask Evaluation

**Notebook file:** `06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb`

This notebook evaluates the frozen LocateAnything and SAM 2.1 outputs on the complete DS1 development screen. DS1 contains 100 scenes, 20 views per scene and 2,000 images.

It answers four questions:

1. Does LocateAnything find the expected objects with useful box accuracy?
2. Does SAM 2.1 preserve the objects when prompted by reference boxes?
3. How much quality is lost when SAM uses automatic boxes?
4. Is the detection and segmentation path ready for the next development stage?

This is evaluator-only work. Models have already finished before this notebook opens supplied masks, source crosswalks or protected subgroup fields.
"""
    ),
    markdown(
        """
## Evaluation boundary

The notebook reads three immutable inputs: the private DS1 evaluator package, the complete LocateAnything output and the complete SAM output. Before any metric is calculated, it verifies both model run records, both artifact manifests, the evaluator crosswalk, every prompt box, every RGB hash and the full upstream hash chain. It does not rerun either model and does not change predictions after seeing the labels.

Detailed object records stay under `private`. GitHub and ClearML receive aggregate tables, charts, hashes and the component decision. A separate `model_input` folder contains the supplied-mask, reference-box and automatic-box routes without source IDs, labels, roles, anomaly fields or protected ordering.
"""
    ),
    code(
        """
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from IPython.display import Image as DisplayImage, display
import pandas as pd

NOTEBOOK_NAME = "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb"
ENVIRONMENT = "kaggle" if Path("/kaggle/working").is_dir() else "local"


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    if Path("/kaggle/input").is_dir():
        candidates.extend(path.parents[3] for path in Path("/kaggle/input").rglob("mask_evaluation_ds1.py"))
    for candidate in candidates:
        marker = candidate / "src" / "visual_peer_inspector" / "workflows" / "mask_evaluation_ds1.py"
        if marker.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the Notebook 06.1 code snapshot")


CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.verification import verify_code_snapshot_record

snapshot_path = CODE_ROOT / "snapshot_record.json"
snapshot = verify_code_snapshot_record(CODE_ROOT, snapshot_path)
print(f"Environment: {ENVIRONMENT}")
print(f"Notebook: {NOTEBOOK_NAME}")
print(f"Code snapshot: {snapshot['snapshot_sha256']}")
"""
    ),
    markdown(
        """
## Input paths and frozen settings

Attach the evaluator package and the downloaded outputs from Notebooks 04.1 and 05.1 as private Kaggle datasets. Each input is found through a unique internal marker rather than a mutable folder name. Bootstrap intervals resample the frozen base-asset clusters and keep every view in its sampled cluster.
"""
    ),
    code(
        """
def find_one(pattern: str, required_file: str) -> Path:
    configured = os.getenv(pattern)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / required_file).is_file() and not (root / "private" / required_file).is_file():
            raise FileNotFoundError(f"Configured input does not contain {required_file}")
        return root
    search_root = Path("/kaggle/input") if ENVIRONMENT == "kaggle" else CODE_ROOT
    candidates = set()
    for path in search_root.rglob(required_file):
        candidates.add(path.parent.parent if path.parent.name == "private" else path.parent)
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one input containing {required_file}, found {len(candidates)}")
    return candidates.pop().resolve()


EVALUATOR_ROOT = find_one("VPI_DS1_EVALUATOR_ROOT", "evaluator_crosswalk.jsonl")
PROPOSAL_ROOT = find_one("VPI_DS1_PROPOSAL_ROOT", "proposal_results.jsonl")
SAM_ROOT = find_one("VPI_DS1_SAM_ROOT", "mask_results.jsonl")
OUTPUT_ROOT = Path("/kaggle/working/toysad8k_ds1_mask_evaluation" if ENVIRONMENT == "kaggle" else "outputs/toysad8k_ds1_mask_evaluation").resolve()
BOOTSTRAP_REPETITIONS = int(os.getenv("VPI_BOOTSTRAP_REPETITIONS", "10000"))
BOOTSTRAP_SEED = 32040

paths = pd.DataFrame([
    {"input": "private evaluator package", "folder": EVALUATOR_ROOT.name},
    {"input": "LocateAnything output", "folder": PROPOSAL_ROOT.name},
    {"input": "SAM 2.1 output", "folder": SAM_ROOT.name},
])
display(paths)
"""
    ),
    markdown(
        """
## Run the frozen evaluator

The workflow checks all input coverage and upstream identities before calculating metrics. Proposal matching uses one-to-one assignment at box IoU 0.50. Mask quality reports IoU, Dice, boundary F1, detection-aware recall at mask IoU 0.75 and detection-aware mean IoU. Misses, extras, merge candidates and split candidates remain in the denominator.
"""
    ),
    code(
        """
from visual_peer_inspector.workflows.mask_evaluation_ds1 import run_ds1_mask_evaluation

provenance = {
    "notebook_name": NOTEBOOK_NAME,
    "execution_environment": ENVIRONMENT,
    "git_commit": snapshot["git_commit"],
    "git_dirty": snapshot["git_dirty"],
    "snapshot_sha256": snapshot["snapshot_sha256"],
    "code_dataset_version": snapshot["platform_versions"]["code_dataset_version"],
}

run_record = run_ds1_mask_evaluation(
    evaluator_root=EVALUATOR_ROOT,
    proposal_output_root=PROPOSAL_ROOT,
    sam_output_root=SAM_ROOT,
    output_root=OUTPUT_ROOT,
    evaluator_provenance=provenance,
    bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
    bootstrap_seed=BOOTSTRAP_SEED,
)
display(pd.DataFrame([run_record["coverage"]]))
"""
    ),
    markdown("## Proposal metrics"),
    code(
        """
proposal_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "proposal_summary.csv")
proposal_subgroups = pd.read_csv(OUTPUT_ROOT / "safe" / "proposal_subgroups.csv")
proposal_intervals = pd.read_csv(OUTPUT_ROOT / "safe" / "proposal_bootstrap_intervals.csv")
display(proposal_summary)
display(proposal_intervals)
display(proposal_subgroups.groupby("dimension", as_index=False)["value"].count().rename(columns={"value": "subgroup_count"}))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "proposal_matched_iou_distribution.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "proposal_precision_recall.png")))
"""
    ),
    markdown("## Mask-route metrics"),
    code(
        """
mask_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "mask_route_summary.csv")
mask_intervals = pd.read_csv(OUTPUT_ROOT / "safe" / "mask_bootstrap_intervals.csv")
display(mask_summary)
display(mask_intervals)
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "reference_box_mask_iou_distribution.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "automatic_box_mask_iou_distribution.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "mask_recall_by_route.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "automatic_spill_vs_mask_iou.png")))
subgroup_intervals = pd.read_csv(OUTPUT_ROOT / "safe" / "subgroup_bootstrap_intervals.csv")
worst_subgroups = pd.read_csv(OUTPUT_ROOT / "safe" / "worst_subgroup_summary.csv")
display(worst_subgroups)
display(subgroup_intervals.groupby(["component", "metric", "eligible"], as_index=False).size())
"""
    ),
    markdown(
        """
## Private qualitative diagnostics

The evaluator deterministically selects a representative image, the lowest automatic-route image and the largest reference-versus-automatic route gap. Each panel compares the supplied mask, reference-box SAM mask and automatic-box SAM mask. Green is missed reference area, pink is extra prediction area and white is overlap. These panels stay in the private output and are never sent to ClearML or GitHub.
"""
    ),
    code(
        """
for diagnostic_kind in ("representative", "worst", "outlier"):
    display(DisplayImage(filename=str(OUTPUT_ROOT / "private" / f"{diagnostic_kind}_mask_panel.png")))
"""
    ),
    markdown("## Runtime, failures and component decision"),
    code(
        """
decision_gates = pd.read_csv(OUTPUT_ROOT / "safe" / "decision_gates.csv")
failures = pd.read_csv(OUTPUT_ROOT / "safe" / "failure_summary.csv")
runtime = json.loads((OUTPUT_ROOT / "safe" / "runtime_summary.json").read_text())
display(decision_gates)
display(failures)
display(pd.DataFrame([runtime]))
display(pd.read_csv(OUTPUT_ROOT / "safe" / "diagnostic_summary.csv"))
print(json.dumps(run_record["component_decision"], indent=2))
"""
    ),
    markdown(
        """
## Bounded ClearML finalisation

Tracking starts only after evaluator compute and artifact manifests have finished. The finaliser verifies the complete aggregate allowlist before it reads ClearML credentials. Kaggle Secrets are loaded only inside that isolated subprocess. Tracking failure writes a receipt and does not remove or invalidate the completed evaluation outputs.
"""
    ),
    code(
        """
tracking_receipt = OUTPUT_ROOT / "tracking_receipt.json"
tracking_result = {"status": "not_started", "clearml_task_id": None}
tracking_environment = os.environ.copy()
existing_pythonpath = tracking_environment.get("PYTHONPATH")
tracking_environment["PYTHONPATH"] = str(CODE_ROOT / "src") + (
    os.pathsep + existing_pythonpath if existing_pythonpath else ""
)
try:
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/finalize_ds1_mask_evaluation_clearml.py"),
            "--safe-root",
            str(OUTPUT_ROOT / "safe"),
            "--receipt",
            str(tracking_receipt),
            "--mode",
            "online",
            "--secret-source",
            "kaggle" if ENVIRONMENT == "kaggle" else "environment",
            "--wall-time-seconds",
            "180",
            "--tag",
            "candidate",
            "--tag",
            "notebook-06.1",
            "--tag",
            "ds1",
            "--tag",
            "locateanything",
            "--tag",
            "sam2.1",
            "--tag",
            "evaluation",
            "--tag",
            "kaggle",
            "--tag",
            "full",
        ],
        env=tracking_environment,
        text=True,
        capture_output=True,
        timeout=240,
        check=True,
    )
    tracking_result = json.loads(tracking_receipt.read_text(encoding="utf-8"))
except Exception as error:
    tracking_result = {
        "schema_version": 1,
        "status": "failed",
        "tracking_mode": "online",
        "clearml_task_id": None,
        "error_type": type(error).__name__,
    }
    tracking_receipt.write_text(
        json.dumps(tracking_result, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
display(tracking_result)
"""
    ),
    markdown(
        """
## Outputs and next use

The `safe` folder contains the report and ClearML evidence. The `private` folder contains evaluator-only object records and must not be committed or uploaded to ClearML. The `model_input` folder is the verified label-free three-route package for Vaibhav's representation pipeline.

The DS1 decision is a development-stage component decision. It does not replace evaluation on the locked internal role or the final official test role.
"""
    ),
    code(
        """
final_summary = {
    "status": run_record["status"],
    "scenes": run_record["cohort"]["scene_count"],
    "images": run_record["coverage"]["sam_images"],
    "component_decision": run_record["component_decision"]["status"],
    "all_component_gates_passed": run_record["component_decision"]["all_gates_passed"],
    "tracking_status": tracking_result["status"],
    "clearml_task_id": tracking_result.get("clearml_task_id"),
    "snapshot_sha256": snapshot["snapshot_sha256"],
}
display(final_summary)
if run_record["status"] != "completed":
    raise RuntimeError("Notebook 06.1 evaluation did not complete")
"""
    ),
]

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
