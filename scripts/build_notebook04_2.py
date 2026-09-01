#!/usr/bin/env python3
"""Build the clean Notebook 04.2 source from reviewed cells."""

from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "notebooks"
    / "detection-and-segmentation"
    / "toysad8k-ds1-proposal-refinement"
    / "04.2-toysad8k-ds1-proposal-refinement.ipynb"
)


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K DS1 Proposal Refinement

**Notebook file:** `04.2-toysad8k-ds1-proposal-refinement.ipynb`

Notebook 04.1 showed that LocateAnything has strong coverage but returns too many extra boxes and still misses some objects. This notebook tests whether a small, label-free geometry policy can improve those proposals without rerunning or training the model.

The experiment covers all 100 DS1 scenes and all 2,000 images. It compares the unchanged baseline with predeclared area and overlap filters, reports paired uncertainty, checks category and scene subgroups, and shows before-and-after examples. It does not tune a rule per image and it does not use labels while filtering.
"""
    ),
    markdown(
        """
## Experiment boundary

The proposal file is frozen before the evaluator opens. Each policy uses only box coordinates, relative area, original model order, and overlap between model boxes. The evaluator package is opened later to calculate precision, recall, F1, count accuracy, failures, paired confidence intervals, and subgroup results.

The acceptance gates are precision at least 0.90 and recall at least 0.95 at IoU 0.50. A filter may be kept as a useful diagnostic even when it cannot recover the objects missed by LocateAnything. DS1 remains development data, so no result in this notebook is a final test claim.
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

NOTEBOOK_NAME = "04.2-toysad8k-ds1-proposal-refinement.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/working").is_dir() else "local"


def find_code_root() -> Path:
    # Find the attached repository snapshot without depending on an account slug.
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    if Path("/kaggle/input").is_dir():
        candidates.extend(path.parents[3] for path in Path("/kaggle/input").rglob("proposal_refinement_ds1.py"))
    for candidate in candidates:
        marker = candidate / "src" / "visual_peer_inspector" / "workflows" / "proposal_refinement_ds1.py"
        if marker.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the Notebook 04.2 code snapshot")


CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
print(f"Environment: {EXECUTION_ENVIRONMENT}")
print(f"Notebook: {NOTEBOOK_NAME}")
print(f"Code snapshot folder: {CODE_ROOT.name}")
"""
    ),
    markdown(
        """
## Locate and verify the frozen inputs

The notebook reuses the existing DS1 RGB package, evaluator package, and completed Notebook 04.1 output. No new dataset is created. Path discovery uses package records and run stages rather than fixed Kaggle mount names.
"""
    ),
    code(
        """
def locate_package(variable: str, predicate) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    if Path("/kaggle/input").is_dir():
        candidates.extend(path.parent for path in Path("/kaggle/input").rglob("package_record.json"))
    for candidate in candidates:
        record_path = candidate / "package_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if predicate(record):
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate required package: {variable}")


def locate_completed_run(variable: str, stage: str, result_name: str) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    if Path("/kaggle/input").is_dir():
        candidates.extend(path.parent.parent for path in Path("/kaggle/input").rglob("safe/run_record.json"))
    for candidate in candidates:
        run_path = candidate / "safe" / "run_record.json"
        result_path = candidate / "private" / result_name
        if not run_path.is_file() or not result_path.is_file():
            continue
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if record.get("stage") == stage and record.get("status") == "completed":
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate completed input: {variable}")


RGB_ROOT = locate_package(
    "VPI_RGB_ROOT",
    lambda record: record.get("release_id") == "toysad8k-train-2000-v2" and record.get("image_count") == 2_000,
)
EVALUATOR_ROOT = locate_package(
    "VPI_EVALUATOR_ROOT",
    lambda record: record.get("package_kind") == "toysad8k-ds1-mask-evaluator",
)
PROPOSAL_ROOT = locate_completed_run(
    "VPI_PROPOSAL_ROOT", "ds1_locateanything_screen", "proposal_results.jsonl"
)
default_output = (
    Path("/kaggle/working/toysad8k_ds1_proposal_refinement")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else CODE_ROOT / ".local" / "runs" / "notebook04_2"
)
OUTPUT_ROOT = Path(os.getenv("VPI_OUTPUT_ROOT", str(default_output))).expanduser().resolve()
BOOTSTRAP_REPETITIONS = int(os.getenv("VPI_BOOTSTRAP_REPETITIONS", "5000"))

display(pd.DataFrame([
    {"input": "DS1 RGB package", "folder": RGB_ROOT.name},
    {"input": "DS1 evaluator package", "folder": EVALUATOR_ROOT.name},
    {"input": "Notebook 04.1 output", "folder": PROPOSAL_ROOT.name},
]))
"""
    ),
    markdown(
        """
## Predeclared policies

The baseline keeps every box. The remaining policies remove extreme-size boxes, suppress strong overlaps, or combine both checks. Non-maximum suppression keeps the first box in the model's original order because LocateAnything does not provide a calibrated confidence score.
"""
    ),
    code(
        """
from visual_peer_inspector.workflows.proposal_refinement_ds1 import POLICIES

display(pd.DataFrame([
    {
        "policy": policy.key,
        "minimum area fraction": policy.minimum_area_fraction,
        "maximum area fraction": policy.maximum_area_fraction,
        "overlap threshold": policy.nms_iou,
    }
    for policy in POLICIES
]))
"""
    ),
    markdown(
        """
## Run the full DS1 policy screen

The workflow first creates and hashes the policy proposal package. Only then does it open the evaluator. This makes it possible to prove that labels, categories, anomaly families, and object identities did not influence filtering.
"""
    ),
    code(
        """
from visual_peer_inspector.workflows.proposal_refinement_ds1 import run_ds1_proposal_refinement

if (OUTPUT_ROOT / "safe" / "run_record.json").is_file():
    run_record = json.loads((OUTPUT_ROOT / "safe" / "run_record.json").read_text(encoding="utf-8"))
    print("Reused the complete proposal-refinement output.")
else:
    run_record = run_ds1_proposal_refinement(
        proposal_root=PROPOSAL_ROOT,
        evaluator_root=EVALUATOR_ROOT,
        rgb_root=RGB_ROOT,
        output_root=OUTPUT_ROOT,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        bootstrap_seed=32_040,
    )
display(pd.DataFrame([run_record]))
"""
    ),
    markdown("## Overall metrics and paired uncertainty"),
    code(
        """
summary = pd.read_csv(OUTPUT_ROOT / "safe" / "policy_summary.csv")
intervals = pd.read_csv(OUTPUT_ROOT / "safe" / "paired_f1_intervals.csv")
gates = pd.read_csv(OUTPUT_ROOT / "safe" / "policy_gates.csv")
display(summary)
display(intervals)
display(gates)
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "policy_metrics.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "proposal_count_vs_f1.png")))
"""
    ),
    markdown("## Removed boxes, misses, and extras"),
    code(
        """
removals = pd.read_csv(OUTPUT_ROOT / "safe" / "removal_summary.csv")
display(removals.query("removed_count > 0"))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "misses_and_extras.png")))
"""
    ),
    markdown(
        """
## Subgroup checks

Aggregate subgroup results show whether a policy has a concentrated failure pattern. These tables are evaluator-only summaries. Scene IDs, object IDs, individual boxes, and labels are not included.
"""
    ),
    code(
        """
subgroups = pd.read_csv(OUTPUT_ROOT / "safe" / "subgroup_summary.csv")
selected_policy = run_record["selected_diagnostic_policy"]
selected_subgroups = subgroups.query("policy == @selected_policy")
display(selected_subgroups.sort_values(["dimension", "recall_at_0_50"]).groupby("dimension", as_index=False).head(5))
"""
    ),
    markdown(
        """
## Private before-and-after examples

The panel selects images where the chosen policy removes the most boxes. Red boxes are the original proposals and green boxes are the filtered proposals. This image contains private RGB data, so it stays in Kaggle output and review-only evidence. It is not sent to GitHub or ClearML.
"""
    ),
    code(
        """
display(DisplayImage(filename=str(OUTPUT_ROOT / "private" / "figures" / "baseline_vs_selected_boxes.png")))
"""
    ),
    markdown("## Completion gates and decision"),
    code(
        """
completion = pd.read_csv(OUTPUT_ROOT / "safe" / "completion_gates.csv")
decision = json.loads((OUTPUT_ROOT / "safe" / "decision.json").read_text(encoding="utf-8"))
display(completion)
display(pd.DataFrame([decision]))
if not completion["passed"].astype(bool).all():
    raise RuntimeError("Notebook 04.2 has a failed completion gate")
"""
    ),
    markdown(
        """
## Safe ClearML finalisation

ClearML starts only after the model-independent computation and manifests finish. The finaliser accepts a strict aggregate allowlist. It rejects RGB images, boxes, labels, IDs, paths, crosswalks, and credentials. If Kaggle Secrets are unavailable or tracking times out, the notebook records that status without losing the completed experiment.
"""
    ),
    code(
        """
def read_kaggle_secret(name: str) -> str | None:
    if EXECUTION_ENVIRONMENT != "kaggle":
        return os.environ.get(name)
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None


TRACKING_MODE = os.getenv("VPI_TRACKING_MODE", "online" if EXECUTION_ENVIRONMENT == "kaggle" else "disabled")
tracking_environment = os.environ.copy()
if TRACKING_MODE == "online":
    credentials = {name: read_kaggle_secret(name) for name in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")}
    if all(credentials.values()):
        tracking_environment.update(credentials)
        tracking_environment.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
        tracking_environment.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
        tracking_environment.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    else:
        TRACKING_MODE = "disabled"

receipt = OUTPUT_ROOT / "clearml_receipt.json"
command = [
    sys.executable,
    str(CODE_ROOT / "scripts" / "finalize_ds1_proposal_refinement_clearml.py"),
    "--safe-root", str(OUTPUT_ROOT / "safe"),
    "--receipt", str(receipt),
    "--mode", TRACKING_MODE,
    "--task-name", "ds1-proposal-refinement-kaggle-candidate",
    "--tag", "candidate", "--tag", "notebook-04.2", "--tag", "ds1", "--tag", "kaggle",
]
try:
    subprocess.run(command, env=tracking_environment, check=False, capture_output=True, text=True, timeout=240)
except subprocess.TimeoutExpired:
    print("ClearML finalisation timed out after the completed experiment.")
if not receipt.is_file():
    receipt.write_text(json.dumps({"status": "not_recorded", "task_id": None}) + "\\n", encoding="utf-8")
display(pd.DataFrame([json.loads(receipt.read_text(encoding="utf-8"))]))
"""
    ),
    markdown("## Final artifact verification"),
    code(
        """
verified = subprocess.run(
    [sys.executable, str(CODE_ROOT / "scripts" / "verify_ds1_proposal_refinement.py"), "--output-root", str(OUTPUT_ROOT)],
    check=True,
    text=True,
    capture_output=True,
)
display(pd.DataFrame([json.loads(verified.stdout)]))
"""
    ),
    markdown(
        """
## Next use

Notebook 04.2 does not block the representation work. Its decision controls only the automatic proposal route. The supplied-mask and reference-box routes continue into Notebook 07.1. If the selected policy still misses the recall gate, later work must replace or reprompt the detector because filtering cannot recover absent objects.
"""
    ),
]


notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
