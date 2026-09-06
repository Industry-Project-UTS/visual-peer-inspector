#!/usr/bin/env python3
"""Build the clean Notebook 09D and Notebook 10B one-shot evaluators."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = (
    {
        "stage": "09d",
        "title": "ToysAD-8K Internal-Lock One-Shot Evaluation",
        "filename": "09d-toysad8k-internal-lock-one-shot-evaluation.ipynb",
        "directory": "protected-evaluation",
        "contract_name": "evaluation_contract_09d.json",
        "output_name": "toysad8k_internal_lock_one_shot_evaluation",
        "description": (
            "This notebook opens the 400-scene internal-lock evaluator only after "
            "the frozen prediction, method, and review-policy hashes verify. It "
            "reports the result once and does not change the method or policy."
        ),
        "scope_note": (
            "The protocol receipt confirms only that the fixed evaluation completed "
            "without retuning. It does not claim that scientific performance was accepted."
        ),
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-internal-lock400-evaluator",
            "esreekarreddy/vpi-toysad-internal-lock400-model-input",
            "esreekarreddy/vpi-toysad-development-method-freeze",
            "esreekarreddy/vpi-toysad-calibration-review-policy",
        ],
        "kernel_sources": [
            "esreekarreddy/09c-toysad8k-protected-internal-lock-inference",
        ],
        "expected_splits": "internal-lock",
    },
    {
        "stage": "10b",
        "title": "ToysAD-8K Official Seen and Unseen Test Evaluation",
        "filename": "10b-toysad8k-official-seen-unseen-evaluation.ipynb",
        "directory": "official-evaluation",
        "contract_name": "evaluation_contract_10b.json",
        "output_name": "toysad8k_official_seen_unseen_evaluation",
        "description": (
            "This notebook opens the official seen-test and unseen-test evaluators "
            "only after both prediction locks and all frozen hashes verify. It "
            "reports the two test splits separately and adds a pooled descriptive table."
        ),
        "scope_note": (
            "Seen-test and unseen-test are the primary results. The pooled row is "
            "descriptive only. No official result may change the method or policy."
        ),
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-official-seen-evaluator",
            "esreekarreddy/vpi-toysad-official-unseen-evaluator",
            "esreekarreddy/vpi-toysad-official-seen-prediction",
            "esreekarreddy/vpi-toysad-official-unseen-prediction",
            "esreekarreddy/vpi-toysad-official-prediction-summary",
            "esreekarreddy/vpi-toysad-official-seen-model-input",
            "esreekarreddy/vpi-toysad-official-unseen-model-input",
            "esreekarreddy/vpi-toysad-development-method-freeze",
            "esreekarreddy/vpi-toysad-calibration-review-policy",
            "esreekarreddy/vpi-toysad-internal-lock-decision",
        ],
        "kernel_sources": [],
        "expected_splits": "seen-test and unseen-test",
    },
)


def markdown(value: str):
    return nbformat.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbformat.v4.new_code_cell(value.strip())


HELPERS = r"""
def mounted_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        return sorted((path.resolve() for path in input_root.iterdir() if path.is_dir()), key=str)
    return [Path.cwd().resolve()]


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents, *mounted_roots()))
    for candidate in candidates:
        if (candidate / "scripts/run_one_shot_evaluation.py").is_file():
            return candidate.resolve()
        if (candidate / "code/scripts/run_one_shot_evaluation.py").is_file():
            return (candidate / "code").resolve()
    raise FileNotFoundError("Could not locate the Visual Peer Inspector code snapshot")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def display_images(paths: list[Path], limit: int | None = None) -> None:
    selected = paths if limit is None else paths[:limit]
    for path in selected:
        print(path.name)
        display(NotebookImage(filename=str(path)))


def run_tracking_finalizer(result_root: Path) -> dict:
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts/run_one_shot_evaluation.py"),
        "track",
        "--result-root", str(result_root),
    ]
    receipt_path = result_root / "clearml_receipt.json"
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
            env=os.environ.copy(),
        )
        print(completed.stdout.strip())
    except Exception as error:
        receipt = {
            "schema_version": "visual-peer-clearml-finalizer/v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "scientific_output_preserved": True,
            "credentials_persisted": False,
        }
        temporary = receipt_path.with_suffix(".json.incomplete")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)
    return json.loads(receipt_path.read_text(encoding="utf-8"))
"""


def build_cells(configuration: dict) -> list:
    stage = configuration["stage"]
    filename = configuration["filename"]
    output_name = configuration["output_name"]
    contract_name = configuration["contract_name"]
    final_receipt = (
        """receipt = json.loads((OUTPUT_ROOT / 'internal_lock_result_receipt.json').read_text(encoding='utf-8'))
display(receipt)
display({'minimal decision resource': str(OUTPUT_ROOT / 'publish' / 'internal-lock-decision')})"""
        if stage == "09d"
        else "display(summary.loc[summary['split_name'].isin(['seen-test', 'unseen-test', 'pooled-descriptive'])])"
    )
    upstream_gate_note = (
        "- The completed Notebook 09D protocol receipt and the Notebook 10A summary must "
        "link the same method, policy, and both official prediction locks."
        if stage == "10b"
        else "- The completed evaluation emits the immutable receipt required by Notebook 10A."
    )
    correspondence_note = (
        "- Official prediction objects match evaluator references one-to-one at box IoU 0.50. Lower overlaps remain unmatched and cannot earn a correct odd-one-out result."
        if stage == "10b"
        else "- The internal-lock evaluator uses its locked prediction-to-reference crosswalk."
    )
    release_note = (
        "Publish only `publish/internal-lock-decision` as the private Kaggle "
        "dataset `vpi-toysad-internal-lock-decision`. Notebook 10A must not "
        "mount the complete Notebook 09D evaluator output."
        if stage == "09d"
        else "Official evaluator outputs remain separated from all model-facing resources."
    )
    return [
        markdown(
            f"""
# {configuration["title"]}

**Notebook file:** `{filename}`

{configuration["description"]}

The final route is LocateAnything, SAM 2.1, DINOv2-small with P1 scoring,
label-free cross-view identity matching, mean normalized-rank fusion, and the
review policy frozen in Notebook 09B. The evaluator cannot alter any part of
that route.
"""
        ),
        markdown(
            f"""
## Evaluation contract

- Every prediction artifact must match its immutable prediction lock.
- The exact Notebook 08F method hash and Notebook 09B policy hash must match.
- The content-addressed execution contract must identify the exact cohort membership,
  prediction, run-record, and evaluator-package hashes.
{upstream_gate_note}
{correspondence_note}
- Evaluator labels are parsed only after all non-label checks pass.
- A reused nonempty result directory is rejected.
- Failed stages, zero proposals, missing results, and review outcomes remain in
  the denominator.
- Results are reported for 2, 3, and 5 uploaded views.
- Base-asset-clustered confidence intervals preserve repeated-scene dependence.
- The notebook does not tune a threshold or apply a universal accuracy gate.
- ClearML starts only after the scientific outputs and hashes are complete.

{configuration["scope_note"]}
"""
        ),
        code(
            f"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from IPython.display import Image as NotebookImage, display

sys.dont_write_bytecode = True

NOTEBOOK_FILE = "{filename}"
STAGE = "{stage}"
CONTRACT_FILE = "{contract_name}"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/working/{output_name}"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/{output_name}",
)).expanduser().resolve()
"""
        ),
        code(HELPERS),
        markdown(
            """
## Resolve and lock the inputs

The notebook first verifies the frozen method, review policy, prediction locks,
prediction-only run records, fixed cohort identities, and evaluator manifests.
It then writes their exact hashes to a local execution contract. Evaluator label
rows are not parsed during this step. A filename match alone is never accepted.
"""
        ),
        code(
            f"""
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/{configuration['directory']}/{filename}",
)
MOUNTED_ROOTS = mounted_roots()
EVALUATION_CONTRACT = Path(
    "/kaggle/working" if EXECUTION_ENVIRONMENT == "kaggle" else ".local/runs"
) / CONTRACT_FILE

contract_command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_one_shot_evaluation.py"),
    "contract",
    "--stage", STAGE,
    "--output", str(EVALUATION_CONTRACT),
]
for root in MOUNTED_ROOTS:
    contract_command.extend(("--mount-root", str(root)))
subprocess.run(contract_command, check=True, timeout=1800)

from visual_peer_inspector.workflows.one_shot_evaluation import load_evaluation_contract

contract = load_evaluation_contract(EVALUATION_CONTRACT, expected_stage=STAGE)
display({{
    "notebook": NOTEBOOK_FILE,
    "stage": STAGE,
    "contract SHA-256": contract.contract_sha256,
    "method SHA-256": contract.method_record_sha256,
    "policy SHA-256": contract.policy_record_sha256,
    "cohorts": [item.cohort_id for item in contract.cohorts],
    "expected splits": {configuration["expected_splits"]!r},
}})
"""
        ),
        markdown(
            """
## Run the sealed evaluation

The command verifies all model-facing files first, writes a boundary receipt,
and only then opens the evaluator package. Output files are written through
temporary paths and renamed only when complete.
"""
        ),
        code(
            """
if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise FileExistsError("The one-shot output directory is already in use")

command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_one_shot_evaluation.py"),
    "evaluate",
    "--stage", STAGE,
    "--evaluation-contract", str(EVALUATION_CONTRACT),
    "--output-root", str(OUTPUT_ROOT),
    "--bootstrap-repetitions", "10000",
    "--panel-count", "16",
]
for root in MOUNTED_ROOTS:
    command.extend(("--mount-root", str(root)))

subprocess.run(command, check=True, timeout=7200)
completion = json.loads((OUTPUT_ROOT / "evaluation_complete.json").read_text(encoding="utf-8"))
if completion.get("status") != "complete":
    raise RuntimeError("Evaluation did not produce a complete atomic result")
display(completion)
"""
        ),
        markdown(
            """
## Final odd-one-out results

Top-1 is the web application's primary single-candidate result. Average
precision retains the full ranking and includes missed anomalous objects in its
denominator. Recall at two measures recovery of up to two anomalous objects,
while exact-set accuracy checks whether the first `anomaly_count` identities are
exactly the complete anomalous set. Top-2 and MRR retain the first-target view of
the ranking. Coverage includes every scene, including pipeline failures.
Selective accuracy describes only results released without a review flag, while
selective risk is its error rate.
"""
        ),
        code(
            """
SAFE_ROOT = OUTPUT_ROOT / "safe"
summary = read_csv(SAFE_ROOT / "summary_metrics.csv")
intervals = read_csv(SAFE_ROOT / "clustered_confidence_intervals.csv")
display(summary)
display(intervals)
display_images([
    SAFE_ROOT / "figures/ranking_by_view.png",
    SAFE_ROOT / "figures/coverage_and_review.png",
])
"""
        ),
        markdown(
            """
## Reliability and subgroup checks

Margin and agreement are diagnostic signals, not probabilities. Their fixed
bins show whether larger values correspond to better empirical results. The
subgroup tables expose category, anomaly-type, object-count, and view-count
variation without changing the frozen policy.
"""
        ),
        code(
            """
reliability = read_csv(SAFE_ROOT / "margin_agreement_reliability.csv")
subgroups = read_csv(SAFE_ROOT / "subgroup_and_category_metrics.csv")
display(reliability)
display(subgroups)
display_images([SAFE_ROOT / "figures/margin_reliability.png"])
"""
        ),
        markdown(
            """
## Pipeline attrition, identity failures, and error taxonomy

The attrition table follows evaluator-supplied target boxes through proposals,
nonempty SAM masks, cross-view identity tracks, and ranked tracks. Box and mask
IoU are reported only when the locked evaluator supplies the corresponding
geometry. The official ToysAD evaluation therefore measures anomalous target
retention rather than full normal-object detection recall. The error table
separates missing outputs, identity errors, wrong automatic results, and
results correctly sent for review.
"""
        ),
        code(
            """
for table_name in (
    "pipeline_attrition.csv",
    "identity_failures.csv",
    "prediction_failures.csv",
    "runtime_summary.csv",
    "decision_confusion.csv",
    "error_taxonomy.csv",
):
    print(table_name)
    display(read_csv(SAFE_ROOT / table_name))
display_images([SAFE_ROOT / "figures/pipeline_attrition.png"])
"""
        ),
        markdown(
            """
## Private qualitative evidence

Sixteen deterministic panels are selected across best, typical, wrong, and
failure outcomes where those outcomes exist. Each panel shows the input views,
LocateAnything boxes, SAM masks, P1 evidence, the fused odd-one-out, and private
reference overlays. These panels are evaluator-only and must not enter a
model-facing or public application package.
"""
        ),
        code(
            """
inventory = json.loads((OUTPUT_ROOT / "review-only/qualitative_panel_inventory.json").read_text(encoding="utf-8"))
display(pd.DataFrame(inventory["items"]))
panel_paths = sorted((OUTPUT_ROOT / "review-only").rglob("panel-*.png"))
if len(panel_paths) != inventory["panel_count"]:
    raise RuntimeError("Panel files differ from the qualitative inventory")
display_images(panel_paths)
"""
        ),
        markdown(
            """
## Bounded ClearML finalization

ClearML receives aggregate CSV tables and charts only. Scene-level labels,
crosswalks, RGB files, masks, and private panels remain local to this evaluator
run. A tracking timeout or service error creates a sanitized failure receipt and
does not invalidate the completed evaluation.
"""
        ),
        code(
            """
clearml_receipt = run_tracking_finalizer(OUTPUT_ROOT)
display(clearml_receipt)
"""
        ),
        markdown(
            f"""
## Final locked record

The tables above are measurements, not settings. The method and policy remain
unchanged after this notebook. Any later report must cite the result hashes and
must preserve separate failure, review, and insufficient-evidence outcomes.
{release_note}
"""
        ),
        code(final_receipt),
    ]


def metadata(configuration: dict) -> dict:
    return {
        "id": f"esreekarreddy/{configuration['filename'].removesuffix('.ipynb')}",
        "title": configuration["title"],
        "code_file": configuration["filename"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": configuration["dataset_sources"],
        "competition_sources": [],
        "kernel_sources": configuration["kernel_sources"],
        "model_sources": [],
    }


def build(configuration: dict) -> tuple[Path, Path]:
    directory = ROOT / "notebooks" / configuration["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    notebook_path = directory / configuration["filename"]
    cells = build_cells(configuration)
    for index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        cell["id"] = sha256(
            f"{configuration['stage']}|{index}|{cell['cell_type']}|{source}".encode()
        ).hexdigest()[:8]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbformat.write(notebook, notebook_path)
    metadata_path = directory / f"kernel-metadata-{configuration['stage']}.json"
    metadata_path.write_text(
        json.dumps(metadata(configuration), indent=2) + "\n", encoding="utf-8"
    )
    return notebook_path, metadata_path


def main() -> None:
    for configuration in CONFIGURATIONS:
        notebook_path, metadata_path = build(configuration)
        print(notebook_path.relative_to(ROOT))
        print(metadata_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
