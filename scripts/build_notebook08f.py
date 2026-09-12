#!/usr/bin/env python3
"""Build the clean Notebook 08F cross-view identity freeze."""

from __future__ import annotations

from pathlib import Path

import nbformat
from notebook_runtime_cells import harden_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "development-freeze"
NOTEBOOK_NAME = "08f-toysad8k-cross-view-identity-freeze.ipynb"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K Cross-View Identity Freeze

**Notebook file:** `08f-toysad8k-cross-view-identity-freeze.ipynb`

This notebook tests whether the same automatically detected object can be linked across two to five uploaded views. It consumes the locked LocateAnything, SAM 2.1, DINOv2-small, and P1 outputs from Notebook 08E. It compares DINO global-embedding matching with one bounded normalized-geometry variant and freezes the simplest method justified by development evidence.

The final figures show the actual automatic odd-one-out result, its object masks, and the supporting P1 heatmaps. Human correction remains an optional review feature after the automatic result. It is not part of the main model path.
"""
    ),
    markdown(
        """
## Evaluation boundary

The stage follows a strict order:

1. Verify every Notebook 08E prediction, embedding, index, and lock hash.
2. Produce embedding-only and normalized-geometry identity predictions.
3. Test two, three, and five arbitrary views, reversed input order, one missing view, and duplicate-view rejection.
4. Lock all identity predictions and their configuration hashes.
5. Only then open evaluator correspondence and base-asset cluster fields.
6. Report coverage, unmatched rate, ambiguity rate, switch rate, pairwise identity accuracy, and cycle consistency with base-asset-clustered intervals.
7. Freeze one method and a versioned web-application output contract.

The method resource also binds the exact Notebook 08E base-model identity. A later stage cannot silently change the LocateAnything prompt or cleanup, SAM 2.1 configuration, DINOv2-small configuration, preprocessing, or output schema while claiming to use this freeze.

ToysAD camera metadata is allowed only as an evaluator diagnostic. It cannot enter either candidate method or the selected web method. No arbitrary universal accuracy threshold is used. The normalized-geometry candidate is selected only when paired clustered evidence shows a favorable change, no adverse change, and all structural stability checks pass.
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

import pandas as pd
from IPython.display import Image as NotebookImage, display

sys.dont_write_bytecode = True

NOTEBOOK_NAME = "08f-toysad8k-cross-view-identity-freeze.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "workspace"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_08F_OUTPUT_ROOT",
    "/kaggle/working/toysad8k_cross_view_identity_freeze"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/toysad8k-cross-view-identity-freeze",
)).expanduser().resolve()
"""
    ),
    code(
        r'''
def mounted_roots() -> list[Path]:
    """Return bounded Kaggle input roots without scanning the source archive."""

    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return []
    roots = []
    for pattern in (
        "*",
        "datasets/*/*",
        "datasets/*/*/versions/*",
        "notebooks/*/*",
        "notebooks/*/*/versions/*",
    ):
        roots.extend(path for path in input_root.glob(pattern) if path.is_dir())
    return sorted(set(roots), key=lambda path: path.as_posix())


def configured_paths(variable: str) -> list[Path]:
    value = os.getenv(variable, "").strip()
    if not value:
        return []
    return [Path(item).expanduser() for item in value.split(os.pathsep) if item]


def find_code_root() -> Path:
    candidates = configured_paths("VPI_CODE_ROOT")
    candidates.extend((Path.cwd(), *Path.cwd().parents, *mounted_roots()))
    for candidate in candidates:
        if (candidate / "scripts/run_cross_view_identity_freeze.py").is_file():
            return candidate.resolve()
        if (candidate / "code" / "scripts/run_cross_view_identity_freeze.py").is_file():
            return (candidate / "code").resolve()
    raise FileNotFoundError("The Visual Peer Inspector code snapshot was not found")


def find_prediction_roots() -> list[Path]:
    configured = configured_paths("VPI_08E_PREDICTION_ROOTS")
    if configured:
        roots = [path.resolve() for path in configured]
    else:
        roots = []
        patterns = (
            "prediction_manifest.jsonl",
            "*/model-safe/prediction_manifest.jsonl",
            "*/*/model-safe/prediction_manifest.jsonl",
            "*/*/*/model-safe/prediction_manifest.jsonl",
        )
        for mounted in mounted_roots():
            for pattern in patterns:
                roots.extend(path.parent.resolve() for path in mounted.glob(pattern))
    unique = sorted(set(roots), key=lambda path: path.as_posix())
    if not unique:
        raise FileNotFoundError("No Notebook 08E model-safe prediction root was found")
    return unique


def find_marker(name: str, variable: str) -> Path:
    configured = configured_paths(variable)
    candidates = configured + mounted_roots()
    matches = []
    for candidate in candidates:
        direct = candidate / name
        if direct.is_file():
            matches.append(direct.resolve())
        for pattern in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
            matches.extend(path.resolve() for path in candidate.glob(pattern) if path.is_file())
    unique = sorted(set(matches), key=lambda path: path.as_posix())
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected one {name}; found {len(unique)}. Set {variable} explicitly."
        )
    return unique[0]


def find_media_roots() -> list[Path]:
    configured = configured_paths("VPI_MEDIA_ROOTS")
    if configured:
        return sorted({path.resolve() for path in configured})
    roots = []
    for mounted in mounted_roots():
        if (mounted / "media").is_dir():
            roots.append(mounted.resolve())
        for child in mounted.glob("*/model-input"):
            if (child / "media").is_dir():
                roots.append(child.resolve())
    unique = sorted(set(roots), key=lambda path: path.as_posix())
    if not unique:
        raise FileNotFoundError("No released RGB media roots were found")
    return unique


def code_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(CODE_ROOT / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not previous else source_root + os.pathsep + previous
    return environment


def read_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.getenv(name)
'''
    ),
    markdown("## Locate and verify the locked inputs"),
    code(
        """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/development-freeze/08f-toysad8k-cross-view-identity-freeze.ipynb",
)
PREDICTION_ROOTS = find_prediction_roots()
EVALUATOR_PATH = find_marker(
    "evaluator_correspondence.jsonl",
    "VPI_EVALUATOR_CORRESPONDENCE",
)
MEDIA_ROOTS = find_media_roots()

from visual_peer_inspector.workflows.cross_view_identity_freeze import load_08e_predictions

verified = load_08e_predictions(PREDICTION_ROOTS)
display({
    "environment": EXECUTION_ENVIRONMENT,
    "input cohorts": list(verified.cohort_ids),
    "scene count": len(verified.views_by_scene),
    "embedding dimension": verified.embedding_dimension,
    "verified Notebook 08E locks": len(verified.input_lock_sha256s),
})
"""
    ),
    markdown("## Run the prediction-first identity freeze"),
    code(
        """
# ClearML runs only after the complete workflow has written its local evidence.
# A separate child process bounds tracking so a network delay cannot hold the stage.
environment = code_environment()
clearml_access = read_secret("CLEARML_API_ACCESS_KEY")
clearml_secret = read_secret("CLEARML_API_SECRET_KEY")
clearml_mode = "online" if clearml_access and clearml_secret else "disabled"
if clearml_mode == "online":
    try:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", "-r",
                str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt"),
            ],
            check=True,
            timeout=180,
            env=environment,
        )
    except Exception as error:
        clearml_mode = "disabled"
        print(f"ClearML dependency setup was unavailable: {type(error).__name__}")
if clearml_mode == "online":
    environment.update({
        "CLEARML_API_ACCESS_KEY": clearml_access,
        "CLEARML_API_SECRET_KEY": clearml_secret,
        "CLEARML_API_HOST": "https://api.clear.ml",
        "CLEARML_WEB_HOST": "https://app.clear.ml",
        "CLEARML_FILES_HOST": "https://files.clear.ml",
    })

command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_cross_view_identity_freeze.py"),
]
for path in PREDICTION_ROOTS:
    command.extend(("--prediction-root", str(path)))
for path in MEDIA_ROOTS:
    command.extend(("--media-root", str(path)))
command.extend((
    "--evaluator-correspondence", str(EVALUATOR_PATH),
    "--output-root", str(OUTPUT_ROOT),
    "--clearml-mode", clearml_mode,
    "--clearml-wall-time-seconds", "180",
))
subprocess.run(command, check=True, env=environment)

SAFE_ROOT = OUTPUT_ROOT / "safe"
run_record = json.loads((SAFE_ROOT / "run_record.json").read_text(encoding="utf-8"))
display({
    "status": run_record["status"],
    "scenes": run_record["scene_count"],
    "identity predictions": run_record["prediction_count"],
    "resumed predictions": run_record["resumed_from_verified_receipts"],
    "chosen method": run_record["chosen_variant"],
    "odd-one-out panels": run_record["qualitative_panel_count"],
})
"""
    ),
    markdown("## Structural checks and stability"),
    code(
        """
gates = pd.read_csv(SAFE_ROOT / "acceptance_gates.csv")
stability = pd.read_csv(SAFE_ROOT / "stability_summary.csv")
display(gates)
display(stability)

if not bool(gates["passed"].all()):
    failed = gates.loc[~gates["passed"].astype(bool), "gate"].tolist()
    raise RuntimeError(f"Notebook 08F acceptance gates did not pass: {failed}")
"""
    ),
    markdown("## Identity metrics and clustered uncertainty"),
    code(
        """
summary = pd.read_csv(SAFE_ROOT / "method_summary.csv")
intervals = pd.read_csv(SAFE_ROOT / "clustered_intervals.csv")
paired = pd.read_csv(SAFE_ROOT / "paired_method_intervals.csv")

display(summary)
display(intervals)
display(paired)

# Two-view cycle consistency is not defined because a cycle needs three views.
# It remains explicit as not applicable rather than being reported as a perfect score.
"""
    ),
    markdown("## Frozen identity method and web output contract"),
    code(
        """
method_freeze = json.loads((SAFE_ROOT / "method_freeze.json").read_text(encoding="utf-8"))
application_contract = json.loads(
    (SAFE_ROOT / "application_output_contract.json").read_text(encoding="utf-8")
)
display(method_freeze)
display(application_contract)

application_examples = [
    json.loads(line)
    for line in (OUTPUT_ROOT / "downstream" / "application_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
display({
    "versioned application results": len(application_examples),
    "supported view counts observed": sorted({row["view_count"] for row in application_examples}),
    "example result hash": application_examples[0]["result_sha256"],
})
"""
    ),
    markdown("## Deployment-valid odd-one-out evidence"),
    code(
        """
qualitative = pd.read_csv(OUTPUT_ROOT / "review-only" / "qualitative_manifest.csv")
display(qualitative[[
    "result_id", "fused_odd_one_out", "agreement", "margin", "review_state"
]])

for path in sorted((OUTPUT_ROOT / "review-only" / "figures" / "odd-one-out").glob("*-prediction.png"))[:6]:
    display(NotebookImage(filename=str(path)))
for path in sorted((OUTPUT_ROOT / "review-only" / "figures" / "odd-one-out").glob("*-sam-evidence.png"))[:3]:
    display(NotebookImage(filename=str(path)))
"""
    ),
    markdown("## Aggregate figures and saved evidence"),
    code(
        """
for name in (
    "five_view_identity_accuracy.png",
    "missing_view_link_stability.png",
):
    path = SAFE_ROOT / "figures" / name
    if path.is_file():
        display(NotebookImage(filename=str(path)))

artifact_manifest = json.loads(
    (SAFE_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
)
tracking_receipt = json.loads(
    (OUTPUT_ROOT / "tracking" / "clearml_receipt.json").read_text(encoding="utf-8")
)
display({
    "safe artifact count": artifact_manifest["artifact_count"],
    "safe artifact inventory SHA-256": artifact_manifest["artifact_inventory_sha256"],
    "ClearML status": tracking_receipt["status"],
    "ClearML task ID": tracking_receipt.get("task_id"),
    "download root": str(OUTPUT_ROOT),
    "minimal method resource": str(OUTPUT_ROOT / "publish" / "method-freeze"),
})
"""
    ),
    markdown(
        """
## Result boundary

Notebook 08F freezes the cross-view identity method and the versioned application result shape. It does not use calibration or internal-lock data to select the method. The next stage may run the frozen automatic pipeline on calibration RGB, lock those predictions, and only then use calibration labels to set the review policy. Internal-lock RGB and labels remain sealed until the final audit.

The published resource includes the exact base-model identity hash inherited from Notebook 08E. Every protected and official prediction notebook must rebuild that identity from its selected prompt and model configuration and reject the run before GPU inference if the identity differs.

Publish only `publish/method-freeze` as the private Kaggle dataset `vpi-toysad-development-method-freeze`. Later prediction-only notebooks must not mount the complete Notebook 08F output because it also contains development evaluator evidence.

The web application can be built after the automatic route, cross-view identity, odd-one-out fusion, and review policy have passed their declared integrity and evaluation checks. Human correction stays available as a separate review action and must never replace reporting of the automatic result.
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"vpi-08f-{index:02d}"


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
NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
nbformat.write(harden_notebook(notebook), NOTEBOOK_ROOT / NOTEBOOK_NAME)
print(NOTEBOOK_ROOT / NOTEBOOK_NAME)
