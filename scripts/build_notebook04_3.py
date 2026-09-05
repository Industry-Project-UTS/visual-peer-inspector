#!/usr/bin/env python3
"""Build the clean Notebook 04.3 source from reviewed cells."""

from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "notebooks"
    / "detection-and-segmentation"
    / "toysad8k-ds1-locateanything-prompt-freeze"
    / "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb"
)


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K DS1 LocateAnything Prompt Freeze

**Notebook file:** `04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb`

This notebook compares two short, generic prompts for the automatic object proposal route. It reuses the completed `object` baseline from Notebook 04.1 and runs `individual object` through LocateAnything. The model is loaded once.

The new prompt must complete a one-image check, a 20-image pilot, a measured pilot repeat, and the full 2,000-image DS1 screen. After both proposal files are complete and hash locked, the protected evaluator compares raw boxes, area and overlap filtering, and an eight-proposal outlier safeguard. The outlier cap is derived from the label-free DS1 proposal-count distribution. The selected candidate maximises F2 at box IoU 0.50, with documented tie breakers. F2 is used because a missed object is more harmful than an extra proposal that a user can remove. No fixed metric cutoff is used to select the candidate.
"""
    ),
    markdown(
        """
## Experiment boundary

The two supported prompts are fixed before the run:

1. `object`
2. `individual object`

DS1 is development data. This notebook can choose one prompt and postprocessing policy for the next stage, but it cannot make a final performance claim. The chosen settings must be applied unchanged to DS2 confirmation data. RGB images, per-image boxes, model responses, evaluator records, and example panels stay in private Kaggle output. ClearML receives aggregate tables, charts, configuration records, and hashes only.
"""
    ),
    code(
        """
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from IPython.display import Image as DisplayImage, display
import pandas as pd

NOTEBOOK_NAME = "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/working").is_dir() else "local"
default_output = (
    Path("/kaggle/temp/toysad8k_ds1_locateanything_prompt_freeze")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else Path.cwd() / ".local" / "runs" / "notebook04_3"
)
OUTPUT_ROOT = Path(os.getenv("VPI_OUTPUT_ROOT", str(default_output))).expanduser().resolve()

print(f"Environment: {EXECUTION_ENVIRONMENT}")
print(f"Notebook: {NOTEBOOK_NAME}")
print(f"Output folder: {OUTPUT_ROOT}")
"""
    ),
    markdown("## Locate the code and frozen inputs"),
    code(
        """
def kaggle_dataset_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return []
    candidates: list[Path] = []
    for pattern in ("*", "datasets/*/*", "datasets/*/*/versions/*"):
        candidates.extend(path for path in input_root.glob(pattern) if path.is_dir())
    return sorted(set(candidates), key=lambda path: path.as_posix())


def find_code_root() -> Path:
    # Prefer an explicit path, then known local repository and Kaggle mount shapes.
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for mounted in kaggle_dataset_roots():
        candidates.extend((mounted, mounted / "code"))
    for candidate in candidates:
        marker = candidate / "src/visual_peer_inspector/workflows/locateanything_prompt_freeze_ds1.py"
        if marker.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the Notebook 04.3 code snapshot")


def find_package(variable: str, predicate, required_entries: tuple[str, ...]) -> Path:
    # Accept a mounted package only when its own record satisfies the contract.
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    for mounted in kaggle_dataset_roots():
        candidates.append(mounted)
        candidates.extend(child for child in sorted(mounted.iterdir()) if child.is_dir())
    for candidate in candidates:
        record_path = candidate / "package_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if predicate(record) and all((candidate / entry).exists() for entry in required_entries):
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate required package: {variable}")


def find_baseline_result() -> Path:
    configured = os.getenv("VPI_BASELINE_RESULTS")
    if configured:
        candidates = [Path(configured).expanduser()]
    else:
        candidates = []
        for mounted in kaggle_dataset_roots():
            for pattern in (
                "private/proposal_results.jsonl",
                "*/private/proposal_results.jsonl",
                "*/*/private/proposal_results.jsonl",
                "*/*/*/private/proposal_results.jsonl",
            ):
                candidates.extend(mounted.glob(pattern))
    matches_by_identity: dict[tuple[str, str], list[Path]] = {}
    for result_path in candidates:
        run_path = result_path.parent.parent / "safe" / "run_record.json"
        if not run_path.is_file():
            continue
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if (
            record.get("stage") == "ds1_locateanything_screen"
            and record.get("status") == "completed"
            and record.get("configuration", {}).get("prompt") == "object"
            and record.get("summary", {}).get("job_count") == 2_000
        ):
            identity = (
                hashlib.sha256(result_path.read_bytes()).hexdigest(),
                hashlib.sha256(run_path.read_bytes()).hexdigest(),
            )
            matches_by_identity.setdefault(identity, []).append(result_path.resolve())
    if len(matches_by_identity) != 1:
        raise RuntimeError(
            "Expected one distinct completed Notebook 04.1 result, "
            f"found {len(matches_by_identity)}"
        )
    return sorted(next(iter(matches_by_identity.values())), key=lambda path: path.as_posix())[0]


CODE_ROOT = find_code_root()
MODEL_INPUT_ROOT = find_package(
    "VPI_MODEL_INPUT_ROOT",
    lambda record: (
        record.get("cohort_id") == "toysad8k-development-screen-100-v1"
        and record.get("release_id") == "toysad8k-train-2000-v2"
        and record.get("image_count") == 2_000
    ),
    ("neutral_corpus_manifest.jsonl", "media"),
)
EVALUATOR_ROOT = find_package(
    "VPI_EVALUATOR_ROOT",
    lambda record: record.get("package_kind") == "toysad8k-ds1-mask-evaluator",
    ("evaluator_crosswalk.jsonl", "masks"),
)
BASELINE_RESULTS = find_baseline_result()
sys.path.insert(0, str(CODE_ROOT / "src"))

display(pd.DataFrame([
    {"input": "Code snapshot", "folder": CODE_ROOT.name},
    {"input": "DS1 RGB package", "folder": MODEL_INPUT_ROOT.name},
    {"input": "DS1 evaluator package", "folder": EVALUATOR_ROOT.name},
    {"input": "Notebook 04.1 baseline", "folder": BASELINE_RESULTS.parent.parent.name},
]))
"""
    ),
    markdown("## Install the pinned inference dependencies"),
    code(
        """
if EXECUTION_ENVIRONMENT == "kaggle":
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/locateanything-lock.txt")],
        timeout=600,
    )

from visual_peer_inspector.workflows.locateanything_prompt_freeze_ds1 import (
    POSTPROCESSING_POLICIES,
    PROMPT_CANDIDATES,
    prompt_configurations,
)

configurations = prompt_configurations()
display(pd.DataFrame([
    {
        "prompt key": key,
        "prompt": prompt,
        "source": "Notebook 04.1 baseline" if key == "object" else "new model run",
        "model revision": configurations[key].model_revision,
    }
    for key, prompt in PROMPT_CANDIDATES
]))
display(pd.DataFrame({"postprocessing policy": list(POSTPROCESSING_POLICIES)}))
"""
    ),
    markdown(
        """
## Run the ordered gates and full DS1 screen

ClearML is not started during model inference. The command prints model-load checkpoints and progress after the first image, every 100 images, and the last image. Existing candidate result files are resumed only when their configuration and image hashes match. If a gate fails, the next stage does not start.
"""
    ),
    code(
        """
if EXECUTION_ENVIRONMENT == "kaggle":
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Notebook 04.3 requires a Kaggle GPU")
    display(pd.DataFrame([{
        "GPU count": torch.cuda.device_count(),
        "active GPU": torch.cuda.get_device_name(0),
        "CUDA runtime": torch.version.cuda,
    }]))

command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_ds1_locateanything_prompt_freeze.py"),
    "--model-input-root", str(MODEL_INPUT_ROOT),
    "--evaluator-root", str(EVALUATOR_ROOT),
    "--baseline-results", str(BASELINE_RESULTS),
    "--output-root", str(OUTPUT_ROOT),
]
model_source = os.getenv("VPI_LOCATEANYTHING_MODEL_SOURCE")
if model_source:
    command.extend(["--model-source", model_source])

process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
assert process.stdout is not None
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
if return_code != 0:
    raise RuntimeError(f"Notebook 04.3 runner exited with status {return_code}")
"""
    ),
    markdown("## Prompt and policy results"),
    code(
        """
safe_root = OUTPUT_ROOT / "evaluation" / "safe"
summary = pd.read_csv(safe_root / "prompt_policy_summary.csv")
operational = pd.read_csv(safe_root / "prompt_operational_summary.csv")
repeatability = pd.read_csv(safe_root / "prompt_repeatability.csv")

display(summary.sort_values("f2_at_0_50", ascending=False))
display(operational)
display(repeatability)
display(DisplayImage(filename=str(safe_root / "prompt_policy_f2.png")))
display(DisplayImage(filename=str(safe_root / "complete_detection_rate.png")))
display(DisplayImage(filename=str(safe_root / "missed_objects.png")))
display(DisplayImage(filename=str(safe_root / "detection_aware_mean_iou.png")))
display(DisplayImage(filename=str(safe_root / "split_or_duplicate_candidates.png")))
display(DisplayImage(filename=str(safe_root / "mean_proposals_by_prompt.png")))
display(DisplayImage(filename=str(safe_root / "median_runtime_by_prompt.png")))
"""
    ),
    markdown(
        """
## Private visual review

The following panels show the same fixed 20-image selection for each prompt. Supplied reference boxes are thin green outlines. LocateAnything proposals are wider red outlines. Green drawn over red makes close matches visible in both colours. The panels are a visual diagnostic for missing, duplicate, merged, loose, background, and partial-object boxes. The success panel shows cases with no missed or extra boxes. The failure panel deliberately shows the most difficult cases and must not be presented as evidence of a fully automatic detector. These panels do not replace the evaluator metrics. They contain RGB images and remain in private Kaggle output.
"""
    ),
    code(
        """
example_root = OUTPUT_ROOT / "evaluation" / "private" / "prompt_examples"
for prompt_key, _prompt in PROMPT_CANDIDATES:
    for policy_key in POSTPROCESSING_POLICIES:
        display(DisplayImage(filename=str(example_root / f"{prompt_key}_{policy_key}_proposal_examples.png")))
for panel_name in ("selected_successes.png", "selected_failures.png"):
    panel_path = example_root / panel_name
    if panel_path.is_file():
        display(DisplayImage(filename=str(panel_path)))
"""
    ),
    markdown("## Selected DS1 configuration"),
    code(
        """
selection = json.loads((safe_root / "selected_prompt.json").read_text(encoding="utf-8"))
checks = pd.read_csv(safe_root / "completion_checks.csv")
display(pd.DataFrame([selection["selected"]]))
display(checks)
if not checks["passed"].astype(str).str.lower().isin(["true", "1"]).all():
    raise RuntimeError("Notebook 04.3 has an incomplete run or failed integrity check")

print(selection["selection_rule"])
print("This DS1 choice must be confirmed unchanged on DS2 before it is treated as frozen.")
"""
    ),
    markdown(
        """
## Safe ClearML finalisation

Tracking starts only after the experiment and artifact manifests are complete. The finaliser uses a strict aggregate allowlist and a wall-time limit. It rejects private identifiers, boxes, masks, RGB paths, raw responses, and credentials. A ClearML problem cannot invalidate or block the completed model run.
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


tracking_mode = os.getenv(
    "VPI_TRACKING_MODE", "online" if EXECUTION_ENVIRONMENT == "kaggle" else "disabled"
).strip().lower()
tracking_environment = os.environ.copy()
if tracking_mode == "online":
    credentials = {
        name: read_kaggle_secret(name)
        for name in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    }
    if all(credentials.values()):
        tracking_environment.update(credentials)
        tracking_environment.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
        tracking_environment.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
        tracking_environment.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    else:
        tracking_mode = "disabled"

receipt = OUTPUT_ROOT / "evaluation" / "clearml_receipt.json"
tracking_command = [
    sys.executable,
    str(CODE_ROOT / "scripts/finalize_ds1_locateanything_prompt_freeze_clearml.py"),
    "--safe-root", str(safe_root),
    "--receipt", str(receipt),
    "--mode", tracking_mode,
    "--task-name", "ds1-locateanything-prompt-freeze",
    "--tag", "notebook-04.3", "--tag", "ds1", "--tag", "prompt-freeze", "--tag", "kaggle",
]
try:
    subprocess.run(
        tracking_command,
        env=tracking_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
except subprocess.TimeoutExpired:
    print("ClearML finalisation reached its wall-time limit after the experiment completed.")
if not receipt.is_file():
    receipt.write_text(json.dumps({"status": "not_recorded", "task_id": None}) + "\\n", encoding="utf-8")
display(pd.DataFrame([json.loads(receipt.read_text(encoding="utf-8"))]))
"""
    ),
    markdown("## Final output check"),
    code(
        """
manifest = json.loads((safe_root / "artifact_manifest.json").read_text(encoding="utf-8"))
run_record = json.loads((safe_root / "run_record.json").read_text(encoding="utf-8"))
final_summary = {
    "status": run_record["status"],
    "selected candidate": run_record["selected_candidate"],
    "evaluated images per prompt": run_record["image_count"],
    "candidate and policy combinations": run_record["candidate_count"],
    "safe artifact count": manifest["artifact_count"],
    "baseline reused": run_record["baseline_reused"],
    "model loads in this run": run_record["model_load_count"],
}
display(pd.DataFrame([final_summary]))
"""
    ),
    markdown(
        """
## Build compact download bundles

The packager validates the safe manifest before creating three deterministic archives. Repository evidence contains aggregate safe outputs. Review evidence contains the private prompt panels. Downstream input contains only the selected prompt record and selected proposal file required by the next stage. Credentials and Kaggle logs are excluded.
"""
    ),
    code(
        """
bundle_root = (
    Path("/kaggle/working/notebook04_3_bundles")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else OUTPUT_ROOT.parent / "notebook04_3_bundles"
)
verification_path = bundle_root / "ds1-locateanything-prompt-freeze-verification.json"
if not verification_path.is_file():
    # Keep aggregate, private-review, and downstream files in separate archives.
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/package_kaggle_stage_output.py"),
            "--stage-root", str(OUTPUT_ROOT / "evaluation"),
            "--output-root", str(bundle_root),
            "--stage-name", "ds1-locateanything-prompt-freeze",
            "--review-path", "private/prompt_examples",
            "--review-path", "clearml_receipt.json",
            "--downstream-path", "downstream",
        ],
        check=True,
        text=True,
    )
bundle_verification = json.loads(verification_path.read_text(encoding="utf-8"))
display(pd.DataFrame(bundle_verification["archives"]))
print(f"Compact downloads: {bundle_root}")
"""
    ),
    markdown(
        """
## Next use

The chosen prompt and postprocessing policy become an unchanged input to DS2. The next confirmation run must report proposal precision, recall, complete-detection rate, SAM mask quality, DINOv2 ranking results, B0 and P1 results, and multi-view consistency without another DS2-based tuning step. The automatic and assisted routes remain separate in the report.
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
