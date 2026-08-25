#!/usr/bin/env python3
"""Build the clean Kaggle source for Notebook 05.1."""

from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "notebooks"
    / "detection-and-segmentation"
    / "toysad8k-ds1-sam21"
    / "05.1-toysad8k-ds1-sam21-segmentation.ipynb"
)


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K DS1 Prompted SAM 2.1 Segmentation

**File:** `05.1-toysad8k-ds1-sam21-segmentation.ipynb`

This notebook applies the pinned SAM 2.1 model to the frozen reference-box and automatic-box prompts for the complete DS1 screen. It contains 100 development scenes and all 20 RGB views per scene.

The model run records mask coverage, repeatability, runtime, memory and failures. It does not measure mask accuracy. The evaluator-only stage performs IoU, Dice and boundary comparisons after these predictions are frozen.
"""
    ),
    markdown(
        """
## Run contract

- the model input contains 100 scenes and 2,000 views;
- both prompt routes are present for every view;
- a one-image preflight runs before the pilot;
- the 20-image pilot spans 20 scenes and all 20 view positions;
- the pilot is repeated exactly before full inference;
- completed image results are reused on restart or from one attached prior Kaggle output; and
- private masks and visual examples stay outside the ClearML and GitHub evidence roots.

SAM predicted quality is model evidence, not a substitute for evaluator metrics.
"""
    ),
    code(
        """
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True

from IPython.display import HTML, Image as NotebookImage, display


NOTEBOOK_NAME = "05.1-toysad8k-ds1-sam21-segmentation.ipynb"
NOTEBOOK_RELATIVE = Path(
    "notebooks/detection-and-segmentation/toysad8k-ds1-sam21"
) / NOTEBOOK_NAME
NOTEBOOK_SLUG = "esreekarreddy/toysad-8k-ds1-sam-2-1-segmentation"
OUTPUT_ROOT = Path("/kaggle/working/toysad8k_ds1_sam21_segmentation")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"


def emit_notebook_stage(stage: str, **details) -> None:
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)
"""
    ),
    markdown("## Locate and verify the frozen inputs"),
    code(
        """
def locate_directory(marker: str) -> Path:
    marker_path = Path(marker)
    candidates: list[Path] = []
    configured_root = os.getenv("VPI_CODE_ROOT") if marker.startswith("src/") else None
    if configured_root:
        candidates.append(Path(configured_root).expanduser())
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for path in input_root.rglob(marker_path.name):
            if path.parts[-len(marker_path.parts):] == marker_path.parts:
                candidates.append(path.parents[len(marker_path.parts) - 1])
    candidates.extend((Path.cwd(), *Path.cwd().parents))
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate an input containing {marker}")


CODE_ROOT = locate_directory("src/visual_peer_inspector/workflows/sam_ds1.py")
PROMPT_ROOT = locate_directory("sam_prompt_manifest.jsonl")
MODEL_INPUT_ROOT = locate_directory("neutral_corpus_manifest.jsonl")
MEDIA_ROOT = MODEL_INPUT_ROOT / "media"
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    file_sha256,
    verify_model_input_package,
)
from visual_peer_inspector.data.sam_prompt_package_ds1 import (
    verify_ds1_sam_prompt_package,
)
from visual_peer_inspector.verification import verify_code_snapshot_record
from visual_peer_inspector.workflows.sam_ds1 import (
    build_ds1_sam_jobs,
    import_ds1_sam_resume,
)


def locate_resume_root() -> Path | None:
    configured = os.getenv("VPI_DS1_SAM_RESUME_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve(strict=True)
        if not (candidate / "private" / "image_results").is_dir():
            raise FileNotFoundError("VPI_DS1_SAM_RESUME_ROOT has no image results")
        return candidate
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return None
    candidates = {
        path.parent.parent.resolve()
        for path in input_root.rglob("image_results")
        if path.is_dir() and path.parent.name == "private"
    }
    if len(candidates) > 1:
        raise RuntimeError("More than one DS1 SAM resume input is attached")
    return next(iter(candidates), None)


snapshot_path = CODE_ROOT / "snapshot_record.json"
snapshot = verify_code_snapshot_record(CODE_ROOT, snapshot_path)
if snapshot.get("git_dirty") is not False or snapshot.get("precommit_candidate") is not False:
    raise RuntimeError("Notebook 05.1 requires a clean committed code snapshot")
prompt_record = verify_ds1_sam_prompt_package(PROMPT_ROOT)
model_audit = verify_model_input_package(
    MODEL_INPUT_ROOT,
    contract=DS1_SCREEN_CONTRACT,
)
model_manifest_path = MODEL_INPUT_ROOT / "neutral_corpus_manifest.jsonl"
if file_sha256(model_manifest_path) != prompt_record["model_manifest_sha256"]:
    raise RuntimeError("The DS1 SAM prompts identify another model-input package")
resume_root = locate_resume_root()
resume_report = (
    import_ds1_sam_resume(resume_root, OUTPUT_ROOT)
    if resume_root is not None
    else {
        "used": False,
        "copied_result_files": 0,
        "reused_result_files": 0,
        "attempt_history_copied": False,
        "attempt_history_policy": "not_imported_without_bound_run_identity",
    }
)
jobs = build_ds1_sam_jobs(PROMPT_ROOT, MEDIA_ROOT)
display(
    {
        "cohort": DS1_SCREEN_CONTRACT.cohort_id,
        "scenes": len({job.opaque_scene_id for job in jobs}),
        "images": len(jobs),
        "reference prompts": sum(
            prompt.route == "reference_box" for job in jobs for prompt in job.prompts
        ),
        "automatic prompts": sum(
            prompt.route == "automatic_box" for job in jobs for prompt in job.prompts
        ),
        "model input checks passed": model_audit.all_checks_passed,
        "resume": resume_report,
        "snapshot": snapshot["snapshot_sha256"],
    }
)
"""
    ),
    markdown("## Prepare and verify the pinned runtime"),
    code(
        """
install_started = time.perf_counter()
model_load_checkpoints: list[dict] = []


def record_model_load_stage(stage: str, **details) -> None:
    row = {
        "stage": stage,
        "elapsed_seconds": round(time.perf_counter() - install_started, 3),
        **details,
    }
    model_load_checkpoints.append(row)
    emit_notebook_stage(stage, **{key: value for key, value in row.items() if key != "stage"})


install_environment = os.environ.copy()
install_environment["SAM2_BUILD_CUDA"] = "0"
record_model_load_stage("dependency_install_started")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/sam2-lock.txt")],
    env=install_environment,
    timeout=300,
)
record_model_load_stage("dependency_install_completed")
record_model_load_stage("sam_repo_install_started")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
    ],
    env=install_environment,
    timeout=600,
)
record_model_load_stage("sam_repo_install_completed")

from visual_peer_inspector.models.sam2 import Sam2Configuration


configuration = Sam2Configuration()
distribution = importlib.metadata.distribution("SAM-2")
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
if direct_url.get("vcs_info", {}).get("commit_id") != configuration.code_revision:
    raise RuntimeError("The installed SAM source revision is not pinned")
if str(direct_url.get("url", "")).removesuffix(".git") != configuration.code_repository:
    raise RuntimeError("The installed SAM repository is not pinned")
display(
    {
        "package": distribution.metadata["Name"],
        "version": distribution.version,
        "revision": configuration.code_revision,
    }
)
"""
    ),
    markdown("## Verify the GPU and model checkpoint"),
    code(
        """
import torch
from huggingface_hub import hf_hub_download


if not torch.cuda.is_available():
    raise RuntimeError("Notebook 05.1 requires a Kaggle GPU")
device_index = 0
device = torch.cuda.get_device_properties(device_index)
architecture = f"sm_{device.major}{device.minor}"
gpu_evidence = {
    "cuda_available": True,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(device_index),
    "compute_capability": f"{device.major}.{device.minor}",
    "total_memory_gib": round(device.total_memory / (1024**3), 3),
    "torch_architecture": architecture,
    "torch_architecture_supported": architecture in set(torch.cuda.get_arch_list()),
}
if not gpu_evidence["torch_architecture_supported"]:
    raise RuntimeError("The installed PyTorch build does not support this GPU")
record_model_load_stage("checkpoint_download_started")
download_started = time.perf_counter()
checkpoint_path = Path(
    hf_hub_download(
        repo_id=configuration.model_repository,
        filename=configuration.checkpoint_filename,
        revision=configuration.model_revision,
        cache_dir="/tmp/vpi-sam21-model-cache",
    )
)
from visual_peer_inspector.data.model_safe_package import file_sha256


checkpoint_sha256 = file_sha256(checkpoint_path)
if checkpoint_path.stat().st_size != configuration.checkpoint_size_bytes:
    raise RuntimeError("The downloaded SAM checkpoint size is incorrect")
if checkpoint_sha256 != configuration.checkpoint_sha256:
    raise RuntimeError("The downloaded SAM checkpoint hash is incorrect")
record_model_load_stage(
    "checkpoint_download_completed",
    download_seconds=round(time.perf_counter() - download_started, 3),
    checkpoint_size_bytes=checkpoint_path.stat().st_size,
    checkpoint_sha256=checkpoint_sha256,
)
display({**gpu_evidence, "checkpoint_sha256": checkpoint_sha256})
"""
    ),
    markdown(
        """
## Generate the two mask routes

ClearML is not started in this section. Completed image files and attempt records are durable inside the output folder. If a previous private Kaggle output is attached, its atomic checkpoints are verified against the frozen jobs before accepted images are reused.
"""
    ),
    code(
        """
from visual_peer_inspector.workflows.sam_ds1 import run_ds1_sam_workflow


snapshot_versions = snapshot.get("platform_versions", {})


def required_version(key: str) -> str:
    value = os.getenv(f"VPI_{key.upper()}") or snapshot_versions.get(key)
    if not value or str(value).strip().lower() in {"pending", "unknown", "unavailable"}:
        raise RuntimeError(f"The code snapshot is missing version pin: {key}")
    return str(value)


version_pins = {
    key: required_version(key)
    for key in (
        "kaggle_notebook_version_05_1",
        "code_dataset_version",
    )
}
notebook_path = CODE_ROOT / NOTEBOOK_RELATIVE
if not notebook_path.is_file():
    raise FileNotFoundError("The code snapshot does not contain the exact Notebook 05.1 source")
provenance = {
    "git_commit": snapshot["git_commit"],
    "git_dirty": snapshot["git_dirty"],
    "snapshot_sha256": snapshot["snapshot_sha256"],
    "precommit_candidate": snapshot["precommit_candidate"],
    "notebook_sha256": file_sha256(notebook_path),
    "kaggle_notebook_slug": NOTEBOOK_SLUG,
    "prompt_package_record_sha256": file_sha256(PROMPT_ROOT / "package_record.json"),
    "prompt_manifest_sha256": file_sha256(PROMPT_ROOT / "sam_prompt_manifest.jsonl"),
    "model_input_package_record_sha256": file_sha256(MODEL_INPUT_ROOT / "package_record.json"),
    "model_input_manifest_sha256": file_sha256(model_manifest_path),
    "resume_import": resume_report,
    **version_pins,
}
platform_versions = {
    "python": platform.python_version(),
    "numpy": importlib.metadata.version("numpy"),
    "pillow": importlib.metadata.version("pillow"),
    "torch": importlib.metadata.version("torch"),
    "torchvision": importlib.metadata.version("torchvision"),
    **version_pins,
}


def report_progress(stage: str, details: dict) -> None:
    if stage in {"sam_import_started", "sam_model_build_started", "sam_worker_ready"}:
        if not any(row["stage"] == stage for row in model_load_checkpoints):
            model_load_checkpoints.append(
                {
                    "stage": stage,
                    "elapsed_seconds": round(time.perf_counter() - install_started, 3),
                }
            )
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


run_record = run_ds1_sam_workflow(
    prompt_root=PROMPT_ROOT,
    media_root=MEDIA_ROOT,
    output_root=OUTPUT_ROOT,
    configuration=configuration,
    run_mode="model_inference",
    inference_stage="full",
    checkpoint_path=checkpoint_path,
    checkpoint_sha256=checkpoint_sha256,
    provenance=provenance,
    platform_versions=platform_versions,
    gpu_evidence=gpu_evidence,
    model_load_rows=model_load_checkpoints,
    progress_callback=report_progress,
)
if not run_record["completion_gate_passed"]:
    print(
        "WARNING: Notebook 05.1 materialized a non-promotable run. "
        "The output remains downloadable for diagnosis, but ClearML and downstream "
        "verification must reject it until every acceptance gate passes.",
        flush=True,
    )
display(run_record)
"""
    ),
    markdown("## Operational evidence"),
    code(
        """
import pandas as pd
import matplotlib.pyplot as plt


safe_root = OUTPUT_ROOT / "safe"
display(HTML("<h4>Completion gates</h4>"))
display(pd.read_csv(safe_root / "completion_gates.csv"))
display(HTML("<h4>Route summary</h4>"))
display(pd.read_csv(safe_root / "route_summary.csv"))
display(HTML("<h4>View summary</h4>"))
display(pd.read_csv(safe_root / "view_summary.csv"))
chart_specs = [
    ("runtime_distribution.png", "Runtime"),
    ("runtime_by_prompt_count.png", "Runtime by prompt count"),
    ("mask_count_distribution.png", "Masks per image"),
    ("mask_count_by_route.png", "Masks by route"),
    ("reference_box_mask_area_distribution.png", "Reference-box mask area"),
    ("automatic_box_mask_area_distribution.png", "Automatic-box mask area"),
    ("reference_box_predicted_score_distribution.png", "Reference-box quality score"),
    ("automatic_box_predicted_score_distribution.png", "Automatic-box quality score"),
]
figure, axes = plt.subplots(2, 4, figsize=(18, 8.4))
for axis, (name, title) in zip(axes.flat, chart_specs, strict=True):
    axis.imshow(plt.imread(safe_root / name))
    axis.set_title(title)
    axis.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## Private qualitative route examples

These images are retained in Kaggle output for review and report preparation. They are not part of the safe GitHub or ClearML evidence set.
"""
    ),
    code(
        """
route_examples = OUTPUT_ROOT / "private" / "mask_route_examples.png"
if route_examples.is_file():
    display(NotebookImage(filename=str(route_examples)))
else:
    print(
        "No qualitative panel is available because the pilot did not complete. "
        "Use the failed run record, gates and attempt summary for diagnosis.",
        flush=True,
    )
"""
    ),
    markdown(
        """
## Bounded ClearML finalisation

Tracking starts only after inference, evidence generation and artifact manifests finish. The isolated finaliser uses the official ClearML hosts and retries only transient task-creation connection failures. Its receipt stores sanitized categories and retry counts without exception text. A ClearML failure does not remove the completed model output.
"""
    ),
    code(
        """
def finalizer_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(CODE_ROOT / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    environment.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    environment.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    environment.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    environment.setdefault("CLEARML_LOG_ENVIRONMENT", "")
    return environment


tracking_receipt = OUTPUT_ROOT / "private" / "tracking_receipt.json"
tracking_result = {"status": "not_started", "clearml_task_id": None}
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "clearml==2.1.10"],
        text=True,
        capture_output=True,
        timeout=180,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/finalize_ds1_sam_clearml.py"),
            "--safe-root",
            str(safe_root),
            "--receipt",
            str(tracking_receipt),
            "--mode",
            "online",
            "--credentials-source",
            "kaggle" if EXECUTION_ENVIRONMENT == "kaggle" else "environment",
            "--wall-time-seconds",
            "180",
            "--tag",
            "candidate",
            "--tag",
            "notebook-05.1",
            "--tag",
            "ds1",
            "--tag",
            "sam2.1",
            "--tag",
            "kaggle",
            "--tag",
            "full",
        ],
        env=finalizer_environment(),
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
    markdown("## Final result"),
    code(
        """
final_summary = {
    "status": run_record["status"],
    "completion_gate_passed": run_record["completion_gate_passed"],
    "scenes": run_record["summary"]["scene_count"],
    "images": run_record["summary"]["image_count"],
    "masks": run_record["summary"]["mask_count"],
    "empty_masks": run_record["summary"]["empty_mask_count"],
    "repeat_mask_agreement": run_record["repeat_agreement"]["mask_digest_agreement_rate"],
    "repeat_score_agreement": run_record["repeat_agreement"]["score_agreement_rate"],
    "median_seconds_per_image": run_record["summary"]["median_image_runtime_seconds"],
    "p95_seconds_per_image": run_record["summary"]["p95_image_runtime_seconds"],
    "peak_gpu_memory_gib": run_record["gpu"].get("peak_allocated_gib"),
    "tracking_status": tracking_result["status"],
    "clearml_task_id": tracking_result.get("clearml_task_id"),
}
from visual_peer_inspector.workflows.sam_ds1 import _write_artifact_manifest


_write_artifact_manifest(OUTPUT_ROOT / "private")
display(final_summary)
if not run_record["completion_gate_passed"]:
    print(
        "WARNING: Notebook 05.1 finished with failed acceptance gates. "
        "Do not publish or promote this run; retain it only as diagnostic evidence.",
        flush=True,
    )
"""
    ),
]


notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
