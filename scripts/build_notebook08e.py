#!/usr/bin/env python3
"""Build Notebook 08E for the frozen development pipeline."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "development-freeze"
NOTEBOOK_NAME = "08e-toysad8k-frozen-development-pipeline.ipynb"
METADATA_NAME = "kernel-metadata-08e.json"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K Frozen Development Pipeline

**Notebook file:** `08e-toysad8k-frozen-development-pipeline.ipynb`

This notebook applies the frozen automatic route to the complete development role. It processes the 100-scene screen and the remaining 600 development scenes with LocateAnything-3B, SAM 2.1 Hiera-large, DINOv2-small, B0 and P1. The existing 300-scene confirmation result is reused only when every split, model, configuration, preprocessing and schema hash matches.

Predictions are written and locked before any development evaluator resource can be opened. Calibration and internal-lock scenes remain sealed.
"""
    ),
    markdown(
        """
## Acceptance contract

- The input contains the fixed 100-scene screen and 600-scene development remainder, with all 20 views.
- LocateAnything uses the frozen prompt and `area_nms_075_cap8` proposal policy.
- SAM 2.1 and DINOv2-small use immutable model revisions and fixed preprocessing.
- Every stage writes content-addressed shards that an identical execution can resume.
- Zero-proposal, empty-mask and feature failures remain explicit terminal outcomes.
- Global DINOv2 embeddings are stored as hashed float16 arrays with an object-level index.
- The model-safe boundary contains `prediction_manifest.jsonl`, `global_embeddings.npy`, `embedding_index.jsonl` and `prediction_lock.json`.
- The prediction manifest and embeddings are locked before evaluator access.
- The 300-scene confirmation result is combined only after a complete identity check.
- The 6,000-view DS2 representation backfill has a separate bounded recovery bundle bound to its verified proposal and SAM records.
- Prediction figures contain no protected labels. Evaluator correspondence, when inspected, is marked reference-only and is not a deployment claim.
"""
    ),
    code(
        """
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from IPython.display import Image as NotebookImage, display

sys.dont_write_bytecode = True

NOTEBOOK_NAME = "08e-toysad8k-frozen-development-pipeline.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
RUN_PHASE = os.getenv("VPI_RUN_PHASE", "incremental").strip().lower()
if RUN_PHASE not in {"incremental", "full"}:
    raise ValueError("VPI_RUN_PHASE must be incremental or full")
MAXIMUM_NEW_SHARDS = int(os.getenv("VPI_MAXIMUM_NEW_SHARDS", "10"))
if MAXIMUM_NEW_SHARDS < 1:
    raise ValueError("VPI_MAXIMUM_NEW_SHARDS must be positive")
DS2_BACKFILL_PHASE = os.getenv("VPI_DS2_BACKFILL_PHASE", "incremental").strip().lower()
if DS2_BACKFILL_PHASE not in {"incremental", "full"}:
    raise ValueError("VPI_DS2_BACKFILL_PHASE must be incremental or full")
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/working/toysad8k_frozen_development_pipeline"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/toysad8k-frozen-development-pipeline",
)).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RECOVERY_OUTPUT_ROOT = Path(os.getenv(
    "VPI_RECOVERY_OUTPUT_ROOT",
    "/kaggle/working/notebook08e_recovery"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/notebook08e_recovery",
)).expanduser().resolve()
RECOVERY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DS2_RECOVERY_OUTPUT_ROOT = Path(os.getenv(
    "VPI_DS2_RECOVERY_OUTPUT_ROOT",
    "/kaggle/working/notebook08e_ds2_backfill_recovery"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/notebook08e_ds2_backfill_recovery",
)).expanduser().resolve()
DS2_RECOVERY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
"""
    ),
    code(
        r'''
def kaggle_roots() -> list[Path]:
    """Return bounded roots for both Kaggle input mount layouts."""

    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return []
    roots: list[Path] = []
    for pattern in (
        "*",
        "datasets/*/*",
        "datasets/*/*/versions/*",
        "notebooks/*/*",
        "notebooks/*/*/versions/*",
    ):
        roots.extend(path for path in input_root.glob(pattern) if path.is_dir())
    return sorted(set(roots), key=lambda path: path.as_posix())


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents, *kaggle_roots()))
    for candidate in candidates:
        if (candidate / "scripts/run_frozen_development_pipeline.py").is_file():
            return candidate.resolve()
        if (candidate / "code/scripts/run_frozen_development_pipeline.py").is_file():
            return (candidate / "code").resolve()
    mounted_names = [path.as_posix() for path in kaggle_roots()]
    raise FileNotFoundError(
        f"Could not locate the Visual Peer Inspector code snapshot. Mounted inputs: {mounted_names}"
    )


def find_model_input(cohort_id: str, variable: str) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(kaggle_roots())
    for mounted in kaggle_roots():
        candidates.extend(path for path in mounted.iterdir() if path.is_dir())
    for candidate in candidates:
        record_path = candidate / "package_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            record.get("cohort_id") == cohort_id
            and (candidate / "neutral_corpus_manifest.jsonl").is_file()
            and (candidate / "media").is_dir()
        ):
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate model-safe cohort {cohort_id}")


def find_named_file(name: str, variable: str) -> Path:
    configured = os.getenv(variable)
    if configured and Path(configured).expanduser().is_file():
        return Path(configured).expanduser().resolve()
    matches = []
    for root in kaggle_roots():
        matches.extend(root.rglob(name))
    files = sorted({path.resolve() for path in matches if path.is_file()})
    if not files:
        raise FileNotFoundError(f"Could not locate {name}")
    digests = {sha256(path.read_bytes()).hexdigest() for path in files}
    if len(digests) != 1:
        raise ValueError(f"Mounted copies of {name} differ")
    return files[0]


def find_directory(marker: str, variable: str) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    for root in kaggle_roots():
        candidates.extend((root, *(path for path in root.iterdir() if path.is_dir())))
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate {variable} containing {marker}")


def find_recovery_pair(
    stem: str = "notebook08e",
    environment_name: str = "VPI_RESUME_ROOT",
) -> tuple[Path, Path] | None:
    configured = os.getenv(environment_name)
    sources = [Path(configured).expanduser()] if configured else kaggle_roots()
    bundles = sorted({
        path.resolve()
        for source in sources
        for path in source.rglob(f"{stem}-recovery.zip")
        if path.is_file()
    })
    manifests = sorted({
        path.resolve()
        for source in sources
        for path in source.rglob(f"{stem}-recovery-manifest.json")
        if path.is_file()
    })
    if not bundles and not manifests:
        return None
    if len(bundles) != 1 or len(manifests) != 1:
        raise ValueError(f"{environment_name} must contain exactly one {stem} recovery pair")
    return bundles[0], manifests[0]


def code_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(CODE_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not current else source_root + os.pathsep + current
    return environment


def read_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.getenv(name)


def finish_clearml(safe_root: Path, task_name: str, receipt: Path, result_tag: str) -> dict:
    """Send aggregate evidence through a bounded subprocess after inference."""

    mode = "online" if EXECUTION_ENVIRONMENT == "kaggle" else "disabled"
    environment = code_environment()
    if mode == "online":
        access = read_secret("CLEARML_API_ACCESS_KEY")
        secret = read_secret("CLEARML_API_SECRET_KEY")
        if not access or not secret:
            mode = "disabled"
        else:
            environment.update({
                "CLEARML_API_ACCESS_KEY": access,
                "CLEARML_API_SECRET_KEY": secret,
                "CLEARML_API_HOST": "https://api.clear.ml",
                "CLEARML_WEB_HOST": "https://app.clear.ml",
                "CLEARML_FILES_HOST": "https://files.clear.ml",
            })
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts/finalize_ds2_clearml.py"),
        "--safe-root", str(safe_root),
        "--receipt", str(receipt),
        "--stage", "frozen_development_pipeline",
        "--project", "Visual Peer Inspector/05 Development Freeze",
        "--task-name", task_name,
        "--tag", "notebook-08e",
        "--tag", "development-freeze",
        "--tag", "kaggle",
        "--tag", result_tag,
        "--mode", mode,
        "--wall-time-seconds", "180",
    ]
    try:
        subprocess.run(command, check=True, timeout=240, env=environment)
    except Exception as error:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "task_id": None,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(receipt.read_text(encoding="utf-8"))
'''
    ),
    markdown("## Resolve the frozen inputs"),
    code(
        """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/development-freeze/08e-toysad8k-frozen-development-pipeline.ipynb",
)

DS1_ROOT = find_model_input("toysad8k-development-screen-100-v1", "VPI_DS1_MODEL_INPUT_ROOT")
REMAINDER_ROOT = find_model_input("toysad8k-development-remainder-600-v1", "VPI_REMAINDER_MODEL_INPUT_ROOT")
DS2_RGB_ROOT = find_model_input("toysad8k-development-confirm-300-v1", "VPI_DS2_RGB_ROOT")
DS2_RUN_ROOT = find_directory("mask-routes/package_record.json", "VPI_DS2_RUN_ROOT")
SELECTED_PROMPT = find_named_file("selected_prompt.json", "VPI_SELECTED_PROMPT")

display({
    "execution environment": EXECUTION_ENVIRONMENT,
    "run phase": RUN_PHASE,
    "maximum new shards per stage": MAXIMUM_NEW_SHARDS,
    "previous recovery attached": find_recovery_pair() is not None,
    "screen scenes": 100,
    "development remainder scenes": 600,
    "views per scene": 20,
    "confirmation backfill scenes": 300,
    "confirmation reuse boundary": "LocateAnything and SAM identities must verify",
})
"""
    ),
    markdown("## Verify the frozen identity and GPU runtime"),
    code(
        """
from visual_peer_inspector.workflows.frozen_development_pipeline import verify_prediction_lock
from visual_peer_inspector.models.sam2 import Sam2Configuration
from visual_peer_inspector.data.model_safe_package import file_sha256

checkpoint_path = None

install_environment = code_environment()
install_environment["SAM2_BUILD_CUDA"] = "0"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-lock.txt")],
    check=True,
    timeout=900,
    env=install_environment,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
    ],
    check=True,
    timeout=600,
    env=install_environment,
)

import torch
from huggingface_hub import hf_hub_download

if not torch.cuda.is_available():
    raise RuntimeError("Notebook 08E requires a Kaggle GPU")
sam_configuration = Sam2Configuration()
checkpoint_path = Path(hf_hub_download(
    repo_id=sam_configuration.model_repository,
    filename=sam_configuration.checkpoint_filename,
    revision=sam_configuration.model_revision,
    cache_dir="/tmp/vpi-sam21-model-cache",
))
if (
    checkpoint_path.stat().st_size != sam_configuration.checkpoint_size_bytes
    or file_sha256(checkpoint_path) != sam_configuration.checkpoint_sha256
):
    raise RuntimeError("The pinned SAM 2.1 checkpoint identity differs")

display({
    "GPU": torch.cuda.get_device_name(0),
    "GPU count": torch.cuda.device_count(),
    "LocateAnything": "nvidia/LocateAnything-3B at c32291ca5e996f5a7a485845b4f57a233936bba0",
    "SAM": "facebook/sam2.1-hiera-large at 665f8e2ad61cf5f53d65644ff27c8ee525124610",
    "DINO": "facebook/dinov2-small at ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "proposal policy": "area_nms_075_cap8",
})
"""
    ),
    markdown(
        """
## Run the frozen automatic route

The three neural models load sequentially so that their weights do not compete for GPU memory. Each stage is split into deterministic shards. A completed shard is reused only when its input hash, output hash, embedding hashes and full pipeline identity still match. Incremental runs stop after a bounded number of new shards per stage and write a verified recovery ZIP. Attach that private recovery output to the next Kaggle version through `VPI_RESUME_ROOT`. ClearML is not active during model loading or inference.
"""
    ),
    code(
        """
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_frozen_development_pipeline.py"),
    "run-development",
    "--ds1-model-input-root", str(DS1_ROOT),
    "--remainder-model-input-root", str(REMAINDER_ROOT),
    "--selected-prompt", str(SELECTED_PROMPT),
    "--sam-checkpoint", str(checkpoint_path),
    "--output-root", str(OUTPUT_ROOT),
    "--device", "cuda",
    "--shard-size", "200",
    "--run-mode", RUN_PHASE,
    "--maximum-new-shards", str(MAXIMUM_NEW_SHARDS),
    "--recovery-output-directory", str(RECOVERY_OUTPUT_ROOT),
]
recovery_pair = find_recovery_pair()
if recovery_pair is not None:
    command.extend(["--recovery-bundle", str(recovery_pair[0]), "--recovery-manifest", str(recovery_pair[1])])
preflight_path = OUTPUT_ROOT / "preflight/model_preflight.json"
preflight_command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_frozen_development_pipeline.py"),
    "preflight",
    "--cohort", "remainder",
    "--model-input-root", str(REMAINDER_ROOT),
    "--selected-prompt", str(SELECTED_PROMPT),
    "--sam-checkpoint", str(checkpoint_path),
    "--output-path", str(preflight_path),
    "--device", "cuda",
    "--maximum-image-attempts", "5",
]
subprocess.run(preflight_command, check=True, timeout=1800, env=code_environment())
preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
if preflight_record["status"] != "passed":
    raise RuntimeError("The bounded model preflight did not pass")
inference_environment = code_environment()
inference_environment["HF_HUB_OFFLINE"] = "1"
inference_environment["TRANSFORMERS_OFFLINE"] = "1"
subprocess.run(command, check=True, env=inference_environment)

screen_root = OUTPUT_ROOT / "toysad8k-development-screen-100-v1"
remainder_root = OUTPUT_ROOT / "toysad8k-development-remainder-600-v1"
screen_record = json.loads((screen_root / "safe/run_record.json").read_text(encoding="utf-8"))
remainder_record = json.loads((remainder_root / "safe/run_record.json").read_text(encoding="utf-8"))
development_progress = json.loads((OUTPUT_ROOT / "development_progress.json").read_text(encoding="utf-8"))
PIPELINE_COMPLETE = development_progress["status"] == "complete"
display(pd.DataFrame([
    {
        "cohort": record["cohort_id"],
        "scenes": record["scene_count"],
        "images": record["image_count"],
        "audit panels": record["audit_panel_count"],
        "status": record["status"],
    }
    for record in (screen_record, remainder_record)
]))
display({
    "model preflight": preflight_record["status"],
    "pipeline complete": PIPELINE_COMPLETE,
    "recovery ZIP": development_progress.get("recovery", {}).get("bundle"),
    "recovery SHA-256": development_progress.get("recovery", {}).get("bundle_sha256"),
    "next action": "finalise the development result" if PIPELINE_COMPLETE else "publish the recovery output privately and run the next incremental version",
})
"""
    ),
    markdown(
        """
## Backfill the confirmed 300-scene route

The earlier confirmation output predates the 08E prediction bundle. Its LocateAnything and automatic SAM results are reused only after the exact model revision, prompt, proposal policy, configuration, result hash and schema fields verify. DINOv2-small, B0 and P1 are then rebuilt under the current frozen identity. A missing or different field stops the notebook and requires a complete confirmation rerun.
"""
    ),
    code(
        """
confirmation_root = OUTPUT_ROOT / "toysad8k-development-confirm-300-v1"
confirmation_record = None
CONFIRMATION_COMPLETE = False
if PIPELINE_COMPLETE:
    existing_confirmation_lock = confirmation_root / "model-safe/prediction_lock.json"
    if existing_confirmation_lock.is_file():
        verify_prediction_lock(existing_confirmation_lock)
    else:
        backfill_command = [
            sys.executable,
            str(CODE_ROOT / "scripts/run_frozen_development_pipeline.py"),
            "backfill-ds2",
            "--rgb-root", str(DS2_RGB_ROOT),
            "--mask-route-root", str(DS2_RUN_ROOT / "mask-routes"),
            "--proposal-record", str(DS2_RUN_ROOT / "locateanything/model-input/package_record.json"),
            "--automatic-sam-record", str(DS2_RUN_ROOT / "sam21/automatic-box/safe/run_record.json"),
            "--selected-prompt", str(SELECTED_PROMPT),
            "--output-root", str(OUTPUT_ROOT),
            "--device", "cuda",
            "--shard-size", "200",
            "--run-mode", DS2_BACKFILL_PHASE,
            "--maximum-new-shards", str(MAXIMUM_NEW_SHARDS),
            "--recovery-output-directory", str(DS2_RECOVERY_OUTPUT_ROOT),
        ]
        ds2_recovery_pair = find_recovery_pair(
            "notebook08e-ds2-backfill",
            "VPI_DS2_RESUME_ROOT",
        )
        if ds2_recovery_pair is not None:
            backfill_command.extend(["--recovery-bundle", str(ds2_recovery_pair[0]), "--recovery-manifest", str(ds2_recovery_pair[1])])
        subprocess.run(backfill_command, check=True, env=code_environment())
    confirmation_record = json.loads((confirmation_root / "safe/run_record.json").read_text(encoding="utf-8"))
    CONFIRMATION_COMPLETE = confirmation_record["status"] == "complete"
    display({
        "status": confirmation_record["status"],
        "reused stages": confirmation_record["reused_stages"],
        "rebuilt stages": confirmation_record["rebuilt_stages"],
        "scenes": confirmation_record["scene_count"],
        "images": confirmation_record["image_count"],
        "backfill phase": DS2_BACKFILL_PHASE,
        "final lock": CONFIRMATION_COMPLETE,
    })
else:
    display({"confirmation backfill": "deferred until both GPU cohorts are complete"})
"""
    ),
    markdown("## Terminal outcomes and coverage"),
    code(
        """
stage_rows = []
available_records = [screen_record, remainder_record]
if confirmation_record is not None:
    available_records.append(confirmation_record)
for record in available_records:
    for stage_name, stage in record["stages"].items():
        stage_rows.append({
            "cohort": record["cohort_id"],
            "stage": stage_name,
            "inputs": stage["input_count"],
            "processed this run": stage["processed_this_run"],
            "resumed": stage["resumed"],
            "terminal failures": stage["terminal_failures"],
            "zero proposal": stage["failure_counts"]["zero_proposal"],
            "empty mask": stage["failure_counts"]["empty_mask"],
            "feature failure": stage["failure_counts"]["feature_failure"],
        })
stage_table = pd.DataFrame(stage_rows)
display(stage_table)
stage_table.to_csv(OUTPUT_ROOT / "stage_coverage.csv", index=False)
"""
    ),
    markdown("## Locked model-safe artifacts"),
    code(
        """
lock_rows = []
if PIPELINE_COMPLETE and CONFIRMATION_COMPLETE:
    for root in (screen_root, remainder_root, confirmation_root):
        lock_path = root / "model-safe/prediction_lock.json"
        lock = verify_prediction_lock(lock_path)
        lock_rows.append({
            "cohort": lock["cohort_id"],
            "scenes": lock["scene_count"],
            "images": lock["image_count"],
            "artifacts": lock["artifact_count"],
            "identity SHA-256": lock["identity_sha256"],
            "lock SHA-256": file_sha256(lock_path),
        })
lock_table = pd.DataFrame(lock_rows)
display(lock_table)
lock_table.to_csv(OUTPUT_ROOT / "prediction_locks.csv", index=False)
"""
    ),
    markdown("## Prediction-only qualitative evidence"),
    code(
        """
figure_paths = []
figure_roots = [screen_root, remainder_root]
if CONFIRMATION_COMPLETE:
    figure_roots.append(confirmation_root)
for root in figure_roots:
    if not (root / "review-only/audit_evidence.json").is_file():
        continue
    evidence = json.loads((root / "review-only/audit_evidence.json").read_text(encoding="utf-8"))
    for item in evidence["items"]:
        figure_paths.extend((root / item["prediction_panel"], root / item["sam_evidence_panel"]))

for path in figure_paths[:8]:
    display(NotebookImage(filename=str(path)))
display({"prediction-only audit figures": len(figure_paths)})
"""
    ),
    markdown("## Verify and combine the full development role"),
    code(
        """
combined_path = OUTPUT_ROOT / "full-development-lock.json"
development_record = None
if PIPELINE_COMPLETE and CONFIRMATION_COMPLETE:
    subprocess.run([
        sys.executable,
        str(CODE_ROOT / "scripts/run_frozen_development_pipeline.py"),
        "combine",
        "--ds1-lock", str(screen_root / "model-safe/prediction_lock.json"),
        "--remainder-lock", str(remainder_root / "model-safe/prediction_lock.json"),
        "--ds2-lock", str(confirmation_root / "model-safe/prediction_lock.json"),
        "--output-path", str(combined_path),
    ], check=True, env=code_environment())
    development_record = json.loads(combined_path.read_text(encoding="utf-8"))
    display({
        "status": development_record["status"],
        "development scenes represented": development_record["scene_count"],
        "confirmation reused": development_record.get("ds2_reused_only_after_full_identity_check", False),
    })
else:
    display({"combined development lock": "not published before exact full coverage"})
"""
    ),
    markdown(
        """
## Reference-only development analysis

The prediction locks above are the boundary between inference and evaluation. Any later use of the private development correspondence may measure alignment or compare multi-view fusion methods, but it cannot support a deployment-valid claim because a web upload does not provide evaluator correspondence. Notebook 08F performs that analysis without altering these predictions.
"""
    ),
    markdown("## Aggregate ClearML record"),
    code(
        """
tracking_rows = []
if PIPELINE_COMPLETE and CONFIRMATION_COMPLETE:
    for root, task_name in (
        (screen_root, "frozen-development-screen-100"),
        (remainder_root, "frozen-development-remainder-600"),
        (confirmation_root, "frozen-development-confirmation-300"),
    ):
        receipt = root / "tracking/clearml_receipt.json"
        tracking = finish_clearml(root / "safe", task_name, receipt, "final")
        tracking_rows.append({
            "cohort": root.name,
            "status": tracking["status"],
            "task_id": tracking.get("task_id"),
            "record kind": "final",
        })
else:
    recovery_safe = OUTPUT_ROOT / "tracking-safe"
    receipt = OUTPUT_ROOT / "tracking/clearml_recovery_receipt.json"
    tracking = finish_clearml(
        recovery_safe,
        "frozen-development-recovery",
        receipt,
        "recovery",
    )
    tracking_rows.append({
        "cohort": "development checkpoint",
        "status": tracking["status"],
        "task_id": tracking.get("task_id"),
        "record kind": "recovery",
    })
display(pd.DataFrame(tracking_rows))
"""
    ),
    markdown("## Final artifact register"),
    code(
        """
artifact_rows = []
for path in sorted(OUTPUT_ROOT.rglob("*")):
    if path.is_file() and not path.name.endswith(".incomplete"):
        artifact_rows.append({
            "path": path.relative_to(OUTPUT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
artifact_register = pd.DataFrame(artifact_rows)
artifact_register.to_csv(OUTPUT_ROOT / "artifact_register.csv", index=False)
display(artifact_register)
"""
    ),
    markdown(
        """
## Result boundary

The 100-scene screen and 600-scene remainder are complete when their prediction locks verify. The 300-scene DS2 DINOv2-small and P1 backfill may also use bounded recovery versions and cannot publish a lock before all 6,000 views are covered. The full 1,000-scene development result is complete only when all three locks pass the same identity contract. Calibration and internal-lock results are not part of this notebook.
"""
    ),
]


def build() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells):
        identity = f"{index}:{cell['cell_type']}:{cell['source']}".encode()
        cell["id"] = sha256(identity).hexdigest()[:8]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbformat.write(notebook, NOTEBOOK_ROOT / NOTEBOOK_NAME)
    metadata = {
        "id": "esreekarreddy/toysad-8k-frozen-development-pipeline",
        "title": "ToysAD-8K Frozen Development Pipeline",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-ds1-screen100-model-input",
            "esreekarreddy/vpi-toysad-development-remainder-model-input",
            "esreekarreddy/vpi-toysad-ds2-confirmation-rgb",
        ],
        "competition_sources": [],
        "kernel_sources": ["esreekarreddy/toysad-8k-ds2-frozen-pipeline"],
        "model_sources": [],
    }
    (NOTEBOOK_ROOT / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build()
