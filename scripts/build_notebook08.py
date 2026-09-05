#!/usr/bin/env python3
"""Build the three clean Notebook 08 experiment entry points."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "confirmation-and-multiview"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def notebook(cells: list) -> nbformat.NotebookNode:
    return nbformat.v4.new_notebook(
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


COMMON_LOCATORS = r'''
def kaggle_dataset_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return []
    candidates: list[Path] = []
    for pattern in (
        "*",
        "datasets/*/*",
        "datasets/*/*/versions/*",
        "notebooks/*/*",
        "notebooks/*/*/versions/*",
    ):
        candidates.extend(path for path in input_root.glob(pattern) if path.is_dir())
    return sorted(set(candidates), key=lambda path: path.as_posix())


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents))
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for mounted in kaggle_dataset_roots():
            candidates.extend((mounted, mounted / "code"))
    for candidate in candidates:
        if (candidate / "src/visual_peer_inspector/data/cohort_contract.py").is_file():
            return candidate.resolve()
    mounted_names = sorted(path.name for path in input_root.iterdir()) if input_root.is_dir() else []
    raise FileNotFoundError(
        f"Could not locate the Visual Peer Inspector code snapshot. Mounted inputs: {mounted_names}"
    )


def find_package(package_kind: str, variable: str) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for mounted in kaggle_dataset_roots():
            candidates.append(mounted)
            candidates.extend(
                child for child in sorted(mounted.iterdir()) if child.is_dir()
            )
    for candidate in candidates:
        record_path = candidate / "package_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        is_rgb_package = (
            package_kind == "toysad8k-model-input"
            and record.get("cohort_id") == "toysad8k-development-confirm-300-v1"
            and (candidate / "neutral_corpus_manifest.jsonl").is_file()
            and (candidate / "media").is_dir()
        )
        if record.get("package_kind") == package_kind or is_rgb_package:
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate package kind: {package_kind}")


def find_named_input_file(name: str, variable: str) -> Path:
    configured = os.getenv(variable)
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
    patterns = (
        name,
        f"*/{name}",
        f"*/*/{name}",
        f"*/*/*/{name}",
        f"*/*/*/*/{name}",
    )
    matches = []
    for mounted in kaggle_dataset_roots():
        for pattern in patterns:
            matches.extend(path for path in mounted.glob(pattern) if path.is_file())
    unique = sorted(set(matches), key=lambda path: path.as_posix())
    if not unique:
        raise FileNotFoundError(f"Could not find mounted input file: {name}")
    digests = {sha256(path.read_bytes()).hexdigest() for path in unique}
    if len(digests) != 1:
        raise ValueError(f"Mounted copies of {name} do not match")
    preferred = sorted(
        unique,
        key=lambda path: ("downstream" not in path.parts, path.as_posix()),
    )
    return preferred[0].resolve()


def read_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.getenv(name)


def code_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(CODE_ROOT / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not previous else source_root + os.pathsep + previous
    return environment


def run_clearml_finalizer(safe_root: Path, stage: str, task_name: str, tags: list[str], receipt: Path) -> dict:
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
        "--stage", stage,
        "--task-name", task_name,
        "--mode", mode,
    ]
    for tag in tags:
        command.extend(("--tag", tag))
    try:
        subprocess.run(command, env=environment, check=True, timeout=240)
    except Exception as error:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "task_id": None,
        }, indent=2) + "\\n", encoding="utf-8")
    return json.loads(receipt.read_text(encoding="utf-8"))
'''


release_cells = [
    markdown(
        """
# ToysAD-8K DS2 Confirmation Release

**Notebook file:** `08a-toysad8k-ds2-confirmation-release.ipynb`

This notebook prepares the untouched 300-scene development confirmation cohort. It keeps all 20 views, so the release contains 6,000 RGB images. It creates three physically separate resources: model-safe RGB, model-safe supplied geometry, and evaluator-only labels and attributes.

This stage does not run SAM, DINO, peer scoring, or model selection.
"""
    ),
    markdown(
        """
## Acceptance contract

- The cohort must match the frozen `confirm_300` membership hash.
- All 300 scenes and all 6,000 views must decode and pass checksums.
- RGB and supplied geometry must not contain anomaly labels, categories, raw source IDs, or experimental roles.
- Evaluator attributes must remain in a separate output resource.
- A second build must reuse the RGB files without changing the package identity.
- Only aggregate tables and charts may be sent to ClearML.
"""
    ),
    code(
        """
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

from IPython.display import Image as NotebookImage, display

NOTEBOOK_NAME = "08a-toysad8k-ds2-confirmation-release.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/temp/toysad8k_ds2_confirmation_release" if EXECUTION_ENVIRONMENT == "kaggle" else ".local/runs/toysad8k-ds2-confirmation-release",
)).expanduser().resolve()
"""
    ),
    code(COMMON_LOCATORS),
    markdown("## Locate the trusted inputs"),
    code(
        """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))


def find_directory(marker: str, variable: str) -> Path:
    # Resolve only roots carrying the exact trusted marker for this release stage.
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for mounted in kaggle_dataset_roots():
            candidates.append(mounted)
            candidates.extend(
                child for child in sorted(mounted.iterdir()) if child.is_dir()
            )
    local_shared = CODE_ROOT.parent / "code" / ".local" / "toysad"
    candidates.extend((
        local_shared / "selection/toysad8k-train-2000-v2",
        local_shared / "selection/toysad8k-train-2000-v2-development",
        local_shared / "selection/toysad8k-train-2000-v2-administrative-vault",
        local_shared / "kaggle/toysad8k-full-private-v1",
    ))
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate {variable} containing {marker}")


SELECTION_ROOT = find_directory("neutral_corpus_manifest.jsonl", "VPI_SELECTION_ROOT")
DEVELOPMENT_ROOT = find_directory("development_source_map.jsonl", "VPI_DEVELOPMENT_ROOT")
ADMINISTRATIVE_ROOT = find_directory("required_source_members.jsonl", "VPI_ADMINISTRATIVE_ROOT")
SOURCE_ROOT = find_directory("archive_index.sqlite3", "VPI_SOURCE_ROOT")

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT

display({
    "cohort": DS2_CONFIRM_CONTRACT.cohort_id,
    "scenes": DS2_CONFIRM_CONTRACT.scene_count,
    "views per scene": DS2_CONFIRM_CONTRACT.views_per_scene,
    "images": DS2_CONFIRM_CONTRACT.image_count,
    "asset groups": DS2_CONFIRM_CONTRACT.asset_group_count,
})
"""
    ),
    markdown("## Build and verify the three separated packages"),
    code(
        """
# The builder reads protected records only in this trusted preparation job.
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/build_ds2_confirmation_inputs.py"),
    "--selection-root", str(SELECTION_ROOT),
    "--development-root", str(DEVELOPMENT_ROOT),
    "--administrative-root", str(ADMINISTRATIVE_ROOT),
    "--source-root", str(SOURCE_ROOT),
    "--output-root", str(OUTPUT_ROOT),
]
subprocess.run(command, check=True, env=code_environment())
first_release_record = json.loads((OUTPUT_ROOT / "run_record.json").read_text(encoding="utf-8"))

# The second pass verifies restart behaviour without rereading the 49 GiB source archive.
subprocess.run(command + ["--skip-archive-byte-check"], check=True, env=code_environment())
release_record = json.loads((OUTPUT_ROOT / "run_record.json").read_text(encoding="utf-8"))
identity_fields = ("membership_sha256", "manifest_sha256", "content_checksums_sha256", "image_count")
repeat_identity_matches = all(
    first_release_record["rgb_package"].get(field) == release_record["rgb_package"].get(field)
    for field in identity_fields
)
display({
    "status": release_record["status"],
    "RGB images verified": release_record["rgb_package"]["verified_image_count"],
    "first pass images written": first_release_record["rgb_build"]["files_written_this_run"],
    "second pass images reused": release_record["rgb_build"]["files_reused_this_run"],
    "package identity stable": repeat_identity_matches,
    "visible object masks": release_record["geometry_package"]["visible_object_view_count"],
    "protected scenes": release_record["attributes_package"]["scene_count"],
})
"""
    ),
    markdown("## Aggregate package evidence"),
    code(
        """
import pandas as pd

from visual_peer_inspector.evidence import save_bar_chart
from visual_peer_inspector.workflows.sam_confirmation_ds2 import _artifact_manifest

SAFE_ROOT = OUTPUT_ROOT / "safe"
SAFE_ROOT.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame([
    {"resource": "RGB model input", "items": release_record["rgb_package"]["image_count"], "model_facing": True},
    {"resource": "supplied object masks", "items": release_record["geometry_package"]["visible_object_view_count"], "model_facing": True},
    {"resource": "protected scene labels", "items": release_record["attributes_package"]["scene_count"], "model_facing": False},
])
summary.to_csv(SAFE_ROOT / "resource_summary.csv", index=False)
gates = pd.DataFrame([
    {"gate": "frozen membership", "passed": release_record["cohort"]["membership_sha256"] == DS2_CONFIRM_CONTRACT.membership_sha256},
    {"gate": "RGB coverage", "passed": release_record["rgb_package"]["image_count"] == DS2_CONFIRM_CONTRACT.image_count},
    {"gate": "RGB integrity", "passed": release_record["rgb_package"]["all_checks_passed"]},
    {"gate": "repeat build reused all RGB files", "passed": release_record["rgb_build"]["files_reused_this_run"] == DS2_CONFIRM_CONTRACT.image_count and release_record["rgb_build"]["files_written_this_run"] == 0},
    {"gate": "repeat build kept package identity", "passed": repeat_identity_matches},
    {"gate": "model input labels absent", "passed": not release_record["boundary"]["model_input_contains_labels"]},
    {"gate": "evaluator separated", "passed": not release_record["boundary"]["evaluator_attributes_model_facing"]},
])
gates.to_csv(SAFE_ROOT / "validation_gates.csv", index=False)
publication = pd.DataFrame([
    {
        "resource": "RGB model input",
        "expected_kaggle_dataset": "vpi-toysad-ds2-confirmation-rgb",
        "package_sha256": release_record["rgb_package"]["content_checksums_sha256"],
        "items": release_record["rgb_package"]["image_count"],
    },
    {
        "resource": "supplied geometry",
        "expected_kaggle_dataset": "vpi-toysad-ds2-confirmation-geometry",
        "package_sha256": release_record["geometry_package"]["content_checksums_sha256"],
        "items": release_record["geometry_package"]["visible_object_view_count"],
    },
    {
        "resource": "protected evaluator attributes",
        "expected_kaggle_dataset": "vpi-toysad-ds2-confirmation-evaluator",
        "package_sha256": release_record["attributes_package"]["content_checksums_sha256"],
        "items": release_record["attributes_package"]["scene_count"],
    },
])
publication.to_csv(SAFE_ROOT / "resource_publication.csv", index=False)
safe_record = {
    "schema_version": 1,
    "stage": "ds2_confirmation_input_release",
    "status": "completed",
    "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
    "membership_sha256": DS2_CONFIRM_CONTRACT.membership_sha256,
    "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
    "image_count": DS2_CONFIRM_CONTRACT.image_count,
    "asset_group_count": DS2_CONFIRM_CONTRACT.asset_group_count,
    "all_completion_gates_passed": bool(gates["passed"].all()),
}
(SAFE_ROOT / "run_record.json").write_text(json.dumps(safe_record, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
save_bar_chart(summary["resource"].tolist(), summary["items"].tolist(), SAFE_ROOT / "resource_counts.png", title="DS2 confirmation resources", y_label="Items", show_values=True)
_artifact_manifest(SAFE_ROOT)
display(gates)
display(summary)
display(publication)
display(NotebookImage(filename=str(SAFE_ROOT / "resource_counts.png")))
if not gates["passed"].all():
    raise RuntimeError("DS2 release acceptance failed")
"""
    ),
    markdown(
        """
## ClearML finalisation

Tracking starts only after the packages, tables, chart, run record, and checksums are complete. The isolated process receives only the safe aggregate folder. A tracking timeout cannot stop or invalidate the completed release.
"""
    ),
    code(
        """
if EXECUTION_ENVIRONMENT == "kaggle":
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt")], check=True, timeout=180)
    except Exception as error:
        print(f"ClearML dependency setup was unavailable: {type(error).__name__}")
tracking_receipt = run_clearml_finalizer(
    SAFE_ROOT,
    "ds2_confirmation_input_release",
    "ds2-confirmation-release",
    ["notebook-08a", "ds2", "confirmation", "kaggle"],
    OUTPUT_ROOT / "clearml_receipt.json",
)
display(tracking_receipt)
"""
    ),
    markdown(
        """
## Build compact verified downloads

The expanded RGB, geometry, and evaluator resources stay outside the notebook output. They are published as separate private Kaggle datasets. This notebook keeps aggregate release evidence, the sanitized ClearML receipt, and checksums in `/kaggle/working`.
"""
    ),
    code(
        """
BUNDLE_ROOT = (
    Path("/kaggle/working/notebook08a_bundles")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else OUTPUT_ROOT.parent / "notebook08a_bundles"
)
verification_path = BUNDLE_ROOT / "ds2-confirmation-release-verification.json"
if not verification_path.is_file():
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/package_kaggle_stage_output.py"),
            "--stage-root", str(OUTPUT_ROOT),
            "--output-root", str(BUNDLE_ROOT),
            "--stage-name", "ds2-confirmation-release",
            "--review-path", "clearml_receipt.json",
        ],
        check=True,
        env=code_environment(),
    )
bundle_record = json.loads(verification_path.read_text(encoding="utf-8"))
display(pd.DataFrame(bundle_record["archives"]))
print(f"Verified downloads: {BUNDLE_ROOT}")
"""
    ),
    markdown(
        """
## Output boundary

Publish `model-input` and `oracle-geometry` as separate private Kaggle resources for Notebook 08B. Attach `evaluator-attributes` only to Notebook 08C. Do not attach the protected evaluator resource to a model job.
"""
    ),
]


inference_cells = [
    markdown(
        """
# ToysAD-8K DS2 Fixed Pipeline Inference

**Notebook file:** `08b-toysad8k-ds2-frozen-pipeline.ipynb`

This notebook runs the fixed model-facing confirmation path on 300 untouched development scenes and all 6,000 views. SAM 2.1 receives the supplied reference boxes, then DINOv2-small extracts the frozen object representation and calculates B0 and P1.

The same images also follow the proposed automatic route. The DS1-selected LocateAnything model, prompt, and post-processing rule generate boxes, SAM 2.1 converts those boxes into masks, and DINOv2-small calculates B0 and P1. Supplied masks remain the scientific upper bound. Reference boxes represent a user-confirmed assisted route.

Before evaluator labels are opened, this run applies a predeclared bounded DS2 operational policy. LocateAnything is limited to 4,096 new tokens, two attempts, and 75 seconds per attempt. A terminal generation failure is recorded and carried forward as zero proposals, so it becomes an end-to-end miss instead of stopping the full experiment. This changes the runtime policy, not the selected model, prompt, or box post-processing rule.

On Kaggle, this notebook requires the real pinned models. A local contract mode is available only to test paths, schemas, restart behaviour, tables, and figures. Contract values are not model evidence and must not be used in the progress report.
"""
    ),
    markdown(
        """
## Questions answered by this run

1. Does the fixed LocateAnything choice produce usable proposals across all 6,000 images under the bounded runtime policy?
2. Does SAM 2.1 complete the reference-box and automatic-box routes with stable masks?
3. Does the frozen DINOv2-small B0 and P1 path complete all three routes without opening labels?
4. How many proposal generations or masks are unusable, and are the real proposal panels, mask overlays, and coarse P1 heatmaps visually reasonable?

Accuracy, ranking quality, view fusion, and the DS2 confirmation status are measured only in Notebook 08C.
"""
    ),
    code(
        """
from __future__ import annotations

import importlib.metadata
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True

from IPython.display import Image as NotebookImage, display

NOTEBOOK_NAME = "08b-toysad8k-ds2-frozen-pipeline.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
RUN_MODE = os.getenv(
    "VPI_RUN_MODE",
    "model_inference" if EXECUTION_ENVIRONMENT == "kaggle" else "contract",
)
if RUN_MODE not in {"contract", "model_inference"}:
    raise ValueError("VPI_RUN_MODE must be contract or model_inference")
if EXECUTION_ENVIRONMENT == "kaggle" and RUN_MODE != "model_inference":
    raise RuntimeError("Kaggle evidence must use the real pinned models")
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/temp/toysad8k_ds2_frozen_pipeline" if EXECUTION_ENVIRONMENT == "kaggle" else ".local/runs/toysad8k-ds2-frozen-pipeline",
)).expanduser().resolve()
RECOVERY_OUTPUT_ROOT = (
    Path("/kaggle/working/notebook08b_recovery").resolve()
    if EXECUTION_ENVIRONMENT == "kaggle"
    else OUTPUT_ROOT.parent / "notebook08b_recovery"
)
"""
    ),
    code(COMMON_LOCATORS),
    markdown("## Locate and verify the model-safe inputs"),
    code(
        """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
RGB_ROOT = find_package("toysad8k-model-input", "VPI_RGB_ROOT")
GEOMETRY_ROOT = find_package("ds2-oracle-geometry", "VPI_GEOMETRY_ROOT")


def locate_selected_prompt() -> Path:
    configured = os.getenv("VPI_SELECTED_PROMPT")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
    # Kaggle mounts the completed Notebook 04.3 archive as a notebook input.
    archive_path = find_named_input_file(
        "ds1-locateanything-prompt-freeze-downstream-input.zip",
        "VPI_SELECTED_PROMPT_ARCHIVE",
    )
    verification_path = find_named_input_file(
        "ds1-locateanything-prompt-freeze-verification.json",
        "VPI_SELECTED_PROMPT_VERIFICATION",
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    matches = [row for row in verification["archives"] if row["path"] == archive_path.name]
    actual_sha256 = sha256(archive_path.read_bytes()).hexdigest()
    if len(matches) != 1 or matches[0]["sha256"] != actual_sha256:
        raise ValueError("The DS1 prompt archive differs from its producer record")
    checksums_path = verification_path.parent / "SHA256SUMS"
    if not checksums_path.is_file():
        raise FileNotFoundError("The DS1 prompt checksum manifest is missing")
    checksums = {
        line.split(maxsplit=1)[1].strip(): line.split(maxsplit=1)[0]
        for line in checksums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if checksums.get(archive_path.name) != actual_sha256:
        raise ValueError("The DS1 prompt archive differs from SHA256SUMS")
    destination = Path("/kaggle/temp/ds1_prompt_freeze_input").resolve()
    marker = destination / "downstream/selected_prompt.json"
    if not marker.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("The DS1 prompt archive failed its integrity check")
            for name in archive.namelist():
                member = Path(name)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"Unsafe prompt archive member: {name}")
            archive.extractall(destination)
    if not marker.is_file():
        raise FileNotFoundError("The DS1 prompt archive does not contain selected_prompt.json")
    return marker


SELECTED_PROMPT = locate_selected_prompt()


def locate_recovery_input() -> Path | None:
    configured = os.getenv("VPI_RECOVERY_INPUT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (
            (root / "recovery_snapshot.zip").is_file()
            or (root / "recovery_record.json").is_file()
        ):
            raise FileNotFoundError("Configured DS2 recovery input is invalid")
        return root
    if EXECUTION_ENVIRONMENT != "kaggle":
        return None
    input_root = Path("/kaggle/input")
    snapshot_roots = {
        path.parent.resolve()
        for path in input_root.glob("**/recovery_snapshot.zip")
        if path.is_file()
    }
    legacy_roots = {
        path.parent.resolve()
        for path in input_root.glob("**/recovery_record.json")
        if path.is_file()
    }
    candidates = sorted(snapshot_roots or legacy_roots, key=lambda path: path.as_posix())
    if len(candidates) > 1:
        raise ValueError("Multiple DS2 recovery inputs were found")
    return candidates[0] if candidates else None


RECOVERY_INPUT_ROOT = locate_recovery_input()

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.model_safe_package import verify_model_input_package
from visual_peer_inspector.evaluation.confirmation_reference import verify_ds2_geometry_package
from visual_peer_inspector.workflows.locateanything_confirmation_ds2 import (
    DS2_ATTEMPT_TIMEOUT_SECONDS,
    DS2_MAXIMUM_ATTEMPTS_PER_JOB,
    DS2_MAX_NEW_TOKENS,
    DS2_TERMINAL_FAILURE_DISPOSITION,
)

rgb_audit = verify_model_input_package(RGB_ROOT, contract=DS2_CONFIRM_CONTRACT)
geometry_record = verify_ds2_geometry_package(GEOMETRY_ROOT)
selected_prompt_record = json.loads(SELECTED_PROMPT.read_text(encoding="utf-8"))
if (
    selected_prompt_record.get("stage") != "ds1_locateanything_prompt_freeze"
    or selected_prompt_record.get("development_only") is not True
    or selected_prompt_record.get("requires_unchanged_ds2_confirmation") is not True
):
    raise ValueError("The mounted LocateAnything decision is not a frozen DS1 selection")
display({
    "cohort": DS2_CONFIRM_CONTRACT.cohort_id,
    "scenes": rgb_audit.scene_count,
    "images": rgb_audit.image_count,
    "visible object prompts": geometry_record["visible_object_view_count"],
    "LocateAnything prompt": selected_prompt_record["selected"]["prompt"],
    "proposal post-processing": selected_prompt_record["selected"]["postprocessing_policy"],
    "maximum new tokens": DS2_MAX_NEW_TOKENS,
    "maximum attempts per image": DS2_MAXIMUM_ATTEMPTS_PER_JOB,
    "attempt timeout seconds": DS2_ATTEMPT_TIMEOUT_SECONDS,
    "terminal failure handling": DS2_TERMINAL_FAILURE_DISPOSITION,
    "recovery input": RECOVERY_INPUT_ROOT is not None,
    "labels mounted": False,
})
"""
    ),
    markdown("## Prepare the pinned GPU runtime"),
    code(
        """
checkpoint_path = None
runtime_evidence = {
    "execution environment": EXECUTION_ENVIRONMENT,
    "run mode": RUN_MODE,
    "LocateAnything model": "nvidia/LocateAnything-3B",
    "LocateAnything revision": "c32291ca5e996f5a7a485845b4f57a233936bba0",
    "DINO model": "facebook/dinov2-small",
    "DINO revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
}
if RUN_MODE == "model_inference":
    install_environment = code_environment()
    install_environment["SAM2_BUILD_CUDA"] = "0"

    # Install only the add-ons that are absent from the Kaggle image.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-lock.txt")], check=True, timeout=900, env=install_environment)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
    ], check=True, timeout=600, env=install_environment)

    import torch
    from huggingface_hub import hf_hub_download
    from visual_peer_inspector.data.model_safe_package import file_sha256
    from visual_peer_inspector.models.sam2 import Sam2Configuration

    if not torch.cuda.is_available():
        raise RuntimeError("Notebook 08B requires a Kaggle GPU")
    configuration = Sam2Configuration()
    checkpoint_path = Path(hf_hub_download(
        repo_id=configuration.model_repository,
        filename=configuration.checkpoint_filename,
        revision=configuration.model_revision,
        cache_dir="/tmp/vpi-sam21-model-cache",
    ))
    if checkpoint_path.stat().st_size != configuration.checkpoint_size_bytes or file_sha256(checkpoint_path) != configuration.checkpoint_sha256:
        raise RuntimeError("The pinned SAM 2.1 checkpoint identity differs")
    runtime_evidence.update({
        "GPU": torch.cuda.get_device_name(0),
        "GPU count": torch.cuda.device_count(),
        "SAM checkpoint SHA-256": configuration.checkpoint_sha256,
    })
else:
    runtime_evidence["evidence boundary"] = "Local contract validation only"
display(runtime_evidence)
"""
    ),
    markdown(
        """
## Run the fixed model-facing pipeline

The models run sequentially on one GPU. LocateAnything is released before SAM 2.1 loads, and SAM is released before DINOv2-small loads. The one-image and 20-image stages must pass operational and integrity checks before the 6,000-image run starts. Agreement values are reported, not used as a quality cutoff. The pipeline prints progress after the first image and at fixed intervals. Results are written atomically so an identical rerun can resume from verified checkpoints. Invalid proposal generations and masks remain explicit evidence and do not disappear from the downstream coverage.

ClearML is not active during model loading or inference.
"""
    ),
    code(
        """
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_ds2_inference.py"),
    "--rgb-root", str(RGB_ROOT),
    "--geometry-root", str(GEOMETRY_ROOT),
    "--selected-prompt", str(SELECTED_PROMPT),
    "--output-root", str(OUTPUT_ROOT),
    "--run-mode", RUN_MODE,
    "--device", "cuda" if RUN_MODE == "model_inference" else "cpu",
    "--batch-size", "32",
    "--recovery-output-root", str(RECOVERY_OUTPUT_ROOT),
]
if RECOVERY_INPUT_ROOT is not None:
    command.extend(("--recovery-input-root", str(RECOVERY_INPUT_ROOT)))
if checkpoint_path is not None:
    command.extend(("--sam-checkpoint", str(checkpoint_path)))
subprocess.run(command, check=True, env=code_environment())

proposal_record = json.loads((OUTPUT_ROOT / "locateanything/safe/run_record.json").read_text(encoding="utf-8"))
reference_sam_record = json.loads((OUTPUT_ROOT / "sam21/reference-box/safe/run_record.json").read_text(encoding="utf-8"))
automatic_sam_record = json.loads((OUTPUT_ROOT / "sam21/automatic-box/safe/run_record.json").read_text(encoding="utf-8"))
representation_record = json.loads((OUTPUT_ROOT / "representation/safe/run_record.json").read_text(encoding="utf-8"))
display({
    "run mode": RUN_MODE,
    "LocateAnything status": proposal_record["status"],
    "proposal images": proposal_record["image_count"],
    "proposal generation failures": proposal_record["terminal_failed_image_count"],
    "discarded untrusted boxes": proposal_record["discarded_untrusted_box_count"],
    "zero-proposal images": proposal_record["zero_proposal_image_count"],
    "LocateAnything policy SHA-256": proposal_record["effective_operational_policy_sha256"],
    "reference-box SAM status": reference_sam_record["status"],
    "automatic-box SAM status": automatic_sam_record["status"],
    "SAM images per route": reference_sam_record["image_count"],
    "DINO status": representation_record["status"],
    "score rows": representation_record["score_row_count"],
    "unscoreable score rows": representation_record["unscoreable_row_count"],
    "invalid-mask score rows": representation_record["invalid_mask_row_count"],
    "evaluator opened": representation_record["evaluator_opened"],
})
"""
    ),
    markdown("## Operational tables and report-ready charts"),
    code(
        """
import pandas as pd

proposal_safe = OUTPUT_ROOT / "locateanything/safe"
reference_sam_safe = OUTPUT_ROOT / "sam21/reference-box/safe"
automatic_sam_safe = OUTPUT_ROOT / "sam21/automatic-box/safe"
representation_safe = OUTPUT_ROOT / "representation/safe"
display(pd.read_csv(proposal_safe / "completion_checks.csv"))
display(pd.DataFrame([json.loads((proposal_safe / "operational_summary.json").read_text(encoding="utf-8"))]))
display(pd.DataFrame([json.loads((proposal_safe / "repeat_agreement.json").read_text(encoding="utf-8"))]))
display(NotebookImage(filename=str(proposal_safe / "completion_summary.png")))
display(NotebookImage(filename=str(proposal_safe / "proposal_count_distribution.png")))
display(NotebookImage(filename=str(proposal_safe / "runtime_distribution.png")))
for route_name, route_root in (
    ("reference-box", reference_sam_safe),
    ("automatic-box", automatic_sam_safe),
):
    print(f"SAM route: {route_name}")
    display(pd.read_csv(route_root / "completion_checks.csv"))
    display(pd.read_csv(route_root / "route_summary.csv"))
    display(pd.DataFrame([json.loads((route_root / "repeat_agreement.json").read_text(encoding="utf-8"))]))
    display(NotebookImage(filename=str(route_root / "completion_summary.png")))
    display(NotebookImage(filename=str(route_root / "runtime_distribution.png")))
    display(NotebookImage(filename=str(route_root / "prompt_count_distribution.png")))
display(pd.read_csv(representation_safe / "completion_gates.csv"))
display(pd.read_csv(representation_safe / "runtime_summary.csv"))
display(pd.read_csv(representation_safe / "p1_route_summary.csv"))
display(pd.DataFrame([json.loads((representation_safe / "resolved_configuration.json").read_text(encoding="utf-8"))]))
display(NotebookImage(filename=str(representation_safe / "mean_p1_by_route.png")))
display(NotebookImage(filename=str(representation_safe / "p1_score_distribution.png")))
print("Every proposal remains in the score manifest. A mask with too little foreground is marked invalid_mask with no invented DINO score or rank.")
print("B0 and P1 accuracy is reported only after Notebook 08C opens the protected evaluator.")
"""
    ),
    markdown(
        """
## Private qualitative checks

The proposal panel shows the fixed LocateAnything boxes on 20 varied confirmation images. The next two panels show SAM mask boundaries for the assisted and automatic routes. The final panel shows high P1 mismatch crops. Red heatmap regions indicate stronger feature mismatch and are not exact defect boundaries. These panels remain in private Kaggle output for review.
"""
    ),
    code(
        """
display(NotebookImage(filename=str(OUTPUT_ROOT / "locateanything/private/figures/proposal_examples.png")))
display(NotebookImage(filename=str(OUTPUT_ROOT / "sam21/reference-box/private/figures/reference_box_mask_examples.png")))
display(NotebookImage(filename=str(OUTPUT_ROOT / "sam21/automatic-box/private/figures/automatic_box_mask_examples.png")))
display(NotebookImage(filename=str(OUTPUT_ROOT / "representation/private/figures/p1_heatmap_examples.png")))
"""
    ),
    markdown("## Bounded ClearML finalisation"),
    code(
        """
# Tracking runs after the model artifacts and manifests are complete.
if EXECUTION_ENVIRONMENT == "kaggle":
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt")], check=True, timeout=180)
    except Exception as error:
        print(f"ClearML dependency setup was unavailable: {type(error).__name__}")
proposal_tracking = run_clearml_finalizer(
    proposal_safe,
    "ds2_frozen_locateanything",
    "ds2-frozen-locateanything",
    ["notebook-08b", "ds2", "locateanything", "kaggle"],
    OUTPUT_ROOT / "locateanything/clearml_receipt.json",
)
reference_sam_tracking = run_clearml_finalizer(
    reference_sam_safe,
    "ds2_reference_box_sam21",
    "ds2-reference-box-sam21",
    ["notebook-08b", "ds2", "sam2.1", "kaggle"],
    OUTPUT_ROOT / "sam21/reference-box/clearml_receipt.json",
)
automatic_sam_tracking = run_clearml_finalizer(
    automatic_sam_safe,
    "ds2_automatic_box_sam21",
    "ds2-automatic-box-sam21",
    ["notebook-08b", "ds2", "sam2.1", "automatic-box", "kaggle"],
    OUTPUT_ROOT / "sam21/automatic-box/clearml_receipt.json",
)
representation_tracking = run_clearml_finalizer(
    representation_safe,
    "ds2_frozen_representation_and_peer_scores",
    "ds2-dinov2-small-p1",
    ["notebook-08b", "ds2", "dinov2-small", "p1", "kaggle"],
    OUTPUT_ROOT / "representation/clearml_receipt.json",
)
display({
    "LocateAnything tracking": proposal_tracking,
    "reference-box SAM tracking": reference_sam_tracking,
    "automatic-box SAM tracking": automatic_sam_tracking,
    "representation tracking": representation_tracking,
})
"""
    ),
    markdown(
        """
## Build compact verified downloads

Expanded checkpoints and feature caches stay in temporary Kaggle storage. The following archives retain aggregate GitHub evidence, selected private review panels, and the three files required by the protected evaluator. Each archive is checksum verified before the notebook finishes.
"""
    ),
    code(
        """
BUNDLE_ROOT = (
    Path("/kaggle/working/notebook08b_bundles")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else OUTPUT_ROOT.parent / "notebook08b_bundles"
)
bundle_specs = [
    (OUTPUT_ROOT / "locateanything", "ds2-locateanything", ["private/figures", "clearml_receipt.json"], []),
    (OUTPUT_ROOT / "sam21/reference-box", "ds2-reference-sam21", ["private/figures", "clearml_receipt.json"], ["private/sam_results.jsonl"]),
    (OUTPUT_ROOT / "sam21/automatic-box", "ds2-automatic-sam21", ["private/figures", "clearml_receipt.json"], ["private/sam_results.jsonl"]),
    (OUTPUT_ROOT / "representation", "ds2-dinov2-p1", ["private/figures", "clearml_receipt.json"], ["model-input"]),
]
bundle_rows = []
for stage_root, stage_name, review_paths, downstream_paths in bundle_specs:
    stage_bundle_root = BUNDLE_ROOT / stage_name
    verification_path = stage_bundle_root / f"{stage_name}-verification.json"
    if not verification_path.is_file():
        # Package review-only panels and downstream evaluator inputs separately.
        command = [
            sys.executable,
            str(CODE_ROOT / "scripts/package_kaggle_stage_output.py"),
            "--stage-root", str(stage_root),
            "--output-root", str(stage_bundle_root),
            "--stage-name", stage_name,
        ]
        for value in review_paths:
            command.extend(("--review-path", value))
        for value in downstream_paths:
            command.extend(("--downstream-path", value))
        subprocess.run(command, check=True, env=code_environment())
    record = json.loads(verification_path.read_text(encoding="utf-8"))
    for archive in record["archives"]:
        bundle_rows.append({"stage": stage_name, **archive})
display(pd.DataFrame(bundle_rows))
print(f"Verified downloads: {BUNDLE_ROOT}")
if RECOVERY_OUTPUT_ROOT.exists():
    shutil.rmtree(RECOVERY_OUTPUT_ROOT)
print("Temporary recovery checkpoints removed after verified bundle creation")
"""
    ),
    markdown(
        """
## Handoff to the evaluator

Notebook 08C consumes the reference SAM, automatic SAM, and DINO downstream archives from this completed version. The automatic SAM archive preserves the per-image proposal status and discarded-box count, so terminal model errors remain distinct from genuine zero-proposal outcomes. The evaluator must verify package hashes before opening protected attributes. It must not change a model, crop, prompt, post-processing rule, or score setting.
"""
    ),
]


evaluation_cells = [
    markdown(
        """
# ToysAD-8K DS2 Confirmation and Multi-view Evaluation

**Notebook file:** `08c-toysad8k-ds2-confirmation-and-multiview.ipynb`

This evaluator-only notebook opens the protected DS2 labels after the LocateAnything, SAM 2.1, and DINOv2-small prediction files are complete and hash locked. It evaluates the fixed DS2 operational route v2, including proposal generation failures and masks that are too small for DINO scoring. It measures proposal and mask quality, compares B0 with P1 across supplied, assisted, and automatic routes, and reports fixed camera-spread view subsets and fusion rules.

It does not load or run a model. It does not tune from calibration or internal-lock data.
"""
    ),
    markdown(
        """
## Predeclared evaluation

- View counts: 1, 2, 3, 5, 10, and 20.
- Application condition: three evenly spaced views.
- Fusion reports: mean normalized rank and visibility-weighted normalized rank.
- One-anomaly metrics: Top-1, MRR, and scene AP.
- Two-anomaly metrics: AP, Recall@2, and exact-set recovery.
- Uncertainty: paired confidence intervals clustered by base asset.
- Failure analysis: category, anomaly family, object count, anomaly count, rank disagreement, and low-margin scenes.

DS2 is a confirmation cohort. Both fusion reports are retained, but the predeclared mean-rank rule remains the application setting. No accuracy cutoff or DS2-based model tuning is applied.
"""
    ),
    code(
        """
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True

from IPython.display import Image as NotebookImage, display

NOTEBOOK_NAME = "08c-toysad8k-ds2-confirmation-and-multiview.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/temp/toysad8k_ds2_confirmation_evaluation" if EXECUTION_ENVIRONMENT == "kaggle" else ".local/runs/toysad8k-ds2-confirmation-evaluation",
)).expanduser().resolve()
"""
    ),
    code(COMMON_LOCATORS),
    markdown("## Locate the frozen predictions and protected evaluator inputs"),
    code(
        """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
RGB_ROOT = find_package("toysad8k-model-input", "VPI_RGB_ROOT")
GEOMETRY_ROOT = find_package("ds2-oracle-geometry", "VPI_GEOMETRY_ROOT")
ATTRIBUTES_ROOT = find_package("ds2-evaluator-attributes", "VPI_ATTRIBUTES_ROOT")


def verified_extract(archive_path: Path, destination: Path) -> None:
    # Validate archive integrity and member paths before evaluator inputs are opened.
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Archive integrity failed: {archive_path.name}")
        for name in archive.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe archive member: {name}")
        archive.extractall(destination)


def verified_downstream_archive(archive_name: str) -> Path:
    stage_name = archive_name.removesuffix("-downstream-input.zip")
    archive_path = find_named_input_file(archive_name, "VPI_UNUSED_ARCHIVE")
    verification_path = find_named_input_file(
        f"{stage_name}-verification.json",
        "VPI_UNUSED_VERIFICATION",
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    matches = [row for row in verification["archives"] if row["path"] == archive_name]
    if len(matches) != 1:
        raise ValueError(f"No unique verification record for {archive_name}")
    actual_sha256 = sha256(archive_path.read_bytes()).hexdigest()
    if actual_sha256 != matches[0]["sha256"]:
        raise ValueError(f"Archive SHA-256 differs from the producer record: {archive_name}")
    checksums_path = verification_path.parent / "SHA256SUMS"
    if not checksums_path.is_file():
        raise FileNotFoundError(f"Checksum manifest is missing for {archive_name}")
    checksums = {
        line.split(maxsplit=1)[1].strip(): line.split(maxsplit=1)[0]
        for line in checksums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if checksums.get(archive_name) != actual_sha256:
        raise ValueError(f"Archive SHA-256 differs from SHA256SUMS: {archive_name}")
    return archive_path


configured_inference = os.getenv("VPI_INFERENCE_ROOT")
if configured_inference:
    INFERENCE_ROOT = Path(configured_inference).expanduser().resolve()
else:
    INFERENCE_ROOT = Path("/kaggle/temp/ds2_evaluator_inference_inputs").resolve()
    marker = INFERENCE_ROOT / "representation/model-input/package_record.json"
    if not marker.is_file():
        archive_specs = [
            ("ds2-reference-sam21-downstream-input.zip", INFERENCE_ROOT / "sam21/reference-box"),
            ("ds2-automatic-sam21-downstream-input.zip", INFERENCE_ROOT / "sam21/automatic-box"),
            ("ds2-dinov2-p1-downstream-input.zip", INFERENCE_ROOT / "representation"),
        ]
        for archive_name, destination in archive_specs:
            destination.mkdir(parents=True, exist_ok=True)
            verified_extract(verified_downstream_archive(archive_name), destination)

SCORE_ROOT = INFERENCE_ROOT / "representation/model-input"
REFERENCE_SAM_RESULTS = INFERENCE_ROOT / "sam21/reference-box/private/sam_results.jsonl"
AUTOMATIC_SAM_RESULTS = INFERENCE_ROOT / "sam21/automatic-box/private/sam_results.jsonl"
for required in (SCORE_ROOT / "package_record.json", REFERENCE_SAM_RESULTS, AUTOMATIC_SAM_RESULTS):
    if not required.is_file():
        raise FileNotFoundError(f"Frozen DS2 evaluator input is missing: {required.name}")

score_record = json.loads((SCORE_ROOT / "package_record.json").read_text(encoding="utf-8"))
display({
    "score package status": score_record["all_checks_passed"],
    "score rows": score_record["score_row_count"],
    "model": score_record["model_key"],
    "methods": score_record["methods"],
    "evaluator opened during inference": score_record["evaluator_opened"],
    "reference SAM results": REFERENCE_SAM_RESULTS.name,
    "automatic SAM results": AUTOMATIC_SAM_RESULTS.name,
})
"""
    ),
    markdown("## Run the protected confirmation and multi-view analysis"),
    code(
        """
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_ds2_confirmation_evaluation.py"),
    "--rgb-root", str(RGB_ROOT),
    "--geometry-root", str(GEOMETRY_ROOT),
    "--attributes-root", str(ATTRIBUTES_ROOT),
    "--inference-root", str(INFERENCE_ROOT),
    "--output-root", str(OUTPUT_ROOT),
]
subprocess.run(command, check=True, env=code_environment())
run_record = json.loads((OUTPUT_ROOT / "safe/run_record.json").read_text(encoding="utf-8"))
display({
    "status": run_record["status"],
    "scenes": run_record["scene_count"],
    "metric rows": run_record["scene_metric_row_count"],
    "unscoreable candidate-view rows": run_record["unscoreable_candidate_view_count"],
    "invalid-mask candidate-view rows": run_record["invalid_mask_candidate_view_count"],
    "prediction lock verified": run_record["evaluator_opened_after_prediction_lock"],
})
"""
    ),
    markdown("## Component confirmation"),
    code(
        """
import pandas as pd

SAFE_ROOT = OUTPUT_ROOT / "safe"
display(pd.read_csv(SAFE_ROOT / "proposal_summary.csv"))
display(pd.read_csv(SAFE_ROOT / "mask_summary.csv"))
display(pd.read_csv(SAFE_ROOT / "unscoreable_summary.csv"))
display(pd.read_csv(SAFE_ROOT / "decision_gates.csv"))
display(pd.read_csv(SAFE_ROOT / "paired_intervals.csv"))
display(NotebookImage(filename=str(SAFE_ROOT / "locateanything_proposal_quality.png")))
display(NotebookImage(filename=str(SAFE_ROOT / "sam_mask_quality.png")))
display(NotebookImage(filename=str(SAFE_ROOT / "sam_mask_iou_distribution.png")))
display(NotebookImage(filename=str(SAFE_ROOT / "route_average_precision.png")))
"""
    ),
    markdown("## View-count and fusion results"),
    code(
        """
multiview = pd.read_csv(SAFE_ROOT / "multiview_summary.csv")
automatic_curve = multiview.query("route == 'automatic_box' and method == 'P1'")
display(automatic_curve.sort_values(["view_count", "fusion_method"]))
display(multiview.query("view_count == 20 and fusion_method == 'mean_rank'").sort_values(["route", "method"]))
display(NotebookImage(filename=str(SAFE_ROOT / "multiview_ablation.png")))
display(pd.read_csv(SAFE_ROOT / "primary_three_view_summary.csv"))
display(NotebookImage(filename=str(SAFE_ROOT / "primary_three_view_metrics.png")))
display(NotebookImage(filename=str(SAFE_ROOT / "rank_disagreement_distribution.png")))
print("Fusion uses only scored ranks and applies scored-view support once. Detected support remains a separate coverage measure.")
"""
    ),
    markdown("## Subgroups and real failure examples"),
    code(
        """
subgroups = pd.read_csv(SAFE_ROOT / "subgroup_summary.csv")
display(subgroups.sort_values(["dimension", "average_precision"]).groupby("dimension").head(8))
display(pd.read_csv(SAFE_ROOT / "scene_type_summary.csv"))
display(NotebookImage(filename=str(SAFE_ROOT / "scene_type_metrics.png")))
display(NotebookImage(filename=str(SAFE_ROOT / "lowest_supported_subgroups.png")))
print("The private panel labels each example by route and score. Box and mask outlines show the evidence used for each prediction, and missed references are marked explicitly.")
display(NotebookImage(filename=str(OUTPUT_ROOT / "private/figures/three_view_success_and_failure.png")))
"""
    ),
    markdown("## Frozen configuration and confirmation status"),
    code(
        """
decision = json.loads((SAFE_ROOT / "model_and_route_decision.json").read_text(encoding="utf-8"))
display(pd.DataFrame([decision]))
print(f"Encoder: {decision['encoder']}")
print(f"Scorer: {decision['scorer']}")
print(f"Application mask route: {decision['application_mask_route']}")
print(f"Primary views: {decision['primary_view_ids']}")
print(f"Fusion: {decision['fusion_method']}")
print(f"Decision status: {decision['status']}")
"""
    ),
    markdown(
        """
## Bounded ClearML finalisation

Only aggregate tables, figures, decisions, and hashes are uploaded. Per-scene scores, object IDs, labels, masks, RGB images, and failure panels remain in private Kaggle output.
"""
    ),
    code(
        """
if EXECUTION_ENVIRONMENT == "kaggle":
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt")], check=True, timeout=180)
    except Exception as error:
        print(f"ClearML dependency setup was unavailable: {type(error).__name__}")
tracking_receipt = run_clearml_finalizer(
    SAFE_ROOT,
    "ds2_confirmation_and_multiview",
    "ds2-confirmation-and-multiview",
    ["notebook-08c", "ds2", "multiview", "fusion", "kaggle"],
    OUTPUT_ROOT / "clearml_receipt.json",
)
display(tracking_receipt)
"""
    ),
    markdown(
        """
## Build compact verified downloads

The repository archive contains aggregate tables, charts, decisions, and hashes. The review archive contains only the selected private success and failure panel. Per-scene scores, masks, RGB images, credentials, and extracted evaluator inputs are not included in the GitHub candidate.
"""
    ),
    code(
        """
BUNDLE_ROOT = (
    Path("/kaggle/working/notebook08c_bundles")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else OUTPUT_ROOT.parent / "notebook08c_bundles"
)
verification_path = BUNDLE_ROOT / "ds2-confirmation-and-multiview-verification.json"
if not verification_path.is_file():
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/package_kaggle_stage_output.py"),
            "--stage-root", str(OUTPUT_ROOT),
            "--output-root", str(BUNDLE_ROOT),
            "--stage-name", "ds2-confirmation-and-multiview",
            "--review-path", "private/figures",
            "--review-path", "clearml_receipt.json",
        ],
        check=True,
        env=code_environment(),
    )
bundle_record = json.loads(verification_path.read_text(encoding="utf-8"))
display(pd.DataFrame(bundle_record["archives"]))
print(f"Verified downloads: {BUNDLE_ROOT}")
"""
    ),
    markdown(
        """
## Next stage

Record the measured automatic and assisted routes separately. DS2 measures the predeclared SAM 2.1 Hiera-large, DINOv2-small, B0, P1, three-view, and mean-rank configuration without tuning it. Keep that configuration unchanged for the next development scale run. The automatic route remains experimental until a label-free cross-view association method replaces the evaluator-only box matching used in this notebook.

The next two weeks should scale the frozen method to the 1,000-scene development role, implement label-free cross-view association and the assisted review flow, then use the 600-scene calibration role only to set a review threshold. Internal-lock and official evaluation remain unopened until those choices are frozen.
"""
    ),
]


notebooks = {
    "08a-toysad8k-ds2-confirmation-release.ipynb": notebook(release_cells),
    "08b-toysad8k-ds2-frozen-pipeline.ipynb": notebook(inference_cells),
    "08c-toysad8k-ds2-confirmation-and-multiview.ipynb": notebook(evaluation_cells),
}
NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
for name, value in notebooks.items():
    path = NOTEBOOK_ROOT / name
    nbformat.write(value, path)
    print(path)
