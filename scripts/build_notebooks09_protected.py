#!/usr/bin/env python3
"""Build the clean protected inference notebooks for calibration and internal lock."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "protected-evaluation"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


CONFIGURATIONS = (
    {
        "code": "09a",
        "title": "ToysAD-8K Protected Calibration Inference",
        "filename": "09a-toysad8k-protected-calibration-inference.ipynb",
        "cohort": "calibration",
        "cohort_id": "toysad8k-calibration-600-v1",
        "scene_count": 600,
        "image_count": 12_000,
        "stage": "protected_calibration_inference",
        "output_name": "toysad8k_protected_calibration_inference",
        "project": "Visual Peer Inspector/06 Protected Evaluation",
        "policy_required": False,
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-calibration600-model-input",
            "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
            "esreekarreddy/vpi-toysad-development-method-freeze",
        ],
        "kernel_sources": [],
    },
    {
        "code": "09c",
        "title": "ToysAD-8K Protected Internal-Lock Inference",
        "filename": "09c-toysad8k-protected-internal-lock-inference.ipynb",
        "cohort": "internal-lock",
        "cohort_id": "toysad8k-internal-lock-400-v1",
        "scene_count": 400,
        "image_count": 8_000,
        "stage": "protected_internal_lock_inference",
        "output_name": "toysad8k_protected_internal_lock_inference",
        "project": "Visual Peer Inspector/06 Protected Evaluation",
        "policy_required": True,
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-internal-lock400-model-input",
            "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
            "esreekarreddy/vpi-toysad-development-method-freeze",
            "esreekarreddy/vpi-toysad-calibration-review-policy",
        ],
        "kernel_sources": [],
    },
)


HELPERS = r'''
def kaggle_roots() -> list[Path]:
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return []
    roots = [path for path in input_root.iterdir() if path.is_dir()]
    return sorted(roots, key=lambda path: path.as_posix())


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents, *kaggle_roots()))
    for candidate in candidates:
        if (candidate / "scripts/run_protected_cohort_inference.py").is_file():
            return candidate.resolve()
        if (candidate / "code/scripts/run_protected_cohort_inference.py").is_file():
            return (candidate / "code").resolve()
    raise FileNotFoundError("Could not locate the Visual Peer Inspector code snapshot")


def find_model_input(cohort_id: str) -> Path:
    configured = os.getenv("VPI_MODEL_INPUT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    for root in kaggle_roots():
        candidates.extend((root, *(path for path in root.iterdir() if path.is_dir())))
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
    raise FileNotFoundError(f"Could not locate model-facing package {cohort_id}")


def find_named_file(name: str, environment_name: str) -> Path:
    configured = os.getenv(environment_name)
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


def find_recovery_pair(code: str) -> tuple[Path, Path] | None:
    configured = os.getenv("VPI_RESUME_ROOT")
    sources = [Path(configured).expanduser()] if configured else kaggle_roots()
    stem = "calibration-inference" if code == "09a" else "internal-lock-inference"
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
        raise ValueError("VPI_RESUME_ROOT must contain exactly one protected recovery pair")
    return bundles[0], manifests[0]


def code_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(CODE_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not existing else source_root + os.pathsep + existing
    return environment


def read_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.getenv(name)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finish_clearml(safe_root: Path, receipt: Path, result_tag: str) -> dict:
    """Send aggregate artifacts only, after the prediction lock already exists."""

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
        "--stage", RUN_STAGE,
        "--project", CLEARML_PROJECT,
        "--task-name", f"{NOTEBOOK_CODE}-protected-inference",
        "--tag", f"notebook-{NOTEBOOK_CODE}",
        "--tag", "prediction-only",
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


def write_zip(source_root: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                member = ZipInfo(
                    path.relative_to(source_root).as_posix(),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                member.compress_type = ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                with path.open("rb") as source, archive.open(member, "w") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
    return {
        "file": output_path.name,
        "sha256": file_sha256(output_path),
        "size_bytes": output_path.stat().st_size,
    }
'''


def build_cells(configuration: dict) -> list:
    code_name = configuration["code"]
    policy_text = (
        "The Notebook 09B review policy is also verified against the exact method hash."
        if configuration["policy_required"]
        else "No calibration-derived review policy is applied in this prediction notebook."
    )
    prereq = (
        "Notebook 08F method freeze and Notebook 09B policy freeze"
        if configuration["policy_required"]
        else "Notebook 08F method freeze"
    )
    cells = [
        markdown(
            f"""
# {configuration['title']}

**Notebook file:** `{configuration['filename']}`

This notebook runs the frozen Visual Peer Inspector route on {configuration['scene_count']} protected {configuration['cohort']} scenes. The RGB package is opened only after the {prereq} verifies. LocateAnything proposes objects, SAM 2.1 produces masks, DINOv2-small extracts features, label-free matching connects candidates across views, and P1 ranks are fused into one odd-one-out decision.

The notebook records predictions only. It does not mount or open an evaluator package, and it does not report accuracy. {policy_text}
"""
        ),
        markdown(
            """
## Acceptance contract

- The role-specific package must pass its fixed membership, image-count, and content-hash checks.
- The frozen method record must identify P1, label-free identity matching, and the fixed 2, 3, and 5 view contracts.
- The selected prompt and pinned model configuration must rebuild the exact Notebook 08F base-model identity before GPU inference starts.
- Method and policy prerequisites must come from their exact minimal hash-locked datasets. Complete evaluator notebook outputs are forbidden mounts.
- The `gate`, `incremental`, and `full` phases are separate successful Kaggle versions.
- Gate and incremental phases export a deterministic recovery ZIP after CRC and hash verification.
- A full run must preserve zero-proposal, empty-mask, feature-failure, and insufficient-evidence outcomes.
- An identical restart may reuse a shard only when its input, output, and complete pipeline identity hashes match.
- Predictions, embeddings, multi-view decisions, and their inventory are locked before later evaluation.
- Figures contain predicted boxes, SAM masks, masked-on-black objects, P1 evidence, identities, ranks, fused decision, margin, and review signal. They contain no reference colouring.
- ClearML marks partial versions as `recovery` and only complete prediction locks as `final`.
"""
        ),
        code(
            f"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
from IPython.display import Image as NotebookImage, display

sys.dont_write_bytecode = True

NOTEBOOK_FILE = "{configuration['filename']}"
NOTEBOOK_CODE = "{code_name}"
COHORT_KEY = "{configuration['cohort']}"
COHORT_ID = "{configuration['cohort_id']}"
EXPECTED_SCENES = {configuration['scene_count']}
EXPECTED_IMAGES = {configuration['image_count']}
RUN_STAGE = "{configuration['stage']}"
CLEARML_PROJECT = "{configuration['project']}"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
RUN_PHASE = os.getenv("VPI_RUN_PHASE", "gate").strip().lower()
if RUN_PHASE not in {{"gate", "incremental", "full"}}:
    raise ValueError("VPI_RUN_PHASE must be gate, incremental, or full")
MAXIMUM_NEW_SHARDS = int(os.getenv("VPI_MAXIMUM_NEW_SHARDS", "10"))
if MAXIMUM_NEW_SHARDS < 1:
    raise ValueError("VPI_MAXIMUM_NEW_SHARDS must be positive")
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/working/{configuration['output_name']}"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/{configuration['output_name']}",
)).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RECOVERY_OUTPUT_ROOT = Path(os.getenv(
    "VPI_RECOVERY_OUTPUT_ROOT",
    f"/kaggle/working/notebook-{{NOTEBOOK_CODE}}-recovery"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else f".local/runs/notebook-{{NOTEBOOK_CODE}}-recovery",
)).expanduser().resolve()
RECOVERY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
"""
        ),
        code(HELPERS),
        markdown("## Resolve and verify protected inputs"),
        code(
            f"""
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/protected-evaluation/{configuration['filename']}",
)

MODEL_INPUT_ROOT = find_model_input(COHORT_ID)
SELECTED_PROMPT = find_named_file("selected_prompt.json", "VPI_SELECTED_PROMPT")
METHOD_FREEZE = find_named_file("method_freeze.json", "VPI_METHOD_FREEZE")
APPLICATION_CONTRACT = find_named_file("application_output_contract.json", "VPI_APPLICATION_CONTRACT")
POLICY_FREEZE = {"find_named_file('review_policy_freeze.json', 'VPI_POLICY_FREEZE')" if configuration['policy_required'] else "None"}

from visual_peer_inspector.workflows.protected_cohort_inference import (
    assert_evaluator_absent,
    load_method_freeze,
    load_policy_freeze,
    verify_protected_input,
)
from visual_peer_inspector.data.cohort_contract import {"INTERNAL_LOCK_CONTRACT" if configuration['policy_required'] else "CALIBRATION_CONTRACT"} as COHORT_CONTRACT

package_record = verify_protected_input(MODEL_INPUT_ROOT, COHORT_CONTRACT)
method_record = load_method_freeze(
    METHOD_FREEZE,
    application_contract_path=APPLICATION_CONTRACT,
    require_resource=True,
)
policy_record = None if POLICY_FREEZE is None else load_policy_freeze(
    POLICY_FREEZE,
    method_sha256=method_record["sha256"],
    require_resource=True,
)
permitted_roots = [CODE_ROOT, MODEL_INPUT_ROOT, SELECTED_PROMPT.parent, METHOD_FREEZE.parent]
if POLICY_FREEZE is not None:
    permitted_roots.append(POLICY_FREEZE.parent)
assert_evaluator_absent(kaggle_roots(), permitted_roots=permitted_roots)

display({{
    "cohort": COHORT_ID,
    "scenes": EXPECTED_SCENES,
    "images": EXPECTED_IMAGES,
    "method record SHA-256": method_record["sha256"],
    "policy record SHA-256": None if policy_record is None else policy_record["sha256"],
    "view contracts": method_record["view_contracts"],
    "evaluator mounted": False,
}})
"""
        ),
        markdown("## Prepare the pinned GPU runtime"),
        code(
            """
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
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
    ],
    check=True,
    timeout=600,
    env=install_environment,
)

import torch
from huggingface_hub import hf_hub_download
from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.sam2 import Sam2Configuration

if not torch.cuda.is_available():
    raise RuntimeError("Protected inference requires a Kaggle GPU")
sam_configuration = Sam2Configuration()
SAM_CHECKPOINT = Path(hf_hub_download(
    repo_id=sam_configuration.model_repository,
    filename=sam_configuration.checkpoint_filename,
    revision=sam_configuration.model_revision,
    cache_dir="/tmp/vpi-sam21-model-cache",
))
if SAM_CHECKPOINT.stat().st_size != sam_configuration.checkpoint_size_bytes or file_sha256(SAM_CHECKPOINT) != sam_configuration.checkpoint_sha256:
    raise RuntimeError("The pinned SAM 2.1 checkpoint identity differs")

display({
    "GPU": torch.cuda.get_device_name(0),
    "GPU count": torch.cuda.device_count(),
    "LocateAnything revision": "c32291ca5e996f5a7a485845b4f57a233936bba0",
    "SAM 2.1 revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    "DINOv2-small revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
})
"""
        ),
        markdown(
            """
## Run the selected bounded phase

The gate processes one 100-image shard through all four model stages. An incremental version processes at most `VPI_MAXIMUM_NEW_SHARDS` new shards per stage. Attach one prior private recovery output through `VPI_RESUME_ROOT`. A full version verifies all restored shards, completes any missing work, and publishes the prediction lock only after exact image coverage.
"""
        ),
        code(
            """
BASE_COMMAND = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_protected_cohort_inference.py"),
    "--cohort", COHORT_KEY,
    "--model-input-root", str(MODEL_INPUT_ROOT),
    "--method-freeze", str(METHOD_FREEZE),
    "--application-contract", str(APPLICATION_CONTRACT),
    "--selected-prompt", str(SELECTED_PROMPT),
    "--sam-checkpoint", str(SAM_CHECKPOINT),
    "--output-root", str(OUTPUT_ROOT),
    "--device", "cuda",
    "--shard-size", "100",
    "--recovery-output-directory", str(RECOVERY_OUTPUT_ROOT),
]
if POLICY_FREEZE is not None:
    BASE_COMMAND.extend(["--policy-freeze", str(POLICY_FREEZE)])
for mounted in kaggle_roots():
    BASE_COMMAND.extend(["--mounted-root", str(mounted)])

recovery_pair = find_recovery_pair(NOTEBOOK_CODE)
if recovery_pair is not None:
    BASE_COMMAND.extend(["--recovery-bundle", str(recovery_pair[0]), "--recovery-manifest", str(recovery_pair[1])])
phase_command = [*BASE_COMMAND, "--run-mode", RUN_PHASE]
if RUN_PHASE == "incremental":
    phase_command.extend(["--maximum-new-shards", str(MAXIMUM_NEW_SHARDS)])
preflight_path = OUTPUT_ROOT / "preflight/model_preflight.json"
preflight_command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_frozen_development_pipeline.py"),
    "preflight",
    "--cohort", COHORT_KEY,
    "--model-input-root", str(MODEL_INPUT_ROOT),
    "--selected-prompt", str(SELECTED_PROMPT),
    "--sam-checkpoint", str(SAM_CHECKPOINT),
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
subprocess.run(phase_command, check=True, env=inference_environment)
COHORT_ROOT = OUTPUT_ROOT / COHORT_ID
PROTECTED_COMPLETE = RUN_PHASE == "full" and (COHORT_ROOT / "model-safe/prediction_lock.json").is_file()
if RUN_PHASE == "gate":
    phase_record = json.loads((COHORT_ROOT / "gate_record.json").read_text(encoding="utf-8"))
    if phase_record["status"] != "complete" or phase_record["covered_image_count"] != 100:
        raise RuntimeError("The protected one-shard gate did not complete")
elif RUN_PHASE == "incremental":
    phase_record = json.loads((COHORT_ROOT / "incremental_progress.json").read_text(encoding="utf-8"))
else:
    phase_record = json.loads((COHORT_ROOT / "safe/run_record.json").read_text(encoding="utf-8"))
recovery_manifests = sorted(RECOVERY_OUTPUT_ROOT.glob("*-recovery-manifest.json"))
display({
    "model preflight": preflight_record["status"],
    "phase": RUN_PHASE,
    "status": phase_record["status"],
    "complete prediction lock": PROTECTED_COMPLETE,
    "recovery manifest": None if not recovery_manifests else str(recovery_manifests[0]),
    "evaluator opened": phase_record["evaluator_opened"],
})
"""
        ),
        markdown(
            f"""
## Verify the complete prediction lock

The models load sequentially so their weights do not compete for GPU memory. A full run gives all {configuration['image_count']:,} images either a prediction or an explicit terminal outcome. Partial phases finish successfully with a recovery artifact and do not create a final lock.
"""
        ),
        code(
            """
SAFE_ROOT = COHORT_ROOT / "safe"
MODEL_SAFE_ROOT = COHORT_ROOT / "model-safe"
run_record = None
prediction_lock = None
if PROTECTED_COMPLETE:
    run_record = json.loads((SAFE_ROOT / "run_record.json").read_text(encoding="utf-8"))
    prediction_lock = json.loads((MODEL_SAFE_ROOT / "prediction_lock.json").read_text(encoding="utf-8"))
    display({
        "status": run_record["status"],
        "scenes": run_record["scene_count"],
        "images": run_record["image_count"],
        "prediction rows": run_record["prediction_bundle"]["prediction_count"],
        "multi-view results": run_record["prediction_bundle"]["multiview_result_count"],
        "prediction lock": run_record["prediction_bundle"]["prediction_lock_sha256"],
        "evaluator opened": run_record["evaluator_opened"],
    })
    if run_record["status"] != "completed" or run_record["image_count"] != EXPECTED_IMAGES:
        raise RuntimeError("Protected inference completion checks failed")
else:
    display({"final lock": "not published", "next action": "attach the recovery output to the next Kaggle version"})
"""
        ),
        markdown("## Coverage, terminal outcomes, and review signals"),
        code(
            """
if PROTECTED_COMPLETE:
    view_summary = pd.read_csv(SAFE_ROOT / "view_contract_summary.csv")
    review_summary = pd.read_csv(SAFE_ROOT / "review_signal_summary.csv")
    runtime_receipt = json.loads((SAFE_ROOT / "runtime_receipt.json").read_text(encoding="utf-8"))
    display(view_summary)
    display(review_summary)
    display(pd.DataFrame([
        {"stage": stage, **values}
        for stage, values in runtime_receipt["stages"].items()
    ]))
    display(NotebookImage(filename=str(SAFE_ROOT / "multiview_completion.png")))
else:
    display(pd.DataFrame([
        {"stage": stage, **values}
        for stage, values in phase_record["stages"].items()
    ]))
"""
        ),
        markdown(
            """
## Prediction-only qualitative evidence

Each selected result shows five uploaded views. The rows contain LocateAnything boxes, an all-object SAM overlay, the selected binary mask, the selected object on black, and the P1 unusual-region evidence. The companion SAM panel provides a closer mask inspection. The header records the fused identity, view agreement, score margin, and review signal. All colours describe predictions only.
"""
        ),
        code(
            """
if PROTECTED_COMPLETE:
    inventory = json.loads(
        (COHORT_ROOT / "review-only/qualitative_evidence_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    for item in inventory["items"][:4]:
        display(NotebookImage(filename=str(COHORT_ROOT / item["prediction_panel"])))
        display(NotebookImage(filename=str(COHORT_ROOT / item["sam_evidence_panel"])))
else:
    display({"qualitative panels": "created after exact full coverage"})
"""
        ),
        markdown("## Bounded ClearML finalisation"),
        code(
            """
# Tracking is isolated in a bounded subprocess and cannot hold the completed model run.
if EXECUTION_ENVIRONMENT == "kaggle":
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt")],
            check=True,
            timeout=180,
        )
    except Exception as error:
        print(f"ClearML dependency setup was unavailable: {type(error).__name__}")

tracking_root = SAFE_ROOT if PROTECTED_COMPLETE else OUTPUT_ROOT / "tracking-safe"
tracking_receipt = finish_clearml(
    tracking_root,
    OUTPUT_ROOT / "tracking/clearml_receipt.json",
    "final" if PROTECTED_COMPLETE else "recovery",
)
display(tracking_receipt)
"""
        ),
        markdown(
            """
## Build verified output archives

The repository archive contains aggregate evidence. The review archive contains prediction-only panels and the ClearML receipt. The downstream archive contains the immutable prediction bundle for the next evaluator notebook. RGB inputs, credentials, caches, and evaluator data are not copied into any archive.
"""
        ),
        code(
            """
BUNDLE_ROOT = Path("/kaggle/working") / f"notebook-{NOTEBOOK_CODE}-bundles" if EXECUTION_ENVIRONMENT == "kaggle" else OUTPUT_ROOT / "bundles"
BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
if PROTECTED_COMPLETE:
    bundles = [
        write_zip(SAFE_ROOT, BUNDLE_ROOT / f"notebook-{NOTEBOOK_CODE}-repository-evidence.zip"),
        write_zip(COHORT_ROOT / "review-only", BUNDLE_ROOT / f"notebook-{NOTEBOOK_CODE}-review-evidence.zip"),
        write_zip(MODEL_SAFE_ROOT, BUNDLE_ROOT / f"notebook-{NOTEBOOK_CODE}-downstream-predictions.zip"),
    ]
else:
    recovery_bundle = next(RECOVERY_OUTPUT_ROOT.glob("*-recovery.zip"))
    recovery_manifest = next(RECOVERY_OUTPUT_ROOT.glob("*-recovery-manifest.json"))
    bundles = [
        {"file": recovery_bundle.name, "sha256": file_sha256(recovery_bundle), "size_bytes": recovery_bundle.stat().st_size},
        {"file": recovery_manifest.name, "sha256": file_sha256(recovery_manifest), "size_bytes": recovery_manifest.stat().st_size},
    ]
(BUNDLE_ROOT / "bundle-checksums.json").write_text(
    json.dumps({"schema_version": 1, "bundles": bundles}, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
display(pd.DataFrame(bundles))
"""
        ),
        markdown(
            """
## Interpretation boundary

This notebook establishes that the frozen route produced a complete, reproducible, prediction-only record for the protected cohort. Its figures show what the system predicted, including failures and uncertain cases. Accuracy, calibration quality, and final performance are determined only by the separate evaluator notebook after this prediction lock has been preserved.
"""
        ),
    ]
    return cells


def build() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for configuration in CONFIGURATIONS:
        cells = build_cells(configuration)
        for index, cell in enumerate(cells):
            payload = (
                f"{configuration['filename']}|{index}|{cell['cell_type']}|"
                f"{''.join(cell.get('source', []))}"
            ).encode()
            cell["id"] = sha256(payload).hexdigest()[:12]
        notebook = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3"},
            },
        )
        nbformat.write(notebook, NOTEBOOK_ROOT / configuration["filename"])
        metadata = {
            "id": f"esreekarreddy/{configuration['filename'].removesuffix('.ipynb').replace('_', '-')}",
            "title": configuration["title"],
            "code_file": configuration["filename"],
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": configuration["dataset_sources"],
            "competition_sources": [],
            "kernel_sources": configuration["kernel_sources"],
            "model_sources": [],
        }
        (NOTEBOOK_ROOT / f"kernel-metadata-{configuration['code']}.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    build()
