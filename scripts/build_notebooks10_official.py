#!/usr/bin/env python3
"""Build the official release and frozen inference notebooks."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "official-evaluation"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


COMMON_DISCOVERY = r'''
def kaggle_roots() -> list[Path]:
    root = Path("/kaggle/input")
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.as_posix()) if root.is_dir() else []


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents, *kaggle_roots()))
    for candidate in candidates:
        if (candidate / "scripts/run_official_inference.py").is_file():
            return candidate.resolve()
        if (candidate / "code/scripts/run_official_inference.py").is_file():
            return (candidate / "code").resolve()
    raise FileNotFoundError("Could not locate the Visual Peer Inspector code snapshot")


def find_named_file(name: str, environment_name: str) -> Path:
    configured = os.getenv(environment_name)
    if configured and Path(configured).expanduser().is_file():
        return Path(configured).expanduser().resolve()
    matches = sorted({path.resolve() for root in kaggle_roots() for path in root.rglob(name) if path.is_file()})
    if not matches:
        raise FileNotFoundError(f"Could not locate {name}")
    digests = {sha256(path.read_bytes()).hexdigest() for path in matches}
    if len(digests) != 1:
        raise ValueError(f"Mounted copies of {name} differ")
    return matches[0]


def find_release_recovery_pair(role: str) -> tuple[Path, Path] | None:
    configured = os.getenv("VPI_RESUME_ROOT")
    sources = [Path(configured).expanduser()] if configured else kaggle_roots()
    bundles = sorted({
        path.resolve()
        for source in sources
        for path in source.rglob(f"official-{role}-release-recovery.zip")
        if path.is_file()
    })
    manifests = sorted({
        path.resolve()
        for source in sources
        for path in source.rglob(f"official-{role}-release-recovery-manifest.json")
        if path.is_file()
    })
    if not bundles and not manifests:
        return None
    if len(bundles) != 1 or len(manifests) != 1:
        raise ValueError("VPI_RESUME_ROOT must contain exactly one official release recovery pair")
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


def enable_tracking(environment: dict[str, str]) -> str:
    access = read_secret("CLEARML_API_ACCESS_KEY")
    secret = read_secret("CLEARML_API_SECRET_KEY")
    if not access or not secret:
        return "disabled"
    environment.update({
        "CLEARML_API_ACCESS_KEY": access,
        "CLEARML_API_SECRET_KEY": secret,
        "CLEARML_API_HOST": "https://api.clear.ml",
        "CLEARML_WEB_HOST": "https://app.clear.ml",
        "CLEARML_FILES_HOST": "https://files.clear.ml",
    })
    return "online"


def prepare_tracking_runtime(mode: str, environment: dict[str, str]) -> str:
    if mode != "online":
        return mode
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt")],
            check=True,
            timeout=180,
            env=environment,
        )
    except Exception as error:
        print(f"ClearML dependency setup was unavailable: {type(error).__name__}")
        return "disabled"
    return mode
'''


def _release_cells() -> list:
    return [
        markdown(
            """
# ToysAD-8K Official Evaluation Release

**Notebook file:** `10r-toysad8k-official-release.ipynb`

This trusted notebook prepares one official split at a time. Run it once for seen-test and once for unseen-test. Across the two runs it writes four physically separate resources: seen model-facing RGB, unseen model-facing RGB, seen evaluator-only records, and unseen evaluator-only records. Each run verifies the released master, the source archive, official split membership, and every extracted RGB member.

Model-facing manifests use opaque identifiers and an exact field allowlist. Geometry, labels, source identifiers, categories, and the private crosswalk remain in evaluator-only resources. They are never passed to a model notebook.
"""
        ),
        markdown(
            """
## Release contract

- Seen-test contains 715 scenes, 113 asset groups, and 14,300 RGB images.
- Unseen-test contains 2,437 scenes, 316 asset groups, and 48,740 RGB images.
- The minimal Notebook 09D receipt must verify protocol completion without retuning before the source archive is opened. This is not a scientific performance acceptance decision.
- Source membership and opaque model membership are verified with separate SHA-256 values.
- Existing files are reused only when their saved content hashes and the complete input signature match.
- Incremental versions process a bounded number of new scenes and publish a deterministic recovery ZIP only after CRC and SHA-256 verification.
- A later version attaches exactly one recovery output through `VPI_RESUME_ROOT`; role, source, membership, and release identities must match.
- Final model and evaluator package records are not written before exact split coverage.
- Safe evidence is aggregate only. ClearML receives no evaluator records, source identifiers, or RGB files.
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
NOTEBOOK_FILE = "10r-toysad8k-official-release.ipynb"
RELEASE_ROLE = os.getenv("VPI_OFFICIAL_RELEASE_ROLE", "seen").strip().lower()
if RELEASE_ROLE not in {"seen", "unseen"}:
    raise ValueError("VPI_OFFICIAL_RELEASE_ROLE must be seen or unseen")
RUN_PHASE = os.getenv("VPI_RUN_PHASE", "incremental").strip().lower()
if RUN_PHASE not in {"incremental", "full"}:
    raise ValueError("VPI_RUN_PHASE must be incremental or full")
MAXIMUM_NEW_SCENES = int(os.getenv("VPI_MAXIMUM_NEW_SCENES", "100"))
if MAXIMUM_NEW_SCENES < 1:
    raise ValueError("VPI_MAXIMUM_NEW_SCENES must be positive")
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    f"/kaggle/working/toysad8k_official_{RELEASE_ROLE}_release" if EXECUTION_ENVIRONMENT == "kaggle" else f".local/runs/toysad8k_official_{RELEASE_ROLE}_release",
)).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RECOVERY_OUTPUT_ROOT = Path(os.getenv(
    "VPI_RECOVERY_OUTPUT_ROOT",
    f"/kaggle/working/official_{RELEASE_ROLE}_release_recovery"
    if EXECUTION_ENVIRONMENT == "kaggle"
    else f".local/runs/official_{RELEASE_ROLE}_release_recovery",
)).expanduser().resolve()
RECOVERY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
"""
        ),
        code(COMMON_DISCOVERY),
        markdown("## Verify the internal-lock release decision"),
        code(
            """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/official-evaluation/10r-toysad8k-official-release.ipynb",
)

LOCK_RESULT_RECEIPT = find_named_file(
    "internal_lock_result_receipt.json",
    "VPI_LOCK_RESULT_RECEIPT",
)
from visual_peer_inspector.workflows.official_release import verify_internal_lock_decision

lock_result = verify_internal_lock_decision(LOCK_RESULT_RECEIPT)
display({
    "internal-lock decision": lock_result["status"],
    "decision SHA-256": lock_result["sha256"],
    "source package opened": False,
})
"""
        ),
        markdown("## Locate the pinned source package"),
        code(
            """
configured = os.getenv("VPI_SOURCE_ROOT")
source_candidates = [Path(configured).expanduser()] if configured else kaggle_roots()
SOURCE_ROOT = next(
    candidate.resolve()
    for candidate in source_candidates
    if (candidate / "toysAD8K.zip").is_file()
    and (candidate / "archive_index.sqlite3").is_file()
    and (candidate / "toysad8k_scene_master.json").is_file()
    and (candidate / "package-checksums.sha256").is_file()
)
MASTER = SOURCE_ROOT / "toysad8k_scene_master.json"
display({"source package located": True, "output root": str(OUTPUT_ROOT)})
"""
        ),
        markdown(
            """
## Build the four separated resources

The split run checks the full archive SHA-256. Every required member is also checked against the archive index, its local ZIP header, size, and CRC. The indexed RGB size, restored state, requested increment, and free working storage are checked before extraction. Set `VPI_OFFICIAL_RELEASE_ROLE` to `seen` or `unseen`; the default is `seen`. Use incremental versions until the complete package can be finalised, then publish the model input and evaluator output as different private datasets.
"""
        ),
        code(
            """
environment = code_environment()
tracking_mode = prepare_tracking_runtime(enable_tracking(environment), environment)
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_official_release.py"),
    "--source-root", str(SOURCE_ROOT),
    "--master", str(MASTER),
    "--output-root", str(OUTPUT_ROOT),
    "--checkpoint-every", "10",
    "--clearml-mode", tracking_mode,
    "--role", RELEASE_ROLE,
    "--lock-result-receipt", str(LOCK_RESULT_RECEIPT),
    "--run-mode", RUN_PHASE,
    "--maximum-new-scenes", str(MAXIMUM_NEW_SCENES),
    "--recovery-output-directory", str(RECOVERY_OUTPUT_ROOT),
]
recovery_pair = find_release_recovery_pair(RELEASE_ROLE)
if recovery_pair is not None:
    command.extend(["--recovery-bundle", str(recovery_pair[0]), "--recovery-manifest", str(recovery_pair[1])])
subprocess.run(command, check=True, env=environment)
"""
        ),
        markdown("## Verify release evidence"),
        code(
            """
from visual_peer_inspector.data.cohort_contract import OFFICIAL_SEEN_TEST_CONTRACT, OFFICIAL_UNSEEN_TEST_CONTRACT
from visual_peer_inspector.data.model_safe_package import verify_model_input_package

contract = OFFICIAL_SEEN_TEST_CONTRACT if RELEASE_ROLE == "seen" else OFFICIAL_UNSEEN_TEST_CONTRACT
root = OUTPUT_ROOT / f"{RELEASE_ROLE}-model-input"
evaluator_root = OUTPUT_ROOT / f"{RELEASE_ROLE}-evaluator-only"
run_record = json.loads((OUTPUT_ROOT / "safe/run_record.json").read_text(encoding="utf-8"))
RELEASE_COMPLETE = run_record["status"] == "completed"
if RELEASE_COMPLETE:
    audit = verify_model_input_package(root, contract=contract)
    evaluator_manifest = json.loads((evaluator_root / "evaluator_manifest.json").read_text(encoding="utf-8"))
    verified = [{
        "cohort": contract.role,
        "scenes": audit.scene_count,
        "images": audit.image_count,
        "model manifest SHA-256": audit.manifest_sha256,
        "evaluator files": len(evaluator_manifest["files"]),
        "evaluator is model input": False,
        "status": "complete",
    }]
else:
    recovery_manifest = next(RECOVERY_OUTPUT_ROOT.glob("*-recovery-manifest.json"))
    verified = [{
        "cohort": contract.role,
        "scenes": run_record["completed_scene_count"],
        "images": run_record["completed_scene_count"] * contract.views_per_scene,
        "model manifest SHA-256": None,
        "evaluator files": 0,
        "evaluator is model input": False,
        "status": "checkpointed",
        "recovery manifest": str(recovery_manifest),
    }]
display(pd.DataFrame(verified))
display(NotebookImage(filename=str(OUTPUT_ROOT / "safe/official_cohort_summary.png")))
"""
        ),
        markdown(
            """
## Publication boundary

For a partial run, publish only the recovery ZIP and sidecar as one private recovery output and attach it to the next version. For a complete seen run, publish `seen-model-input` and `seen-evaluator-only` as two different private datasets. For a complete unseen run, publish `unseen-model-input` and `unseen-evaluator-only` the same way. Do not attach an evaluator dataset to Notebook 10A. Keep `trusted-release-state` as restart evidence outside model inputs. The `safe` directory and bounded ClearML task contain aggregate evidence only. If a split cannot fit the verified working volume, use a larger private output volume or a direct private dataset version upload; do not start extraction after the storage preflight fails.
"""
        ),
    ]


def _inference_cells(*, filename: str, default_cohort: str) -> list:
    return [
        markdown(
            f"""
# ToysAD-8K Official Frozen Inference

**Notebook file:** `{filename}`

This notebook runs the final frozen automatic route on the official model-facing RGB resources. It verifies the exact Notebook 08F method, Notebook 09B review policy, and completed Notebook 09D protocol receipt before opening official RGB. LocateAnything proposes boxes, SAM 2.1 produces masks, DINOv2-small extracts features, label-free matching connects objects across views, and P1 fusion selects the odd-one-out candidate.

Seen-test and unseen-test predictions remain separate. This entry point is permanently pinned to the `{default_cohort}` cohort. `VPI_OFFICIAL_COHORT` may repeat that value, but any conflicting value stops the run before inputs are opened. The notebook never mounts or opens evaluator resources and reports no accuracy metrics.
"""
        ),
        markdown(
            """
## Execution and restart contract

- One shard is completed for each selected cohort before a longer run.
- `gate`, `incremental`, and `full` phases are explicit.
- Incremental runs process a bounded number of new shards and preserve exact receipts.
- A later Kaggle version attaches exactly one private recovery output and restores it with `VPI_RESUME_ROOT`.
- A shard is reused only when its complete pipeline identity, input hash, output hash, and embedding hashes match.
- Failures and insufficient-evidence cases are retained as results.
- The selected prompt and pinned model configuration must rebuild the exact Notebook 08F base-model identity before GPU inference starts.
- Method, policy, and internal decision prerequisites come from exact minimal hash-locked datasets. Complete evaluator notebook outputs are forbidden mounts.
- Prediction panels show boxes, an all-object SAM overlay, the selected binary mask, the selected masked-on-black object, identities, P1 ranks, fused odd-one-out, margin, agreement, and review signal. They contain no truth colouring.
- Gate and incremental runs export a hash-locked recovery ZIP, record an aggregate ClearML recovery task, and remove duplicate working state only after the ZIP passes its CRC check.
- Only a full per-split prediction lock and the final two-split summary carry the `final` ClearML tag.
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
NOTEBOOK_FILE = "__NOTEBOOK_FILE__"
RUN_PHASE = os.getenv("VPI_RUN_PHASE", "gate").strip().lower()
PINNED_OFFICIAL_COHORT = "__DEFAULT_COHORT__"
REQUESTED_OFFICIAL_COHORT = os.getenv("VPI_OFFICIAL_COHORT", "__DEFAULT_COHORT__").strip().lower()
if RUN_PHASE not in {"gate", "incremental", "full"}:
    raise ValueError("VPI_RUN_PHASE must be gate, incremental, or full")
if REQUESTED_OFFICIAL_COHORT != PINNED_OFFICIAL_COHORT:
    raise ValueError(
        f"{NOTEBOOK_FILE} is pinned to {PINNED_OFFICIAL_COHORT}; "
        f"VPI_OFFICIAL_COHORT requested {REQUESTED_OFFICIAL_COHORT!r}"
    )
OFFICIAL_COHORT = PINNED_OFFICIAL_COHORT
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/working/toysad8k_official_inference" if EXECUTION_ENVIRONMENT == "kaggle" else ".local/runs/toysad8k_official_inference",
)).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
""".replace("__NOTEBOOK_FILE__", filename).replace(
                "__DEFAULT_COHORT__", default_cohort
            )
        ),
        code(COMMON_DISCOVERY),
        code(
            r'''
def find_model_input(cohort_id: str) -> Path:
    for root in kaggle_roots():
        for candidate in (root, *(path for path in root.iterdir() if path.is_dir())):
            record_path = candidate / "package_record.json"
            if not record_path.is_file():
                continue
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("cohort_id") == cohort_id and (candidate / "media").is_dir():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not locate model-facing package {cohort_id}")


def find_recovery_pair(cohort_id: str) -> tuple[Path, Path] | None:
    configured = os.getenv("VPI_RESUME_ROOT")
    if not configured:
        return None
    source = Path(configured).expanduser().resolve(strict=True)
    if any(term in source.name.lower() for term in ("evaluator", "label", "crosswalk", "ground-truth")):
        raise ValueError("A model run cannot restore evaluator content")
    bundles = sorted(source.rglob(f"{cohort_id}-recovery.zip"))
    manifests = sorted(source.rglob(f"{cohort_id}-recovery-manifest.json"))
    if len(bundles) != 1 or len(manifests) != 1:
        raise ValueError("VPI_RESUME_ROOT must contain exactly one recovery ZIP and its manifest")
    return bundles[0], manifests[0]


'''
        ),
        markdown("## Verify the frozen prerequisites before RGB access"),
        code(
            """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/official-evaluation/" + NOTEBOOK_FILE,
)

METHOD_FREEZE = find_named_file("method_freeze.json", "VPI_METHOD_FREEZE")
APPLICATION_CONTRACT = find_named_file("application_output_contract.json", "VPI_APPLICATION_CONTRACT")
POLICY_FREEZE = find_named_file("review_policy_freeze.json", "VPI_POLICY_FREEZE")
LOCK_RESULT_RECEIPT = find_named_file("internal_lock_result_receipt.json", "VPI_LOCK_RESULT_RECEIPT")

from visual_peer_inspector.workflows.official_inference import load_lock_result_receipt
from visual_peer_inspector.workflows.protected_cohort_inference import load_method_freeze, load_policy_freeze

method_record = load_method_freeze(
    METHOD_FREEZE,
    application_contract_path=APPLICATION_CONTRACT,
    require_resource=True,
)
policy_record = load_policy_freeze(
    POLICY_FREEZE,
    method_sha256=method_record["sha256"],
    require_resource=True,
)
lock_result = load_lock_result_receipt(
    LOCK_RESULT_RECEIPT,
    method_sha256=method_record["sha256"],
    policy_sha256=policy_record["sha256"],
    require_resource=True,
)
display({
    "method SHA-256": method_record["sha256"],
    "policy SHA-256": policy_record["sha256"],
    "internal-lock result": lock_result["status"],
    "view contracts": method_record["view_contracts"],
})
"""
        ),
        markdown("## Locate model inputs and prepare the pinned GPU runtime"),
        code(
            """
environment = code_environment()
SEEN_INPUT = UNSEEN_INPUT = None
cohort_id = "toysad8k-official-seen-test-715-v1" if OFFICIAL_COHORT == "seen" else "toysad8k-official-unseen-test-2437-v1"
selected_input = find_model_input(cohort_id)
if OFFICIAL_COHORT == "seen":
    SEEN_INPUT = selected_input
else:
    UNSEEN_INPUT = selected_input
environment["SAM2_BUILD_CUDA"] = "0"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-lock.txt")],
    check=True, timeout=900, env=environment,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"],
    check=True, timeout=600, env=environment,
)
import torch
from huggingface_hub import hf_hub_download
from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.sam2 import Sam2Configuration
if not torch.cuda.is_available():
    raise RuntimeError("Official inference requires a Kaggle GPU")
sam_configuration = Sam2Configuration()
SAM_CHECKPOINT = Path(hf_hub_download(
    repo_id=sam_configuration.model_repository,
    filename=sam_configuration.checkpoint_filename,
    revision=sam_configuration.model_revision,
    cache_dir="/tmp/vpi-sam21-model-cache",
))
if SAM_CHECKPOINT.stat().st_size != sam_configuration.checkpoint_size_bytes or file_sha256(SAM_CHECKPOINT) != sam_configuration.checkpoint_sha256:
    raise RuntimeError("The pinned SAM 2.1 checkpoint identity differs")
SELECTED_PROMPT = find_named_file("selected_prompt.json", "VPI_SELECTED_PROMPT")
display({"GPU": torch.cuda.get_device_name(0), "GPU count": torch.cuda.device_count(), "cohort": OFFICIAL_COHORT})
"""
        ),
        markdown("## Run the selected restartable phase"),
        code(
            """
maximum_new_shards = int(os.getenv("VPI_MAXIMUM_NEW_SHARDS", "10"))
tracking_mode = prepare_tracking_runtime(enable_tracking(environment), environment)
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_official_inference.py"),
    "--cohort", OFFICIAL_COHORT,
    "--run-mode", RUN_PHASE,
    "--method-freeze", str(METHOD_FREEZE),
    "--application-contract", str(APPLICATION_CONTRACT),
    "--policy-freeze", str(POLICY_FREEZE),
    "--lock-result-receipt", str(LOCK_RESULT_RECEIPT),
    "--output-root", str(OUTPUT_ROOT),
    "--device", "cuda",
    "--shard-size", "100",
    "--maximum-new-shards", str(maximum_new_shards),
    "--clearml-mode", tracking_mode,
]
if SEEN_INPUT is not None:
    command.extend(["--seen-model-input", str(SEEN_INPUT)])
if UNSEEN_INPUT is not None:
    command.extend(["--unseen-model-input", str(UNSEEN_INPUT)])
if SELECTED_PROMPT is not None and SAM_CHECKPOINT is not None:
    command.extend(["--selected-prompt", str(SELECTED_PROMPT), "--sam-checkpoint", str(SAM_CHECKPOINT)])
cohort_id = "toysad8k-official-seen-test-715-v1" if OFFICIAL_COHORT == "seen" else "toysad8k-official-unseen-test-2437-v1"
recovery = find_recovery_pair(cohort_id)
if recovery is not None:
    command.extend(["--recovery-bundle", str(recovery[0]), "--recovery-manifest", str(recovery[1])])
if RUN_PHASE in {"gate", "incremental"}:
    command.extend(["--export-recovery", "--prune-after-recovery"])
for mounted in kaggle_roots():
    command.extend(["--mounted-root", str(mounted)])
preflight_path = OUTPUT_ROOT / f"preflight/{OFFICIAL_COHORT}-model-preflight.json"
preflight_command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_frozen_development_pipeline.py"),
    "preflight",
    "--cohort", f"official-{OFFICIAL_COHORT}",
    "--model-input-root", str(selected_input),
    "--selected-prompt", str(SELECTED_PROMPT),
    "--sam-checkpoint", str(SAM_CHECKPOINT),
    "--output-path", str(preflight_path),
    "--device", "cuda",
    "--maximum-image-attempts", "5",
]
subprocess.run(preflight_command, check=True, timeout=1800, env=environment)
preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
if preflight_record["status"] != "passed":
    raise RuntimeError("The bounded model preflight did not pass")
environment["HF_HUB_OFFLINE"] = "1"
environment["TRANSFORMERS_OFFLINE"] = "1"
subprocess.run(command, check=True, env=environment)
display({"phase": RUN_PHASE, "cohort": OFFICIAL_COHORT, "model preflight": preflight_record["status"], "maximum new shards per stage": maximum_new_shards})
"""
        ),
        markdown("## Inspect prediction-only evidence"),
        code(
            """
for cohort_id in (
    "toysad8k-official-seen-test-715-v1",
    "toysad8k-official-unseen-test-2437-v1",
):
    root = OUTPUT_ROOT / cohort_id
    progress_path = root / "incremental_progress.json"
    if progress_path.is_file():
        display(pd.DataFrame(json.loads(progress_path.read_text(encoding="utf-8"))["stages"]).T)
    safe_root = root / "safe"
    inventory_path = root / "review-only/qualitative_evidence_inventory.json"
    if inventory_path.is_file():
        display(pd.read_csv(safe_root / "view_contract_summary.csv"))
        display(pd.read_csv(safe_root / "review_signal_summary.csv"))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for item in inventory["items"][:3]:
            display(NotebookImage(filename=str(root / item["prediction_panel"])))
            display(NotebookImage(filename=str(root / item["sam_evidence_panel"])))

publish_root = OUTPUT_ROOT / "publish" / f"official-{OFFICIAL_COHORT}-prediction"
if publish_root.is_dir():
    display({"private prediction resource": str(publish_root)})
"""
        ),
        markdown(
            """
## Publication and output contract

After a full seen run, publish only `publish/official-seen-prediction` as the private dataset `vpi-toysad-official-seen-prediction`. After a full unseen run, publish only `publish/official-unseen-prediction` as `vpi-toysad-official-unseen-prediction`. Each resource contains its locked model-only artifacts and sanitized run record. Do not publish recovery state as a final prediction resource. Notebook 10A Finalize attaches both prediction resources and builds one minimal cross-split summary without opening evaluator data.
"""
        ),
    ]


def _finalize_cells() -> list:
    return [
        markdown(
            """
# ToysAD-8K Official Prediction Summary

**Notebook file:** `10a-finalize-toysad8k-official-prediction-summary.ipynb`

This CPU-only notebook joins the completed seen-test and unseen-test prediction locks into one minimal summary. It opens no RGB or evaluator data and computes no accuracy metric. Its only purpose is to prove that both role-specific model runs used the same frozen method, review policy, and completed internal-lock protocol.
"""
        ),
        markdown(
            """
## Finalization contract

- The private datasets `vpi-toysad-official-seen-prediction` and `vpi-toysad-official-unseen-prediction` must both be mounted.
- Exactly one verified prediction lock must exist for each official cohort.
- Method, policy, cohort membership, and internal-lock receipt hashes must match.
- The output contains aggregate lock metadata only.
- A bounded ClearML finalizer runs after the summary and its artifact manifest are complete.
- Evaluator datasets, labels, source identifiers, RGB files, masks, and embeddings are not mounted or uploaded.
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
NOTEBOOK_FILE = "10a-finalize-toysad8k-official-prediction-summary.ipynb"
EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/working/toysad8k_official_prediction_summary" if EXECUTION_ENVIRONMENT == "kaggle" else ".local/runs/toysad8k_official_prediction_summary",
)).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
"""
        ),
        code(COMMON_DISCOVERY),
        code(
            r'''
def find_prediction_lock(cohort_id: str) -> Path:
    matches = []
    for root in kaggle_roots():
        for path in root.rglob("prediction_lock.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("cohort_id") == cohort_id:
                matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(f"Expected one completed prediction lock for {cohort_id}")
    return matches[0]
'''
        ),
        markdown("## Verify and combine the two prediction locks"),
        code(
            """
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/official-evaluation/10a-finalize-toysad8k-official-prediction-summary.ipynb",
)
METHOD_FREEZE = find_named_file("method_freeze.json", "VPI_METHOD_FREEZE")
APPLICATION_CONTRACT = find_named_file("application_output_contract.json", "VPI_APPLICATION_CONTRACT")
POLICY_FREEZE = find_named_file("review_policy_freeze.json", "VPI_POLICY_FREEZE")
LOCK_RESULT_RECEIPT = find_named_file("internal_lock_result_receipt.json", "VPI_LOCK_RESULT_RECEIPT")
SEEN_LOCK = find_prediction_lock("toysad8k-official-seen-test-715-v1")
UNSEEN_LOCK = find_prediction_lock("toysad8k-official-unseen-test-2437-v1")

environment = code_environment()
tracking_mode = prepare_tracking_runtime(enable_tracking(environment), environment)
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_official_inference.py"),
    "--cohort", "seen",
    "--run-mode", "finalize",
    "--method-freeze", str(METHOD_FREEZE),
    "--application-contract", str(APPLICATION_CONTRACT),
    "--policy-freeze", str(POLICY_FREEZE),
    "--lock-result-receipt", str(LOCK_RESULT_RECEIPT),
    "--seen-prediction-lock", str(SEEN_LOCK),
    "--unseen-prediction-lock", str(UNSEEN_LOCK),
    "--output-root", str(OUTPUT_ROOT),
    "--device", "cpu",
    "--clearml-mode", tracking_mode,
]
for mounted in kaggle_roots():
    command.extend(["--mounted-root", str(mounted)])
subprocess.run(command, check=True, timeout=900, env=environment)
"""
        ),
        markdown("## Inspect the aggregate-only result"),
        code(
            """
SAFE_ROOT = OUTPUT_ROOT / "official-safe"
summary = json.loads((SAFE_ROOT / "official_inference_summary.json").read_text(encoding="utf-8"))
table = pd.read_csv(SAFE_ROOT / "official_prediction_lock_summary.csv")
display(summary)
display(table)
display(NotebookImage(filename=str(SAFE_ROOT / "official_prediction_lock_summary.png")))

from visual_peer_inspector.frozen_resource import verify_frozen_resource

publish_root = OUTPUT_ROOT / "publish" / "official-prediction-summary"
resource_record = verify_frozen_resource(
    publish_root,
    expected_stage="official_prediction_summary_release",
    expected_files={"official_inference_summary.json"},
)
display(resource_record)
"""
        ),
        markdown(
            """
## Publication boundary

Publish only `publish/official-prediction-summary` as the private dataset `vpi-toysad-official-prediction-summary`. Notebook 10B mounts this minimal resource together with the two role-specific prediction datasets and the two physically separate evaluator datasets. Do not publish the complete finalizer output as a model input.
"""
        ),
    ]


def _write_notebook(filename: str, cells: list) -> None:
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    for index, cell in enumerate(cells):
        cell["id"] = sha256(f"{filename}|{index}|{cell.cell_type}|{cell.source}".encode()).hexdigest()[:12]
        if cell.cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    nbformat.write(notebook, NOTEBOOK_ROOT / filename)


def build() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    release_name = "10r-toysad8k-official-release.ipynb"
    inference_name = "10a-toysad8k-official-seen-frozen-inference.ipynb"
    unseen_inference_name = "10a1-toysad8k-official-unseen-frozen-inference.ipynb"
    finalize_name = "10a-finalize-toysad8k-official-prediction-summary.ipynb"
    _write_notebook(release_name, _release_cells())
    _write_notebook(
        inference_name,
        _inference_cells(filename=inference_name, default_cohort="seen"),
    )
    _write_notebook(
        unseen_inference_name,
        _inference_cells(filename=unseen_inference_name, default_cohort="unseen"),
    )
    _write_notebook(finalize_name, _finalize_cells())
    metadata = {
        "10r": {
            "id": "esreekarreddy/toysad-8k-official-release",
            "title": "ToysAD-8K Official Evaluation Release",
            "code_file": release_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [
                "esreekarreddy/toysad8k-full-private-v1",
                "esreekarreddy/vpi-code-snapshot",
                "esreekarreddy/vpi-toysad-internal-lock-decision",
            ],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        },
        "10a": {
            "id": "esreekarreddy/toysad-8k-official-frozen-inference",
            "title": "ToysAD-8K Official Frozen Inference",
            "code_file": inference_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [
                "esreekarreddy/vpi-code-snapshot",
                "esreekarreddy/vpi-toysad-official-seen-model-input",
                "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
                "esreekarreddy/vpi-toysad-development-method-freeze",
                "esreekarreddy/vpi-toysad-calibration-review-policy",
                "esreekarreddy/vpi-toysad-internal-lock-decision",
            ],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        "10a1": {
            "id": "esreekarreddy/toysad-8k-official-unseen-frozen-inference",
            "title": "ToysAD-8K Official Unseen Frozen Inference",
            "code_file": unseen_inference_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [
                "esreekarreddy/vpi-code-snapshot",
                "esreekarreddy/vpi-toysad-official-unseen-model-input",
                "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
                "esreekarreddy/vpi-toysad-development-method-freeze",
                "esreekarreddy/vpi-toysad-calibration-review-policy",
                "esreekarreddy/vpi-toysad-internal-lock-decision",
            ],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        "10a-finalize": {
            "id": "esreekarreddy/toysad-8k-official-prediction-summary",
            "title": "ToysAD-8K Official Prediction Summary",
            "code_file": finalize_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [
                "esreekarreddy/vpi-code-snapshot",
                "esreekarreddy/vpi-toysad-official-seen-prediction",
                "esreekarreddy/vpi-toysad-official-unseen-prediction",
                "esreekarreddy/vpi-toysad-development-method-freeze",
                "esreekarreddy/vpi-toysad-calibration-review-policy",
                "esreekarreddy/vpi-toysad-internal-lock-decision",
            ],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    }
    for key, value in metadata.items():
        (NOTEBOOK_ROOT / f"kernel-metadata-{key}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
