#!/usr/bin/env python3
"""Build the optional post-freeze full-pool sensitivity notebooks."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import nbformat
from notebook_runtime_cells import harden_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks/post-freeze-sensitivity"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


DISCOVERY = r'''
def kaggle_roots() -> list[Path]:
    root = Path("/kaggle/input")
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.as_posix()) if root.is_dir() else []


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents, *kaggle_roots()))
    for candidate in candidates:
        for resolved in (candidate, candidate / "code"):
            if (resolved / "scripts/run_full_pool_release.py").is_file():
                return resolved.resolve()
    raise FileNotFoundError("Could not locate the Visual Peer Inspector code snapshot")


def find_named_file(name: str, variable: str) -> Path:
    configured = os.getenv(variable)
    if configured and Path(configured).expanduser().is_file():
        return Path(configured).expanduser().resolve()
    matches = sorted({path.resolve() for root in kaggle_roots() for path in root.rglob(name) if path.is_file()})
    if not matches:
        raise FileNotFoundError(f"Could not locate {name}")
    digests = {sha256(path.read_bytes()).hexdigest() for path in matches}
    if len(digests) != 1:
        raise ValueError(f"Mounted copies of {name} differ")
    return matches[0]


def find_root_with(marker: str, variable: str) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(kaggle_roots())
    matches = []
    for root in candidates:
        if (root / marker).is_file():
            matches.append(root.resolve())
        matches.extend(path.parent.resolve() for path in root.rglob(marker) if path.is_file())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"Expected one mounted root containing {marker}, found {len(unique)}")
    return unique[0]


def find_model_input(cohort_id: str, variable: str) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else kaggle_roots()
    matches = []
    for root in candidates:
        records = [root / "package_record.json"] if (root / "package_record.json").is_file() else []
        records.extend(path for path in root.rglob("package_record.json") if path.is_file())
        for record in records:
            try:
                value = json.loads(record.read_text(encoding="utf-8"))
            except Exception:
                continue
            if value.get("cohort_id") == cohort_id:
                matches.append(record.parent.resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"Expected one model package for {cohort_id}, found {len(unique)}")
    return unique[0]


def code_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(CODE_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return environment


def clearml_mode(environment: dict[str, str]) -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        access = secrets.get_secret("CLEARML_API_ACCESS_KEY")
        secret = secrets.get_secret("CLEARML_API_SECRET_KEY")
    except Exception:
        access = os.getenv("CLEARML_API_ACCESS_KEY")
        secret = os.getenv("CLEARML_API_SECRET_KEY")
    if not access or not secret:
        return "disabled"
    environment.update({
        "CLEARML_API_ACCESS_KEY": access,
        "CLEARML_API_SECRET_KEY": secret,
        "CLEARML_API_HOST": "https://api.clear.ml",
        "CLEARML_WEB_HOST": "https://app.clear.ml",
        "CLEARML_FILES_HOST": "https://files.clear.ml",
    })
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements/confirmation-tracking-lock.txt")],
            check=True,
            timeout=180,
            env=environment,
        )
    except Exception as error:
        print(f"ClearML setup was unavailable: {type(error).__name__}")
        return "disabled"
    return "online"
'''


def release_cells() -> list:
    return [
        markdown(
            """
# ToysAD-8K Post-freeze Full-pool Release

**Notebook file:** `11r-toysad8k-full-pool-release.ipynb`

This optional notebook starts only after Notebook 10B is complete. It prepares the 4,107 eligible training scenes for a descriptive sensitivity analysis. The primary and official results remain unchanged, and this stage cannot be used to tune the frozen method or review policy.

The model-facing RGB package and evaluator-only labels, geometry and crosswalk are physically separate. Original development, calibration and internal-lock role labels are preserved only inside the trusted evaluator path. The remaining 2,107 scenes are recorded as `eligible_remainder`.
"""
        ),
        markdown(
            """
## Execution contract

- The pinned source files and the completed Notebook 10B evidence are verified before any release work starts.
- The eligible pool must contain 4,107 scenes, 545 asset groups, 39 categories and 82,140 RGB images.
- `incremental` mode prepares a bounded number of new scenes and exports one deterministic recovery bundle.
- The release writes an atomic checkpoint before it creates the recovery bundle.
- A later version restores exactly one matching recovery bundle and rejects changed state.
- `full` mode publishes the model and evaluator resources only after complete coverage.
- ClearML receives aggregate counts, one role chart and run receipts. It receives no scene identifiers, labels, geometry or RGB images.
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
NOTEBOOK_FILE = "11r-toysad8k-full-pool-release.ipynb"
RUN_MODE = os.getenv("VPI_RUN_MODE", "incremental").strip().lower()
if RUN_MODE not in {"incremental", "full"}:
    raise ValueError("VPI_RUN_MODE must be incremental or full")
MAXIMUM_NEW_SCENES = int(os.getenv("VPI_MAXIMUM_NEW_SCENES", "100" if RUN_MODE == "incremental" else "4107"))
OUTPUT_ROOT = Path(os.getenv("VPI_OUTPUT_ROOT", "/kaggle/working/toysad8k_full_pool_release")).expanduser().resolve()
RECOVERY_ROOT = Path(os.getenv("VPI_RECOVERY_OUTPUT", "/kaggle/working/toysad8k_full_pool_release_recovery")).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
"""
        ),
        code(DISCOVERY),
        markdown("## Locate and verify the trusted inputs"),
        code(
            r'''
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/post-freeze-sensitivity/11r-toysad8k-full-pool-release.ipynb",
)
SOURCE_ROOT = find_root_with("toysad8k_scene_master.json", "VPI_SOURCE_ROOT")
ADMINISTRATIVE_ROOT = find_root_with("restricted_source_map.jsonl", "VPI_ADMINISTRATIVE_ROOT")

official_candidates = []
for root in kaggle_roots():
    for path in root.rglob("evaluation_complete.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("stage") == "10b" and value.get("status") == "complete":
            official_candidates.append(path.parent.resolve())
if len(set(official_candidates)) != 1:
    raise ValueError("Expected one completed Notebook 10B result")
OFFICIAL_RESULT_ROOT = official_candidates[0]

required_source_files = {
    "master": find_named_file("toysad8k_scene_master.json", "VPI_MASTER_JSON"),
    "catalogue": find_named_file("scene_catalogue.sqlite3", "VPI_CATALOGUE"),
    "readiness": find_named_file("readiness_audit.json", "VPI_READINESS_AUDIT"),
    "archive_index": find_named_file("archive_index.sqlite3", "VPI_ARCHIVE_INDEX"),
    "archive": find_named_file("toysAD8K.zip", "VPI_ARCHIVE_ZIP"),
    "archive_manifest": find_named_file("manifest.json", "VPI_ARCHIVE_MANIFEST"),
    "official_split": find_named_file("ToysAD-8K-split.json", "VPI_OFFICIAL_SPLIT"),
}
display({"Notebook 10B verified input located": True, "source inputs located": len(required_source_files)})
'''
        ),
        markdown("## Restore one previous version when required"),
        code(
            r'''
resume_root = os.getenv("VPI_RESUME_ROOT")
recovery_arguments = []
if resume_root:
    source = Path(resume_root).expanduser().resolve(strict=True)
    bundles = sorted(source.rglob("full-pool-release-recovery.zip"))
    manifests = sorted(source.rglob("full-pool-release-recovery-manifest.json"))
    if len(bundles) != 1 or len(manifests) != 1:
        raise ValueError("VPI_RESUME_ROOT must contain exactly one release recovery pair")
    recovery_arguments = ["--recovery-bundle", str(bundles[0]), "--recovery-manifest", str(manifests[0])]
display({"previous recovery attached": bool(recovery_arguments), "maximum new scenes": MAXIMUM_NEW_SCENES})
'''
        ),
        markdown("## Prepare or resume the full-pool release"),
        code(
            """
environment = code_environment()
tracking_mode = clearml_mode(environment)
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_full_pool_release.py"),
    "--master-json", str(required_source_files["master"]),
    "--catalogue", str(required_source_files["catalogue"]),
    "--readiness-audit", str(required_source_files["readiness"]),
    "--archive-index", str(required_source_files["archive_index"]),
    "--archive-zip", str(required_source_files["archive"]),
    "--archive-manifest", str(required_source_files["archive_manifest"]),
    "--official-split", str(required_source_files["official_split"]),
    "--administrative-root", str(ADMINISTRATIVE_ROOT),
    "--official-result-root", str(OFFICIAL_RESULT_ROOT),
    "--output-root", str(OUTPUT_ROOT),
    "--maximum-new-scenes", str(MAXIMUM_NEW_SCENES),
    "--recovery-output-directory", str(RECOVERY_ROOT),
    "--clearml-mode", tracking_mode,
    *recovery_arguments,
]
subprocess.run(command, check=True, env=environment)
"""
        ),
        markdown("## Review aggregate release evidence"),
        code(
            """
safe_root = OUTPUT_ROOT / "safe"
display(pd.read_csv(safe_root / "primary_role_coverage.csv"))
display(NotebookImage(filename=str(safe_root / "primary_role_coverage.png")))
run_record = json.loads((safe_root / "run_record.json").read_text(encoding="utf-8"))
display({key: run_record[key] for key in (
    "status", "completed_scene_count", "remaining_scene_count",
    "descriptive_sensitivity_only", "no_retuning",
)})
"""
        ),
        markdown(
            """
## Output boundary

For an incremental run, publish only the recovery directory as a private Kaggle dataset version and attach exactly that version to the next run. For a completed run, publish `full-pool-model-input`, `full-pool-evaluator-only`, and `publish/post-freeze-release` as three separate private resources. Notebook 11A mounts the model input and release receipt only. Notebook 11B mounts the evaluator resource only after the prediction lock exists.
"""
        ),
    ]


def inference_cells() -> list:
    return [
        markdown(
            """
# ToysAD-8K Post-freeze Full-pool Frozen Inference

**Notebook file:** `11a-toysad8k-full-pool-frozen-inference.ipynb`

This optional descriptive sensitivity notebook applies the exact frozen LocateAnything, SAM 2.1, DINOv2-small, identity and P1 route to the 4,107-scene eligible training pool. It runs only after the official result is complete and Notebook 11R has released a separate model-facing resource.

The run is prediction-only. It never mounts evaluator labels or geometry and cannot change the method, prompt, view contracts, fusion rule or review policy.
"""
        ),
        markdown(
            """
## Inference and recovery contract

- A bounded one-image preflight loads LocateAnything, SAM 2.1 and DINOv2-small in sequence before a long run starts.
- `gate` then processes one shard through every component before a longer run.
- `incremental` processes a bounded number of new shards and exports a hash-locked recovery bundle.
- `full` publishes immutable predictions only after all 82,140 images have a result or a retained failure.
- Exact stage receipts prevent reuse after any input, method or policy change.
- The saved evidence includes runtime, terminal failures, view-contract tables, review signals and prediction-only panels with boxes, masks, masked objects, identities, ranks and fused odd-one-out results.
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
NOTEBOOK_FILE = "11a-toysad8k-full-pool-frozen-inference.ipynb"
RUN_MODE = os.getenv("VPI_RUN_MODE", "gate").strip().lower()
if RUN_MODE not in {"gate", "incremental", "full"}:
    raise ValueError("VPI_RUN_MODE must be gate, incremental or full")
OUTPUT_ROOT = Path(os.getenv("VPI_OUTPUT_ROOT", "/kaggle/working/toysad8k_full_pool_inference")).expanduser().resolve()
RECOVERY_ROOT = Path(os.getenv("VPI_RECOVERY_OUTPUT", "/kaggle/working/toysad8k_full_pool_inference_recovery")).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
"""
        ),
        code(DISCOVERY),
        markdown("## Verify the frozen resources"),
        code(
            r'''
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/post-freeze-sensitivity/11a-toysad8k-full-pool-frozen-inference.ipynb",
)
MODEL_INPUT = find_model_input("toysad8k-eligible-train-full-pool-4107-v1", "VPI_MODEL_INPUT_ROOT")
RELEASE_RECEIPT = find_named_file("full_pool_release_receipt.json", "VPI_RELEASE_RECEIPT")
METHOD_FREEZE = find_named_file("method_freeze.json", "VPI_METHOD_FREEZE")
APPLICATION_CONTRACT = find_named_file("application_output_contract.json", "VPI_APPLICATION_CONTRACT")
POLICY_FREEZE = find_named_file("review_policy_freeze.json", "VPI_POLICY_FREEZE")
SELECTED_PROMPT = find_named_file("selected_prompt.json", "VPI_SELECTED_PROMPT")
SAM_CHECKPOINT = find_named_file("sam2.1_hiera_small.pt", "VPI_SAM_CHECKPOINT")

package = json.loads((MODEL_INPUT / "package_record.json").read_text(encoding="utf-8"))
if package.get("cohort_id") != "toysad8k-eligible-train-full-pool-4107-v1":
    raise ValueError("Attached model package is not the eligible full-pool release")
display({"scene count": package["scene_count"], "image count": package["image_count"], "run mode": RUN_MODE})
'''
        ),
        markdown("## Restore one exact prediction checkpoint"),
        code(
            r'''
resume_root = os.getenv("VPI_RESUME_ROOT")
recovery_arguments = []
if resume_root:
    source = Path(resume_root).expanduser().resolve(strict=True)
    bundles = sorted(source.rglob("full-pool-inference-recovery.zip"))
    manifests = sorted(source.rglob("full-pool-inference-recovery-manifest.json"))
    if len(bundles) != 1 or len(manifests) != 1:
        raise ValueError("VPI_RESUME_ROOT must contain exactly one inference recovery pair")
    recovery_arguments = ["--recovery-bundle", str(bundles[0]), "--recovery-manifest", str(manifests[0])]
display({"previous recovery attached": bool(recovery_arguments)})
'''
        ),
        markdown("## Run the frozen automatic route"),
        code(
            """
environment = code_environment()
common_arguments = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_full_pool_inference.py"),
    "--model-input-root", str(MODEL_INPUT),
    "--release-receipt", str(RELEASE_RECEIPT),
    "--method-freeze", str(METHOD_FREEZE),
    "--application-contract", str(APPLICATION_CONTRACT),
    "--policy-freeze", str(POLICY_FREEZE),
    "--selected-prompt", str(SELECTED_PROMPT),
    "--sam-checkpoint", str(SAM_CHECKPOINT),
    "--output-root", str(OUTPUT_ROOT),
    *sum((["--mounted-root", str(root)] for root in kaggle_roots()), []),
]
preflight_path = OUTPUT_ROOT / "preflight/full-pool-model-preflight.json"
preflight_command = [
    *common_arguments,
    "--run-mode", "gate",
    "--preflight-only",
    "--preflight-output", str(preflight_path),
]
subprocess.run(preflight_command, check=True, timeout=1800, env=environment)
preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
if preflight_record.get("status") != "passed":
    raise RuntimeError("The bounded model preflight did not pass")

tracking_mode = clearml_mode(environment)
command = [
    *common_arguments,
    "--run-mode", RUN_MODE,
    "--shard-size", os.getenv("VPI_SHARD_SIZE", "100"),
    "--maximum-new-shards", os.getenv("VPI_MAXIMUM_NEW_SHARDS", "10"),
    "--recovery-output-directory", str(RECOVERY_ROOT),
    "--clearml-mode", tracking_mode,
    *recovery_arguments,
]
if os.getenv("VPI_PRUNE_AFTER_RECOVERY", "1") == "1" and RUN_MODE != "full":
    command.append("--prune-after-recovery")
environment["HF_HUB_OFFLINE"] = "1"
environment["TRANSFORMERS_OFFLINE"] = "1"
subprocess.run(command, check=True, env=environment)
display({"model preflight": preflight_record["status"], "run mode": RUN_MODE})
"""
        ),
        markdown("## Review prediction-only evidence"),
        code(
            """
cohort_root = OUTPUT_ROOT / "toysad8k-eligible-train-full-pool-4107-v1"
safe_root = cohort_root / "safe"
if safe_root.is_dir():
    display(pd.read_csv(safe_root / "view_contract_summary.csv"))
    display(pd.read_csv(safe_root / "review_signal_summary.csv"))
    display(NotebookImage(filename=str(safe_root / "multiview_completion.png")))
    panels = sorted((cohort_root / "review-only/figures").glob("*.png"))
    for path in panels[:12]:
        display(NotebookImage(filename=str(path)))
else:
    progress = json.loads((OUTPUT_ROOT / "tracking-safe/run_record.json").read_text(encoding="utf-8"))
    display(progress)
"""
        ),
        markdown(
            """
## Prediction publication boundary

Incremental output is a private recovery resource, not a scientific result. After full coverage, publish the cohort `model-safe`, `safe`, runtime and review-only evidence as the Notebook 11A output. Notebook 11B receives the locked predictions and the separately published evaluator resource. No Notebook 11A output can replace or revise the primary and official findings.
"""
        ),
    ]


def evaluator_cells() -> list:
    return [
        markdown(
            """
# ToysAD-8K Post-freeze Full-pool Sensitivity Evaluation

**Notebook file:** `11b-toysad8k-full-pool-sensitivity-evaluation.ipynb`

This optional evaluator opens the separate full-pool labels and geometry only after the Notebook 11A prediction lock is verified. It describes how the unchanged frozen route behaves across all 4,107 eligible training scenes. It does not select a model, threshold, prompt or policy and does not alter the primary or official results.
"""
        ),
        markdown(
            """
## Evaluation contract

- The Notebook 11R release receipt links the analysis to the completed Notebook 10B result.
- The Notebook 11A receipt links the exact model package, method, policy and prediction lock.
- Correspondence between predicted and reference objects is computed only inside this evaluator.
- All failures remain in the denominators.
- Results are reported for the frozen 2, 3 and 5 view contracts with clustered confidence intervals, category and subgroup tables, reliability, pipeline attrition and error taxonomy.
- Evaluator overlays and masked-object panels remain review-only. ClearML receives aggregate tables and charts only.
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
NOTEBOOK_FILE = "11b-toysad8k-full-pool-sensitivity-evaluation.ipynb"
OUTPUT_ROOT = Path(os.getenv("VPI_OUTPUT_ROOT", "/kaggle/working/toysad8k_full_pool_sensitivity_evaluation")).expanduser().resolve()
CONTRACT_PATH = OUTPUT_ROOT.parent / "evaluation_contract_11b.json"
"""
        ),
        code(DISCOVERY),
        markdown("## Locate the locked prediction and evaluator resources"),
        code(
            r'''
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/post-freeze-sensitivity/11b-toysad8k-full-pool-sensitivity-evaluation.ipynb",
)
RELEASE_RECEIPT = find_named_file("full_pool_release_receipt.json", "VPI_RELEASE_RECEIPT")
INFERENCE_RECEIPT = find_named_file("full_pool_inference_receipt.json", "VPI_INFERENCE_RECEIPT")

prediction_roots = []
evaluator_roots = []
for root in kaggle_roots():
    for path in root.rglob("prediction_lock.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("cohort_id") == "toysad8k-eligible-train-full-pool-4107-v1":
            prediction_roots.append(path.parent.parent.resolve())
    for path in root.rglob("evaluator_manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("cohort_id") == "toysad8k-eligible-train-full-pool-4107-v1":
            evaluator_roots.append(path.parent.resolve())
if len(set(prediction_roots)) != 1 or len(set(evaluator_roots)) != 1:
    raise ValueError("Expected one locked prediction resource and one evaluator resource")
MOUNT_ROOTS = sorted(set([*kaggle_roots(), prediction_roots[0], evaluator_roots[0]]))
display({"prediction resource located": True, "evaluator resource located": True})
'''
        ),
        markdown("## Lock the evaluator inputs before opening labels"),
        code(
            """
common = [
    "--release-receipt", str(RELEASE_RECEIPT),
    "--inference-receipt", str(INFERENCE_RECEIPT),
    *sum((["--mount-root", str(root)] for root in MOUNT_ROOTS), []),
]
subprocess.run([
    sys.executable,
    str(CODE_ROOT / "scripts/run_full_pool_evaluation.py"),
    "contract",
    *common,
    "--output", str(CONTRACT_PATH),
], check=True, env=code_environment())
contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
display({"contract locked": contract["status"], "labels parsed during contract creation": contract["label_files_parsed_during_contract_creation"]})
"""
        ),
        markdown("## Run the descriptive sensitivity evaluation"),
        code(
            """
subprocess.run([
    sys.executable,
    str(CODE_ROOT / "scripts/run_full_pool_evaluation.py"),
    "evaluate",
    *common,
    "--evaluation-contract", str(CONTRACT_PATH),
    "--output-root", str(OUTPUT_ROOT),
    "--bootstrap-repetitions", os.getenv("VPI_BOOTSTRAP_REPETITIONS", "10000"),
    "--panel-count", os.getenv("VPI_PANEL_COUNT", "16"),
], check=True, env=code_environment())
"""
        ),
        markdown("## Review the aggregate metrics and failure evidence"),
        code(
            """
safe_root = OUTPUT_ROOT / "safe"
for name in (
    "summary_metrics.csv",
    "clustered_confidence_intervals.csv",
    "subgroup_and_category_metrics.csv",
    "pipeline_attrition.csv",
    "prediction_failures.csv",
    "error_taxonomy.csv",
):
    table = pd.read_csv(safe_root / name)
    display(name, table.head(30))
for path in sorted((safe_root / "figures").glob("*.png")):
    display(NotebookImage(filename=str(path)))
"""
        ),
        markdown("## Review qualitative evaluator evidence"),
        code(
            """
inventory = json.loads((OUTPUT_ROOT / "review-only/qualitative_panel_inventory.json").read_text(encoding="utf-8"))
display({"review-only panel count": inventory["panel_count"]})
for path in sorted((OUTPUT_ROOT / "review-only").rglob("*.png"))[:16]:
    display(NotebookImage(filename=str(path)))
"""
        ),
        markdown("## Send aggregate evidence to ClearML"),
        code(
            """
environment = code_environment()
tracking_mode = clearml_mode(environment)
subprocess.run([
    sys.executable,
    str(CODE_ROOT / "scripts/run_full_pool_evaluation.py"),
    "track",
    "--result-root", str(OUTPUT_ROOT),
    "--mode", tracking_mode,
], check=True, env=environment)
"""
        ),
        markdown(
            """
## Interpretation boundary

Notebook 11B is a post-freeze sensitivity description. Its tables may show where the fixed route is stronger or weaker across the eligible pool, but they do not supersede Notebook 09D or Notebook 10B and cannot be used to revise the method. Any later alternative is a new study with a new protocol.
"""
        ),
    ]


def write_notebook(path: Path, cells: list) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell["id"] = sha256(f"{path.name}:{index}".encode()).hexdigest()[:8]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(harden_notebook(notebook), path)


def write_metadata(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "11r-toysad8k-full-pool-release.ipynb": release_cells(),
        "11a-toysad8k-full-pool-frozen-inference.ipynb": inference_cells(),
        "11b-toysad8k-full-pool-sensitivity-evaluation.ipynb": evaluator_cells(),
    }
    for name, cells in notebooks.items():
        write_notebook(NOTEBOOK_ROOT / name, cells)
    common = {
        "competition_sources": [],
        "enable_internet": True,
        "enable_tpu": False,
        "is_private": True,
        "kernel_type": "notebook",
        "language": "python",
        "model_sources": [],
    }
    write_metadata(NOTEBOOK_ROOT / "kernel-metadata-11r.json", {
        **common,
        "code_file": "11r-toysad8k-full-pool-release.ipynb",
        "dataset_sources": [
            "esreekarreddy/toysad8k-full-private-v1",
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-selection-audit-inputs",
        ],
        "enable_gpu": False,
        "id": "esreekarreddy/toysad-8k-post-freeze-full-pool-release",
        "kernel_sources": ["esreekarreddy/toysad-8k-official-seen-and-unseen-test-evaluation"],
        "title": "ToysAD-8K Post-freeze Full-pool Release",
    })
    write_metadata(NOTEBOOK_ROOT / "kernel-metadata-11a.json", {
        **common,
        "code_file": "11a-toysad8k-full-pool-frozen-inference.ipynb",
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-full-pool-model-input",
            "esreekarreddy/vpi-toysad-full-pool-release-receipt",
            "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
            "esreekarreddy/vpi-toysad-development-method-freeze",
            "esreekarreddy/vpi-toysad-calibration-review-policy",
        ],
        "enable_gpu": True,
        "id": "esreekarreddy/toysad-8k-post-freeze-full-pool-frozen-inference",
        "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4",
        "title": "ToysAD-8K Post-freeze Full-pool Frozen Inference",
    })
    write_metadata(NOTEBOOK_ROOT / "kernel-metadata-11b.json", {
        **common,
        "code_file": "11b-toysad8k-full-pool-sensitivity-evaluation.ipynb",
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/vpi-toysad-full-pool-predictions",
            "esreekarreddy/vpi-toysad-full-pool-evaluator-only",
            "esreekarreddy/vpi-toysad-full-pool-release-receipt",
            "esreekarreddy/vpi-toysad-full-pool-inference-receipt",
            "esreekarreddy/vpi-toysad-development-method-freeze",
            "esreekarreddy/vpi-toysad-calibration-review-policy",
        ],
        "enable_gpu": False,
        "id": "esreekarreddy/toysad-8k-post-freeze-full-pool-sensitivity-evaluation",
        "kernel_sources": [],
        "title": "ToysAD-8K Post-freeze Full-pool Sensitivity Evaluation",
    })


if __name__ == "__main__":
    main()
