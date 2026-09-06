#!/usr/bin/env python3
"""Build the CPU notebooks that release primary experiment resources."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


CONFIGURATIONS = (
    {
        "code": "08e1",
        "folder": "development-freeze",
        "filename": "08e1-toysad8k-development-identity-evaluator-release.ipynb",
        "title": "ToysAD-8K Development Identity Evaluator Release",
        "mode": "development-identity",
        "scene_count": 1_000,
        "image_count": 20_000,
        "project": "Visual Peer Inspector/05 Development Freeze",
        "publications": ("vpi-toysad-development-identity-evaluator",),
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/toysad8k-full-private-eda",
            "esreekarreddy/vpi-toysad-selection-private",
        ],
        "kernel_sources": ["esreekarreddy/toysad-8k-frozen-development-pipeline"],
    },
    {
        "code": "08f1",
        "folder": "protected-evaluation",
        "filename": "08f1-toysad8k-calibration-resource-release.ipynb",
        "title": "ToysAD-8K Calibration Resource Release",
        "mode": "calibration",
        "scene_count": 600,
        "image_count": 12_000,
        "project": "Visual Peer Inspector/06 Protected Evaluation",
        "publications": (
            "vpi-toysad-calibration600-model-input",
            "vpi-toysad-calibration600-evaluator",
        ),
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/toysad8k-full-private-eda",
            "esreekarreddy/vpi-toysad-selection-private",
            "esreekarreddy/vpi-toysad-development-method-freeze",
        ],
        "kernel_sources": [],
    },
    {
        "code": "09b1",
        "folder": "protected-evaluation",
        "filename": "09b1-toysad8k-internal-lock-resource-release.ipynb",
        "title": "ToysAD-8K Internal-Lock Resource Release",
        "mode": "internal-lock",
        "scene_count": 400,
        "image_count": 8_000,
        "project": "Visual Peer Inspector/06 Protected Evaluation",
        "publications": (
            "vpi-toysad-internal-lock400-model-input",
            "vpi-toysad-internal-lock400-evaluator",
        ),
        "dataset_sources": [
            "esreekarreddy/vpi-code-snapshot",
            "esreekarreddy/toysad8k-full-private-eda",
            "esreekarreddy/vpi-toysad-selection-private",
            "esreekarreddy/vpi-toysad-development-method-freeze",
            "esreekarreddy/vpi-toysad-calibration-review-policy",
        ],
        "kernel_sources": [],
    },
)


SETUP = r'''
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from IPython.display import display

EXECUTION_ENVIRONMENT = "kaggle" if Path("/kaggle/input").is_dir() else "local"
OUTPUT_ROOT = Path(os.getenv(
    "VPI_OUTPUT_ROOT",
    "/kaggle/working/{output_name}" if EXECUTION_ENVIRONMENT == "kaggle"
    else ".local/runs/{output_name}",
)).expanduser().resolve()
RUN_MODE = os.getenv("VPI_RUN_MODE", "full")
MAXIMUM_NEW_SCENES = int(os.getenv("VPI_MAXIMUM_NEW_SCENES", "100"))
'''


HELPERS = r'''
def kaggle_roots() -> list[Path]:
    root = Path("/kaggle/input")
    return [] if not root.is_dir() else sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.as_posix(),
    )


def find_code_root() -> Path:
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.cwd(), *Path.cwd().parents, *kaggle_roots()))
    for candidate in candidates:
        for value in (candidate, candidate / "code"):
            if (value / "scripts/run_primary_resource_gate.py").is_file():
                return value.resolve()
    raise FileNotFoundError("The Visual Peer Inspector code snapshot was not found")


def find_directory(marker: str, environment_name: str) -> Path:
    configured = os.getenv(environment_name)
    if configured:
        candidate = Path(configured).expanduser()
        if (candidate / marker).is_file():
            return candidate.resolve()
    matches = []
    for root in kaggle_roots():
        candidates = (root, *(path for path in root.iterdir() if path.is_dir()))
        matches.extend(candidate.resolve() for candidate in candidates if (candidate / marker).is_file())
    unique = sorted(set(matches), key=lambda path: path.as_posix())
    if len(unique) != 1:
        raise FileNotFoundError(f"Expected one mounted directory containing {marker}; found {len(unique)}")
    return unique[0]


def find_named_file(name: str, environment_name: str) -> Path:
    configured = os.getenv(environment_name)
    if configured and Path(configured).expanduser().is_file():
        return Path(configured).expanduser().resolve()
    files = sorted({
        path.resolve()
        for root in kaggle_roots()
        for path in root.rglob(name)
        if path.is_file()
    })
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one mounted {name}; found {len(files)}")
    return files[0]


def find_prediction_roots() -> list[Path]:
    configured = os.getenv("VPI_PREDICTION_ROOTS")
    if configured:
        return [Path(value).expanduser().resolve() for value in configured.split(os.pathsep)]
    expected = {
        "toysad8k-development-screen-100-v1",
        "toysad8k-development-confirm-300-v1",
        "toysad8k-development-remainder-600-v1",
    }
    roots = []
    for root in kaggle_roots():
        for lock_path in root.rglob("prediction_lock.json"):
            try:
                value = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if value.get("cohort_id") in expected:
                roots.append(lock_path.parent.resolve())
    if {json.loads((root / "prediction_lock.json").read_text())["cohort_id"] for root in roots} != expected:
        raise FileNotFoundError("The three completed Notebook 08E prediction roots were not found")
    return sorted(set(roots), key=lambda path: path.as_posix())


def find_recovery_pair(mode: str) -> tuple[Path, Path] | None:
    stem = f"{mode}-resource-recovery"
    bundles = [path for root in kaggle_roots() for path in root.rglob(f"{stem}.zip")]
    manifests = [path for root in kaggle_roots() for path in root.rglob(f"{stem}-manifest.json")]
    if not bundles and not manifests:
        return None
    if len(bundles) != 1 or len(manifests) != 1:
        raise ValueError("Attach exactly one prior recovery output version")
    return bundles[0], manifests[0]


def code_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(CODE_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else source + os.pathsep + current
    return environment


def clearml_mode() -> str:
    if EXECUTION_ENVIRONMENT != "kaggle":
        return "disabled"
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        access = client.get_secret("CLEARML_API_ACCESS_KEY")
        secret = client.get_secret("CLEARML_API_SECRET_KEY")
    except Exception:
        return "disabled"
    if not access or not secret:
        return "disabled"
    os.environ.update({
        "CLEARML_API_ACCESS_KEY": access,
        "CLEARML_API_SECRET_KEY": secret,
        "CLEARML_API_HOST": "https://api.clear.ml",
        "CLEARML_WEB_HOST": "https://app.clear.ml",
        "CLEARML_FILES_HOST": "https://files.clear.ml",
    })
    return "online"
'''


def _command_cell(configuration: dict) -> str:
    mode = configuration["mode"]
    common = r'''
CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
from visual_peer_inspector.notebook_provenance import verify_notebook_code_snapshot

CODE_PROVENANCE = verify_notebook_code_snapshot(
    CODE_ROOT,
    "notebooks/__NOTEBOOK_FOLDER__/__NOTEBOOK_FILE__",
)
SOURCE_ROOT = find_directory("toysAD8K.zip", "VPI_SOURCE_ROOT")
ADMINISTRATIVE_ROOT = find_directory("restricted_source_map.jsonl", "VPI_ADMINISTRATIVE_ROOT")
RECOVERY_DIRECTORY = OUTPUT_ROOT.parent / (OUTPUT_ROOT.name + "_recovery")
command = [
    sys.executable,
    str(CODE_ROOT / "scripts/run_primary_resource_gate.py"),
    "--mode", "__MODE__",
    "--administrative-root", str(ADMINISTRATIVE_ROOT),
    "--source-root", str(SOURCE_ROOT),
    "--output-root", str(OUTPUT_ROOT),
    "--run-mode", RUN_MODE,
    "--maximum-new-scenes", str(MAXIMUM_NEW_SCENES),
    "--recovery-output-directory", str(RECOVERY_DIRECTORY),
]
'''.replace("__MODE__", mode).replace(
        "__NOTEBOOK_FOLDER__", configuration["folder"]
    ).replace("__NOTEBOOK_FILE__", configuration["filename"])
    if mode == "development-identity":
        inputs = r'''
for prediction_root in find_prediction_roots():
    command.extend(("--prediction-root", str(prediction_root)))
'''
    else:
        label_marker = "calibration_labels.jsonl" if mode == "calibration" else "internal_lock_labels.jsonl"
        prerequisite = "method_freeze.json" if mode == "calibration" else "review_policy_freeze.json"
        inputs = r'''
SELECTION_ROOT = find_directory("experiment_roles.jsonl", "VPI_SELECTION_ROOT")
LABEL_ROOT = find_directory("__LABEL_MARKER__", "VPI_LABEL_ROOT")
PREREQUISITE = find_named_file("__PREREQUISITE__", "VPI_PREREQUISITE_LOCK")
command.extend((
    "--selection-root", str(SELECTION_ROOT),
    "--label-root", str(LABEL_ROOT),
    "--prerequisite-lock", str(PREREQUISITE),
))
'''.replace("__LABEL_MARKER__", label_marker).replace("__PREREQUISITE__", prerequisite)
        if mode == "internal-lock":
            inputs += r'''
METHOD_LOCK = find_named_file("method_freeze.json", "VPI_METHOD_LOCK")
command.extend(("--method-lock", str(METHOD_LOCK)))
'''
    tail = r'''
recovery = find_recovery_pair("__MODE__")
if recovery is not None:
    command.extend(("--recovery-bundle", str(recovery[0]), "--recovery-manifest", str(recovery[1])))
completed = subprocess.run(command, check=True, capture_output=True, text=True, env=code_environment())
result = json.loads(completed.stdout.strip().splitlines()[-1])
display({
    "status": result["status"],
    "completed scenes": result["completed_scene_count"],
    "remaining scenes": result["remaining_scene_count"],
    "processed this version": result["processed_scene_count"],
    "resumed scenes": result["resumed_scene_count"],
    "exact resume": result["exact_resume"],
})
'''.replace("__MODE__", mode)
    return common + inputs + tail


def _tracking_cell(configuration: dict) -> str:
    stage = {
        "development-identity": "development_identity_evaluator_release",
        "calibration": "calibration_resource_release",
        "internal-lock": "internal_lock_resource_release",
    }[configuration["mode"]]
    return r'''
SAFE_ROOT = OUTPUT_ROOT / ("safe" if result["status"] == "complete" else "tracking-safe")
mode = clearml_mode()
receipt = OUTPUT_ROOT / "tracking" / "clearml_receipt.json"
try:
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/finalize_ds2_clearml.py"),
            "--safe-root", str(SAFE_ROOT),
            "--receipt", str(receipt),
            "--stage", "__STAGE__",
            "--project", "__PROJECT__",
            "--task-name", "__CODE__-primary-resource-gate",
            "--tag", "notebook-__CODE__",
            "--tag", "final" if result["status"] == "complete" else "recovery",
            "--mode", mode,
            "--wall-time-seconds", "180",
        ],
        check=False,
        timeout=240,
        env=code_environment(),
    )
except subprocess.TimeoutExpired:
    display("ClearML timed out after the scientific output was saved")
display({"ClearML mode": mode, "receipt": str(receipt)})
'''.replace("__STAGE__", stage).replace(
        "__PROJECT__", configuration["project"]
    ).replace("__CODE__", configuration["code"])


def build_notebook(configuration: dict) -> nbformat.NotebookNode:
    mode = configuration["mode"]
    if mode == "development-identity":
        purpose = (
            "This CPU stage opens reference object masks only after all development predictions are locked. "
            "It maps each predicted object to an opaque reference identity for Notebook 08F."
        )
        outputs = "Only the evaluator resource is released. No evaluator field enters the model-facing path."
    elif mode == "calibration":
        purpose = (
            f"This CPU stage releases the exact {configuration['scene_count']}-scene {mode} RGB resource "
            "and a physically separate raw evaluator resource after the method freeze."
        )
        outputs = (
            "The model resource contains opaque RGB data only. The evaluator resource contains labels and "
            "reference geometry and is mounted only by the later evaluator notebook."
        )
    else:
        purpose = (
            f"This CPU stage releases the exact {configuration['scene_count']}-scene internal-lock RGB resource "
            "and a physically separate raw evaluator resource after the method and review-policy freezes."
        )
        outputs = (
            "The model resource contains opaque RGB data only. The evaluator resource contains labels and "
            "reference geometry and is mounted only by the later evaluator notebook."
        )
    publication_text = ", ".join(
        f"`{name}`" for name in configuration["publications"]
    )
    cells = [
        markdown(
            f"# {configuration['title']}\n\n"
            f"**Notebook file:** `{configuration['filename']}`\n\n{purpose} {outputs}"
        ),
        markdown(
            "## Acceptance contract\n\n"
            f"- The fixed cohort contains exactly {configuration['scene_count']:,} scenes and "
            f"{configuration['image_count']:,} RGB views.\n"
            "- Cohort membership and prerequisite hashes must match before source bytes are read.\n"
            "- Bounded incremental runs save deterministic per-scene state and a verified recovery ZIP.\n"
            "- A final resource manifest is written only after exact full coverage.\n"
            "- ClearML receives aggregate recovery or final evidence only."
        ),
        code(
            SETUP.format(
                output_name=configuration["filename"].removesuffix(".ipynb").replace("-", "_")
            )
        ),
        code(HELPERS),
        markdown("## Resolve and verify inputs\n\nInputs are located by exact marker files. No broad archive extraction is used."),
        code(_command_cell(configuration)),
        markdown(
            "## Release state\n\nA `complete` result is ready for private dataset publication. A `checkpointed` result "
            "publishes only the recovery pair, which can be attached to the next Kaggle version."
        ),
        code(
            "display({key: value for key, value in result.items() if key not in {'recovery'}})\n"
            "if 'recovery' in result:\n"
            "    display(result['recovery'])"
        ),
        markdown("## Bounded ClearML finalisation\n\nTracking starts after local evidence exists and cannot change the release result."),
        code(_tracking_cell(configuration)),
        markdown(
            "## Handoff\n\nPublish the completed output as the private Kaggle dataset"
            + (" " if len(configuration["publications"]) == 1 else "s ")
            + publication_text
            + ". Model input and evaluator output must remain separate. "
            "If the stage is incomplete, attach the recovery ZIP and manifest to a new version and run again. "
            "Do not publish trusted release state, credentials, or evaluator data as model input."
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    }
    for index, cell in enumerate(notebook.cells):
        cell["id"] = sha256(
            f"{configuration['filename']}:{index}".encode()
        ).hexdigest()[:8]
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return notebook


def main() -> None:
    for configuration in CONFIGURATIONS:
        folder = ROOT / "notebooks" / configuration["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        notebook_path = folder / configuration["filename"]
        nbformat.write(build_notebook(configuration), notebook_path)
        metadata = {
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
        metadata_path = folder / f"kernel-metadata-{configuration['code']}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
