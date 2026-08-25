#!/usr/bin/env python3
"""Build the clean DS1 opaque-package notebook from reviewed cells."""

from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_RELATIVE_PATH = Path(
    "notebooks/data-preparation/toysad8k-ds1-opaque-package/"
    "03.1-toysad8k-ds1-opaque-package.ipynb"
)
NOTEBOOK_PATH = Path(__file__).parents[1] / NOTEBOOK_RELATIVE_PATH


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K DS1 Opaque Package

**Notebook file:** `03.1-toysad8k-ds1-opaque-package.ipynb`

This notebook prepares the fixed DS1 component-screen input. It contains 100 development scenes, all 20 RGB views, and 2,000 images. The local preparation path reads trusted records and the private source. The Kaggle path independently verifies the published opaque package.

The model-facing output contains only opaque scene and object identifiers, RGB media, a neutral manifest, checksums, and a package record. It does not contain labels, categories, source paths, source object order, base-asset identifiers, experiment roles, or evaluator records.
"""
    ),
    markdown(
        """
## Acceptance contract

The run is accepted only when:

- the cohort matches `toysad8k-development-screen-100-v1`;
- the frozen membership hash matches the audited 100-scene screen;
- all 100 scenes contain 20 unique views;
- all 2,000 PNG files pass path, CRC, decode, dimension, and checksum checks;
- the package contains only the declared files;
- a second pass reuses every image without changing any package hash;
- the run bundle contains tables, charts, provenance, and artifact hashes; and
- ClearML receives aggregate evidence only.

This notebook prepares data. It does not select proposal settings, evaluate SAM, or calculate anomaly rankings.
"""
    ),
    code(
        """
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True

from IPython.display import HTML, display
from PIL import Image
"""
    ),
    code(
        """
NOTEBOOK_RELATIVE_PATH = Path(
    "data-preparation/toysad8k-ds1-opaque-package/"
    "03.1-toysad8k-ds1-opaque-package.ipynb"
)
NOTEBOOK_NAME = NOTEBOOK_RELATIVE_PATH.name


def find_code_root(start: Path) -> Path:
    # Search the local checkout and attached Kaggle code snapshots.
    candidates = []
    configured = os.getenv("VPI_CODE_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((start, *start.parents))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for mounted in sorted(kaggle_input.iterdir()):
            candidates.extend((mounted, mounted / "code"))
        for marker in kaggle_input.rglob("cohort_contract.py"):
            if marker.parts[-3:] == (
                "visual_peer_inspector",
                "data",
                "cohort_contract.py",
            ):
                candidates.append(marker.parents[3])
    for candidate in candidates:
        marker = (
            candidate
            / "src"
            / "visual_peer_inspector"
            / "data"
            / "cohort_contract.py"
        )
        if marker.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the code snapshot. Set VPI_CODE_ROOT to the repository folder."
    )


def choose_directory(
    variable: str,
    candidates: list[Path],
    marker: str,
) -> Path:
    configured = os.getenv(variable)
    ordered = [Path(configured).expanduser()] if configured else []
    ordered.extend(candidates)
    for candidate in ordered:
        if candidate.is_dir() and (candidate / marker).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate {variable}. Set it to a folder containing {marker}."
    )


def find_published_ds1_package() -> Path:
    configured = os.getenv("VPI_MODEL_INPUT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path.parent for path in kaggle_input.rglob("package_record.json"))
    for candidate in candidates:
        record_path = candidate / "package_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("cohort_id") == "toysad8k-development-screen-100-v1":
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the published DS1 model-input package.")


def read_provenance(code_root: Path) -> dict:
    notebook_path = code_root / "notebooks" / NOTEBOOK_RELATIVE_PATH
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Canonical notebook source is missing: {NOTEBOOK_NAME}")
    snapshot_candidates = [
        Path(os.environ["VPI_SNAPSHOT_RECORD"]).expanduser()
        if os.getenv("VPI_SNAPSHOT_RECORD")
        else None,
        code_root / "snapshot_record.json",
    ]
    snapshot_path = next(
        (path for path in snapshot_candidates if path is not None and path.is_file()),
        None,
    )
    if snapshot_path is not None:
        from visual_peer_inspector.verification import verify_code_snapshot_record

        snapshot = verify_code_snapshot_record(code_root, snapshot_path)
        return {
            "git_commit": snapshot["git_commit"],
            "git_dirty": bool(snapshot["git_dirty"]),
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "precommit_candidate": bool(snapshot.get("precommit_candidate", False)),
            "platform_versions": dict(snapshot.get("platform_versions", {})),
            "notebook_sha256": sha256(notebook_path.read_bytes()).hexdigest(),
        }
    repository_root = code_root
    git_commit = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            text=True,
        ).strip()
    )
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "snapshot_sha256": "local-working-tree",
        "precommit_candidate": git_dirty,
        "platform_versions": {},
        "notebook_sha256": sha256(notebook_path.read_bytes()).hexdigest(),
    }


def display_rows(rows: list[dict], columns: list[str]) -> None:
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    display(HTML(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"))
"""
    ),
    markdown(
        """
## Runtime and inputs

Local mode builds from trusted preparation records and the private ToysAD source. Kaggle mode receives only the code snapshot and a separately published opaque DS1 package. The full source, private crosswalk, labels, and protocol vaults are not mounted in the Kaggle verification job.
"""
    ),
    code(
        """
EXECUTION_ENVIRONMENT = os.getenv(
    "VPI_EXECUTION_ENVIRONMENT",
    "kaggle" if Path("/kaggle/working").is_dir() else "local",
).lower()
if EXECUTION_ENVIRONMENT not in {"local", "kaggle"}:
    raise ValueError("VPI_EXECUTION_ENVIRONMENT must be local or kaggle.")

CODE_ROOT = find_code_root(Path.cwd().resolve())
sys.path.insert(0, str(CODE_ROOT / "src"))
if os.getenv("VPI_INSTALL_REQUIREMENTS", "1" if EXECUTION_ENVIRONMENT == "kaggle" else "0") == "1":
    requirement = CODE_ROOT / "requirements" / "opaque-package-lock.txt"
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirement)]
    )

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    build_cohort_rgb_package,
    collect_model_input_evidence,
    file_sha256,
    verify_model_input_package,
)
from visual_peer_inspector.evidence import (
    save_bar_chart,
    save_histogram,
    save_image_grid,
    write_csv_rows,
)
if EXECUTION_ENVIRONMENT == "kaggle":
    PUBLISHED_INPUT_ROOT = find_published_ds1_package()
    PACKAGE_OPERATION = "verify_published_package"
else:
    shared_state_candidates = [
        CODE_ROOT / ".local" / "toysad",
        CODE_ROOT.parents[1] / "toysad" if len(CODE_ROOT.parents) > 1 else CODE_ROOT,
    ]
    selection_candidates = [
        root / "selection" / "toysad8k-train-2000-v2"
        for root in shared_state_candidates
    ]
    SELECTION_ROOT = choose_directory(
        "VPI_SELECTION_ROOT",
        selection_candidates,
        "neutral_corpus_manifest.jsonl",
    )
    DEVELOPMENT_ROOT = choose_directory(
        "VPI_DEVELOPMENT_ROOT",
        [path.with_name(path.name + "-development") for path in selection_candidates],
        "development_source_map.jsonl",
    )
    ADMINISTRATIVE_ROOT = choose_directory(
        "VPI_ADMINISTRATIVE_ROOT",
        [path.with_name(path.name + "-administrative-vault") for path in selection_candidates],
        "required_source_members.jsonl",
    )
    SOURCE_ROOT = choose_directory(
        "VPI_SOURCE_ROOT",
        [root / "kaggle" / "toysad8k-full-private-v1" for root in shared_state_candidates],
        "package-checksums.sha256",
    )
    PACKAGE_OPERATION = "build_and_verify"

default_output = (
    Path("/kaggle/working/toysad8k_ds1_opaque_package")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else CODE_ROOT / ".local" / "runs" / "toysad8k-ds1-opaque-package"
)
OUTPUT_ROOT = Path(os.getenv("VPI_OUTPUT_ROOT", str(default_output))).expanduser().resolve()
REQUIRE_FRESH_OUTPUT = os.getenv(
    "VPI_REQUIRE_FRESH_OUTPUT",
    "1" if EXECUTION_ENVIRONMENT == "kaggle" else "0",
) == "1"
PROVENANCE = read_provenance(CODE_ROOT)

print(f"Execution environment: {EXECUTION_ENVIRONMENT}")
print(f"Package operation: {PACKAGE_OPERATION}")
print(f"Cohort: {DS1_SCREEN_CONTRACT.cohort_id}")
print(f"Expected images: {DS1_SCREEN_CONTRACT.image_count}")
print(f"Output folder: {OUTPUT_ROOT}")
print(f"Notebook SHA-256: {PROVENANCE['notebook_sha256']}")
"""
    ),
    markdown(
        """
## Build, verify, and resume

The first local pass reads and verifies every selected source image. The second pass must reuse all 2,000 files and return the same package audit. Kaggle verifies the published package, copies it into the run output, and verifies the copied bytes again.
"""
    ),
    code(
        """
if REQUIRE_FRESH_OUTPUT and OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
    raise RuntimeError("A fresh acceptance run requires an empty output folder.")

if EXECUTION_ENVIRONMENT == "kaggle":
    first_started = time.perf_counter()
    first_audit = verify_model_input_package(
        PUBLISHED_INPUT_ROOT,
        contract=DS1_SCREEN_CONTRACT,
    )
    first_elapsed_seconds = time.perf_counter() - first_started
    shutil.copytree(PUBLISHED_INPUT_ROOT, OUTPUT_ROOT / "model-input")
    second_started = time.perf_counter()
    second_audit = verify_model_input_package(
        OUTPUT_ROOT / "model-input",
        contract=DS1_SCREEN_CONTRACT,
    )
    second_elapsed_seconds = time.perf_counter() - second_started
    first_selection_audit = {
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "membership_sha256": first_audit.membership_sha256,
        "scene_count": first_audit.scene_count,
        "asset_group_count": DS1_SCREEN_CONTRACT.asset_group_count,
        "all_checks_passed": True,
        "files_written_this_run": first_audit.image_count,
        "files_reused_this_run": 0,
        "trusted_input_audit": {
            "source_access_mode": "published_opaque_package",
            "source_manifest_sha256": first_audit.manifest_sha256,
            "source_content_checksums_sha256": first_audit.content_checksums_sha256,
            "selected_source_members_verified": first_audit.verified_image_count,
            "all_checks_passed": True,
        },
    }
    second_selection_audit = {
        "files_written_this_run": 0,
        "files_reused_this_run": second_audit.image_count,
    }
else:
    first_started = time.perf_counter()
    first_audit, first_selection_audit = build_cohort_rgb_package(
        selection_root=SELECTION_ROOT,
        development_root=DEVELOPMENT_ROOT,
        administrative_root=ADMINISTRATIVE_ROOT,
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        contract=DS1_SCREEN_CONTRACT,
        verify_archive_bytes=True,
    )
    first_elapsed_seconds = time.perf_counter() - first_started
    second_started = time.perf_counter()
    second_audit, second_selection_audit = build_cohort_rgb_package(
        selection_root=SELECTION_ROOT,
        development_root=DEVELOPMENT_ROOT,
        administrative_root=ADMINISTRATIVE_ROOT,
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        contract=DS1_SCREEN_CONTRACT,
        verify_archive_bytes=False,
    )
    second_elapsed_seconds = time.perf_counter() - second_started

verified_audit = verify_model_input_package(
    OUTPUT_ROOT / "model-input",
    contract=DS1_SCREEN_CONTRACT,
)
if first_audit.to_dict() != second_audit.to_dict():
    raise ValueError("Fresh and resume package audits differ.")
if second_selection_audit["files_reused_this_run"] != DS1_SCREEN_CONTRACT.image_count:
    raise ValueError("The second pass did not reuse every image.")
if second_selection_audit["files_written_this_run"] != 0:
    raise ValueError("The second pass unexpectedly wrote image files.")
if REQUIRE_FRESH_OUTPUT and first_selection_audit["files_written_this_run"] != DS1_SCREEN_CONTRACT.image_count:
    raise ValueError("The fresh pass did not write every image.")

resume_summary = {
    "first_files_written": first_selection_audit["files_written_this_run"],
    "first_files_reused": first_selection_audit["files_reused_this_run"],
    "second_files_written": second_selection_audit["files_written_this_run"],
    "second_files_reused": second_selection_audit["files_reused_this_run"],
    "first_elapsed_seconds": round(first_elapsed_seconds, 4),
    "second_elapsed_seconds": round(second_elapsed_seconds, 4),
    "hashes_identical": first_audit.to_dict() == second_audit.to_dict(),
    "package_operation": PACKAGE_OPERATION,
}
display_rows(
    [{"check": key, "value": value} for key, value in resume_summary.items()],
    ["check", "value"],
)
"""
    ),
    markdown(
        """
## Package evidence

The tables check scene and view completeness. The charts describe aggregate coverage and media size. Private scene rows and image examples are written to a separate private folder and are not displayed in the canonical notebook.
"""
    ),
    code(
        """
SAFE_ROOT = OUTPUT_ROOT / "safe"
PRIVATE_ROOT = OUTPUT_ROOT / "private"
SAFE_ROOT.mkdir(parents=True, exist_ok=True)
PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
evidence = collect_model_input_evidence(
    OUTPUT_ROOT / "model-input",
    contract=DS1_SCREEN_CONTRACT,
)

gate_columns = ["gate", "status", "observed", "expected"]
view_columns = ["view_id", "scene_count", "expected_scene_count", "complete"]
scene_columns = [
    "scene_index",
    "opaque_scene_id",
    "view_count",
    "expected_view_count",
    "complete",
]
write_csv_rows(SAFE_ROOT / "validation_gates.csv", gate_columns, evidence["gate_rows"])
write_csv_rows(SAFE_ROOT / "view_coverage.csv", view_columns, evidence["view_rows"])
write_csv_rows(
    PRIVATE_ROOT / "scene_completeness.csv",
    scene_columns,
    evidence["scene_rows"],
)
(SAFE_ROOT / "media_summary.json").write_text(
    json.dumps(evidence["media_summary"], indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)

view_chart = SAFE_ROOT / "view_coverage.png"
size_chart = SAFE_ROOT / "image_size_distribution.png"
contact_sheet = PRIVATE_ROOT / "model_input_contact_sheet.png"
save_bar_chart(
    [row["view_id"] for row in evidence["view_rows"]],
    [row["scene_count"] for row in evidence["view_rows"]],
    view_chart,
    title="DS1 model-input view coverage",
    y_label="Scenes",
)
save_histogram(
    evidence["media_sizes_kib"],
    size_chart,
    title="DS1 model-input image size distribution",
    x_label="Image size (KiB)",
)
save_image_grid(
    evidence["sample_items"],
    contact_sheet,
    title="Private DS1 model-input examples",
    columns=5,
)

display_rows(evidence["gate_rows"], gate_columns)
display_rows(evidence["view_rows"], view_columns)
display_rows(
    [{"measure": key, "value": value} for key, value in evidence["media_summary"].items()],
    ["measure", "value"],
)
display(Image.open(view_chart))
display(Image.open(size_chart))
print("Private image examples were written to the private run folder.")
"""
    ),
    markdown(
        """
## Run record and ClearML

The run record links the cohort, code, notebook, package, resume check, and evidence hashes. Computation and both evidence manifests finish before a bounded finaliser starts. The finaliser uses the official ClearML hosts and retries only transient task-creation connection failures. Its receipt contains sanitized categories and retry counts, never exception text. ClearML receives the strict safe allowlist only. It does not receive RGB images, opaque scene rows, private contact sheets, labels, masks, source records, or local paths.
"""
    ),
    code(
        """
trusted_input_audit = first_selection_audit["trusted_input_audit"]
safe_selection_summary = {
    key: first_selection_audit[key]
    for key in (
        "cohort_id",
        "membership_sha256",
        "scene_count",
        "asset_group_count",
        "all_checks_passed",
    )
}
tracking_configuration = {
    "scene_count": DS1_SCREEN_CONTRACT.scene_count,
    "views_per_scene": DS1_SCREEN_CONTRACT.views_per_scene,
    "expected_image_count": DS1_SCREEN_CONTRACT.image_count,
    "expected_width": 256,
    "expected_height": 256,
    "media_kind": "opaque PNG colour views converted to RGB at model load",
    "fresh_output_required": REQUIRE_FRESH_OUTPUT,
    "package_operation": PACKAGE_OPERATION,
}
configuration_sha256 = sha256(
    json.dumps(
        tracking_configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
run_record = {
    "schema_version": 1,
    "stage": "ds1_opaque_package",
    "status": "completed",
    "tracking_status": "handled_by_bounded_finalizer",
    "execution_environment": EXECUTION_ENVIRONMENT,
    "notebook_name": NOTEBOOK_NAME,
    "provenance": PROVENANCE,
    "kaggle_notebook_slug": "esreekarreddy/toysad-8k-ds1-opaque-package",
    "configuration": tracking_configuration,
    "configuration_sha256": configuration_sha256,
    "trusted_inputs": trusted_input_audit,
    "package": verified_audit.to_dict(),
    "cohort_summary": safe_selection_summary,
    "resume_summary": resume_summary,
    "media_summary": evidence["media_summary"],
    "error_type": None,
}
run_record_path = SAFE_ROOT / "run_record.json"
run_record_path.write_text(
    json.dumps(run_record, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)

def write_strict_manifest(root: Path, expected_files: set[str]) -> dict:
    manifest_path = root / "artifact_manifest.json"
    manifest_path.unlink(missing_ok=True)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected_files:
        raise ValueError(
            f"Artifact allowlist mismatch for {root.name}: "
            f"missing={sorted(expected_files-actual)}, unexpected={sorted(actual-expected_files)}"
        )
    rows = [
        {
            "path": relative,
            "sha256": file_sha256(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(expected_files)
    ]
    manifest = {
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    return manifest


safe_manifest = write_strict_manifest(
    SAFE_ROOT,
    {
        "validation_gates.csv",
        "view_coverage.csv",
        "media_summary.json",
        "view_coverage.png",
        "image_size_distribution.png",
        "run_record.json",
    },
)

tracking_mode = os.getenv(
    "VPI_TRACKING_MODE",
    "online" if EXECUTION_ENVIRONMENT == "kaggle" else "disabled",
).strip().lower()
if tracking_mode not in {"online", "offline", "disabled"}:
    raise ValueError("VPI_TRACKING_MODE must be online, offline, or disabled.")
receipt_path = PRIVATE_ROOT / "clearml_receipt.json"
receipt_path.unlink(missing_ok=True)
finalizer = CODE_ROOT / "scripts/finalize_ds1_opaque_package_clearml.py"
command = [
    sys.executable,
    str(finalizer),
    "--safe-root",
    str(SAFE_ROOT),
    "--receipt",
    str(receipt_path),
    "--mode",
    tracking_mode,
    "--wall-time-seconds",
    "180",
]
for tag in ("candidate", "notebook-03-1", EXECUTION_ENVIRONMENT, "ds1"):
    command.extend(("--tag", tag))

process = subprocess.Popen(
    command,
    cwd=CODE_ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True,
)
try:
    finalizer_stdout, finalizer_stderr = process.communicate(timeout=210)
    finalizer_return_code = process.returncode
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
    finalizer_return_code = 124
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "notebook_03_1_clearml_finalisation",
                "mode": tracking_mode,
                "status": "timed_out",
                "task_id": None,
            },
            indent=2,
            sort_keys=True,
        ) + "\\n",
        encoding="utf-8",
    )

if not receipt_path.is_file():
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "notebook_03_1_clearml_finalisation",
                "mode": tracking_mode,
                "status": "failed",
                "task_id": None,
                "error_type": "MissingReceipt",
            },
            indent=2,
            sort_keys=True,
        ) + "\\n",
        encoding="utf-8",
    )
tracking_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
private_manifest = write_strict_manifest(
    PRIVATE_ROOT,
    {
        "scene_completeness.csv",
        "model_input_contact_sheet.png",
        "clearml_receipt.json",
    },
)
if {path.name for path in OUTPUT_ROOT.iterdir()} != {"model-input", "safe", "private"}:
    raise ValueError("Output root must contain only model-input, safe, and private folders.")

print(f"Model input: {OUTPUT_ROOT / 'model-input'}")
print(f"GitHub-safe evidence: {SAFE_ROOT}")
print(f"Private review evidence: {PRIVATE_ROOT}")
print(f"Safe artifacts: {safe_manifest['artifact_count']}")
print(f"Private artifacts: {private_manifest['artifact_count']}")
print(f"ClearML finalisation: {tracking_receipt['status']}")
if finalizer_return_code != 0:
    raise RuntimeError(
        "ClearML finalisation failed after computation. The verified package and evidence remain available."
    )
"""
    ),
    markdown(
        """
## Handoff to the next stages

After the complete output is downloaded and verified, publish only `model-input` as a private, versioned Kaggle dataset. `safe` contains the GitHub evidence candidates. `private` and the executed Kaggle notebook remain review-only. Notebook 04.1 may then generate DS1 proposals from the opaque RGB package.
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
        "language_info": {"name": "python", "version": "3"},
    },
)
for index, cell in enumerate(notebook.cells, 1):
    cell["id"] = f"cell-{index:02d}"

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
