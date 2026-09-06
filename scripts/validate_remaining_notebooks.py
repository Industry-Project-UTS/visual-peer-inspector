#!/usr/bin/env python3
"""Apply shared static checks to the remaining research notebooks."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "08d": ("remaining 600 development", "calibration", "internal-lock"),
    "08e": ("frozen development", "float16", "reference-only"),
    "08e1": ("development identity", "reference", "evaluator"),
    "08f": ("cross-view identity", "application output contract", "unmatched"),
    "08f1": ("calibration resource", "model resource", "evaluator resource"),
    "09a": ("calibration", "prediction", "evaluator"),
    "09b": ("risk", "coverage", "review"),
    "09b1": ("internal-lock resource", "method", "policy"),
    "09c": ("internal-lock", "prediction", "policy"),
    "09d": ("internal-lock", "one", "evaluation"),
    "10r": ("official", "seen", "unseen"),
    "10a": ("official", "seen", "prediction"),
    "10a1": ("official", "unseen", "prediction"),
    "10af": ("official", "seen", "unseen"),
    "10b": ("official", "seen", "unseen"),
    "11r": ("full-pool", "post-freeze", "evaluator"),
    "11a": ("full-pool", "frozen", "prediction"),
    "11b": ("full-pool", "sensitivity", "official"),
}
EXPECTED_KERNEL_SOURCES = {
    "08d": (),
    "08e": ("esreekarreddy/toysad-8k-ds2-frozen-pipeline",),
    "08e1": ("esreekarreddy/toysad-8k-frozen-development-pipeline",),
    "08f": ("esreekarreddy/toysad-8k-frozen-development-pipeline",),
    "08f1": (),
    "09a": (),
    "09b": (
        "esreekarreddy/09a-toysad8k-protected-calibration-inference",
    ),
    "09b1": (),
    "09c": (),
    "09d": (
        "esreekarreddy/09c-toysad8k-protected-internal-lock-inference",
    ),
    "10r": (),
    "10a": (),
    "10a1": (),
    "10af": (),
    "10b": (),
    "11r": ("esreekarreddy/10b-toysad8k-official-seen-unseen-evaluation",),
    "11a": (),
    "11b": (),
}
EXPECTED_DATASET_SOURCES = {
    "08d": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/toysad8k-full-private-eda",
        "esreekarreddy/vpi-toysad-selection-private",
    ),
    "08e": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-ds1-screen100-model-input",
        "esreekarreddy/vpi-toysad-development-remainder-model-input",
        "esreekarreddy/vpi-toysad-ds2-confirmation-rgb",
    ),
    "08e1": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/toysad8k-full-private-eda",
        "esreekarreddy/vpi-toysad-selection-private",
    ),
    "08f": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-ds1-screen100-model-input",
        "esreekarreddy/vpi-toysad-development-remainder-model-input",
        "esreekarreddy/vpi-toysad-ds2-confirmation-rgb",
        "esreekarreddy/vpi-toysad-development-identity-evaluator",
    ),
    "08f1": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/toysad8k-full-private-eda",
        "esreekarreddy/vpi-toysad-selection-private",
        "esreekarreddy/vpi-toysad-development-method-freeze",
    ),
    "09a": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-calibration600-model-input",
        "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
        "esreekarreddy/vpi-toysad-development-method-freeze",
    ),
    "09b": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-calibration600-model-input",
        "esreekarreddy/vpi-toysad-calibration600-evaluator",
        "esreekarreddy/vpi-toysad-development-method-freeze",
    ),
    "09b1": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/toysad8k-full-private-eda",
        "esreekarreddy/vpi-toysad-selection-private",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
    ),
    "09c": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-internal-lock400-model-input",
        "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
    ),
    "09d": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-internal-lock400-evaluator",
        "esreekarreddy/vpi-toysad-internal-lock400-model-input",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
    ),
    "10r": (
        "esreekarreddy/toysad8k-full-private-v1",
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-internal-lock-decision",
    ),
    "10a": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-official-seen-model-input",
        "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
        "esreekarreddy/vpi-toysad-internal-lock-decision",
    ),
    "10a1": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-official-unseen-model-input",
        "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
        "esreekarreddy/vpi-toysad-internal-lock-decision",
    ),
    "10af": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-official-seen-prediction",
        "esreekarreddy/vpi-toysad-official-unseen-prediction",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
        "esreekarreddy/vpi-toysad-internal-lock-decision",
    ),
    "10b": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-official-seen-evaluator",
        "esreekarreddy/vpi-toysad-official-unseen-evaluator",
        "esreekarreddy/vpi-toysad-official-seen-prediction",
        "esreekarreddy/vpi-toysad-official-unseen-prediction",
        "esreekarreddy/vpi-toysad-official-prediction-summary",
        "esreekarreddy/vpi-toysad-official-seen-model-input",
        "esreekarreddy/vpi-toysad-official-unseen-model-input",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
        "esreekarreddy/vpi-toysad-internal-lock-decision",
    ),
    "11r": (
        "esreekarreddy/toysad8k-full-private-v1",
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-selection-audit-inputs",
    ),
    "11a": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-full-pool-model-input",
        "esreekarreddy/vpi-toysad-full-pool-release-receipt",
        "esreekarreddy/vpi-toysad-ds1-locateanything-prompt-freeze",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
    ),
    "11b": (
        "esreekarreddy/vpi-code-snapshot",
        "esreekarreddy/vpi-toysad-full-pool-predictions",
        "esreekarreddy/vpi-toysad-full-pool-evaluator-only",
        "esreekarreddy/vpi-toysad-full-pool-release-receipt",
        "esreekarreddy/vpi-toysad-full-pool-inference-receipt",
        "esreekarreddy/vpi-toysad-development-method-freeze",
        "esreekarreddy/vpi-toysad-calibration-review-policy",
    ),
}
GPU_STAGES = frozenset({"08e", "09a", "09c", "10a", "10a1", "11a"})
LONG_STAGE_TOPICS = {
    "08e": ("checkpoint", "incremental", "recovery"),
    "08e1": ("checkpoint", "incremental", "recovery"),
    "08f1": ("checkpoint", "incremental", "recovery"),
    "09a": ("checkpoint", "incremental", "recovery"),
    "09c": ("checkpoint", "incremental", "recovery"),
    "09b1": ("checkpoint", "incremental", "recovery"),
    "10r": ("checkpoint", "incremental", "recovery"),
    "10a": ("checkpoint", "incremental", "recovery"),
    "10a1": ("checkpoint", "incremental", "recovery"),
    "11r": ("checkpoint", "incremental", "recovery"),
    "11a": ("checkpoint", "incremental", "recovery"),
}
METADATA_KEYS = {"10af": "10a-finalize"}
PROHIBITED_PROSE = (
    "ai-generated",
    "as an ai",
    "claude",
    "codex",
    "august 23",
    "what was removed",
    "what next",
)
SECRET_ASSIGNMENT = re.compile(
    r"CLEARML_API_(?:ACCESS|SECRET)_KEY\s*=\s*['\"][^'\"]+['\"]"
)
LEGACY_GATE = re.compile(
    r"(?:minimum|target|required|acceptance|gate)[_\s-]*(?:recall|precision|accuracy)?"
    r"[^\n]{0,40}(?:0\.95|95\s*%)",
    flags=re.IGNORECASE,
)


def notebook_paths(root: Path) -> dict[str, Path]:
    """Return the single clean source notebook for each remaining stage."""

    found: dict[str, Path] = {}
    for path in sorted((root / "notebooks").rglob("*.ipynb")):
        if "executed" in path.stem.lower():
            continue
        if path.name.startswith("08e1-"):
            stage = "08e1"
        elif path.name.startswith("08f1-"):
            stage = "08f1"
        elif path.name.startswith("09b1-"):
            stage = "09b1"
        elif path.name.startswith("10a-finalize-"):
            stage = "10af"
        elif path.name.startswith("10a1-"):
            stage = "10a1"
        elif path.name.startswith("11r-"):
            stage = "11r"
        elif path.name.startswith("11a-"):
            stage = "11a"
        elif path.name.startswith("11b-"):
            stage = "11b"
        else:
            stage = path.name[:3].lower()
        if stage in EXPECTED:
            if stage in found:
                raise ValueError(f"More than one clean source notebook found for {stage}")
            found[stage] = path
    return found


def cell_text(cells: Iterable[dict], cell_type: str | None = None) -> str:
    selected = (
        cell for cell in cells if cell_type is None or cell.get("cell_type") == cell_type
    )
    return "\n".join("".join(cell.get("source", [])) for cell in selected)


def compile_code_cells(path: Path, cells: list[dict]) -> list[str]:
    errors: list[str] = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source, filename=f"{path.name}:cell-{index}")
        except SyntaxError as error:
            errors.append(f"cell {index} does not parse: {error.msg}")
    return errors


def validate_notebook(stage: str, path: Path) -> list[str]:
    """Return all static validation errors for one notebook."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    errors: list[str] = []
    if payload.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    if not cells or cells[0].get("cell_type") != "markdown":
        errors.append("first cell must be markdown")

    all_text = cell_text(cells)
    markdown = cell_text(cells, "markdown")
    code = cell_text(cells, "code")
    lowered = markdown.lower()
    if path.name not in markdown:
        errors.append("opening markdown must state the exact notebook filename")
    for phrase in EXPECTED[stage]:
        if phrase not in lowered:
            errors.append(f"required topic is absent: {phrase}")
    for phrase in PROHIBITED_PROSE:
        if phrase in lowered:
            errors.append(f"prohibited process or tool prose found: {phrase}")
    if "—" in all_text or "–" in all_text:
        errors.append("use ASCII punctuation instead of em or en dashes")
    if SECRET_ASSIGNMENT.search(code):
        errors.append("a ClearML credential appears to be hard-coded")
    if LEGACY_GATE.search(code):
        errors.append("legacy 0.95-style acceptance logic appears in code")
    provenance_call = "verify_notebook_code_snapshot("
    if code.count(provenance_call) != 1:
        errors.append("notebook must invoke the shared code snapshot verifier exactly once")
    else:
        code_root_index = code.find("CODE_ROOT = find_code_root()")
        provenance_index = code.find(provenance_call)
        if code_root_index < 0 or provenance_index < code_root_index:
            errors.append("code snapshot verification must follow CODE_ROOT resolution")
        intervening = code[code_root_index:provenance_index]
        if "find_directory(" in intervening or "find_named_file(" in intervening:
            errors.append("code snapshot verification must precede other input resolution")
    if "CODE_PROVENANCE" not in code:
        errors.append("verified code provenance is not retained by the notebook")
    if "clearml" not in all_text.lower():
        errors.append("ClearML evidence handling is not described")
    if "hash" not in all_text.lower():
        errors.append("hash or receipt validation is not described")
    if not any(word in all_text.lower() for word in ("failure", "error", "reject")):
        errors.append("failure handling is not described")
    for topic in LONG_STAGE_TOPICS.get(stage, ()):
        if topic not in all_text.lower():
            errors.append(f"long stage does not describe {topic} handling")

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"cell {index} has an execution count")
        if cell.get("outputs"):
            errors.append(f"cell {index} contains saved output")
    errors.extend(compile_code_cells(path, cells))
    return errors


def validate_metadata(stage: str, path: Path) -> list[str]:
    """Validate one Kaggle metadata file and its exact upstream kernels."""

    metadata_key = METADATA_KEYS.get(stage, stage)
    metadata_path = path.with_name(f"kernel-metadata-{metadata_key}.json")
    if not metadata_path.is_file():
        return [f"metadata file is missing: {metadata_path.name}"]
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if value.get("code_file") != path.name:
        errors.append("metadata code_file does not match the notebook filename")
    if value.get("kernel_type") != "notebook" or value.get("language") != "python":
        errors.append("metadata must describe a Python notebook kernel")
    if value.get("is_private") is not True:
        errors.append("remaining research notebooks must stay private on Kaggle")
    if bool(value.get("enable_gpu")) != (stage in GPU_STAGES):
        errors.append("metadata GPU setting does not match the stage contract")
    observed = tuple(value.get("kernel_sources", []))
    if observed != EXPECTED_KERNEL_SOURCES[stage]:
        errors.append(
            "metadata kernel_sources differ from the exact dependency contract: "
            f"expected {EXPECTED_KERNEL_SOURCES[stage]}, observed {observed}"
        )
    expected_datasets = EXPECTED_DATASET_SOURCES.get(stage)
    if expected_datasets is not None:
        observed_datasets = tuple(value.get("dataset_sources", []))
        if observed_datasets != expected_datasets:
            errors.append(
                "metadata dataset_sources differ from the exact dependency contract: "
                f"expected {expected_datasets}, observed {observed_datasets}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-missing", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    found = notebook_paths(root)
    missing = sorted(set(EXPECTED).difference(found))
    failures: list[str] = []
    if missing and not arguments.allow_missing:
        failures.append(f"missing notebook stages: {', '.join(missing)}")
    for stage, path in sorted(found.items()):
        for error in validate_notebook(stage, path):
            failures.append(f"{path.relative_to(root)}: {error}")
        for error in validate_metadata(stage, path):
            failures.append(f"{path.relative_to(root)}: {error}")
    if failures:
        print("Notebook validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    checked = ", ".join(sorted(found)) or "none"
    print(f"Notebook validation passed for: {checked}")
    if missing:
        print(f"Not yet present: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
