"""Verify publication-safe executed notebooks and their evidence records."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "reports" / "run-evidence"
EXPECTED_PUBLICATION_NOTEBOOKS = 10
EXPECTED_CLEARML_RECEIPTS = 7
SECRET_TEXT = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\bhf_[A-Za-z0-9]{20,}\b|"
    r"\bKGAT_[A-Za-z0-9_-]{20,}\b|"
    r"(?i:(?:api[_-]?(?:secret|access)?[_-]?key|password|token)"
    r"\s*[:=]\s*[\"'][^\"'${}\s]{8,}[\"'])"
)
LOCAL_PATH_TEXT = re.compile(r"/(?:Users|home)/[^\s\"'<>]+")
PROTECTED_OUTPUT_TEXT = re.compile(
    r"\bS\d{6}\b|\bO\d{2}\b|opaque_scene_id|source_scene_id|"
    r"labels_by_object|mask_rle|box_xyxy",
    re.IGNORECASE,
)
PRIVATE_VISUAL_SOURCE = re.compile(
    r"\bprivate\b|first_scene|prepare_masked_crop", re.IGNORECASE
)
VISUAL_MIME_PREFIXES = ("image/", "application/vnd.plotly")


class EvidenceVerificationError(ValueError):
    """Raised when repository evidence is incomplete or unsafe."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_text(output: Mapping[str, Any]) -> str:
    values: list[str] = []
    text_value = output.get("text")
    if isinstance(text_value, list):
        values.extend(str(item) for item in text_value)
    elif text_value is not None:
        values.append(str(text_value))
    for mime_type, value in output.get("data", {}).items():
        if not str(mime_type).startswith("text/"):
            continue
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return "".join(values)


def _verify_publication_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    publication = notebook.get("metadata", {}).get(
        "visual_peer_inspector_publication"
    )
    if not isinstance(publication, dict):
        raise EvidenceVerificationError(f"Publication metadata is missing: {path}")

    clean_path = path.with_name(path.name.replace("-executed.ipynb", ".ipynb"))
    if not clean_path.is_file():
        raise EvidenceVerificationError(f"Clean notebook is missing: {clean_path}")
    if file_sha256(clean_path) != publication.get("clean_notebook_sha256"):
        raise EvidenceVerificationError(f"Clean notebook hash differs: {path}")

    code_cells = [
        cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    ]
    if not code_cells or any(cell.get("execution_count") is None for cell in code_cells):
        raise EvidenceVerificationError(f"Notebook is not fully executed: {path}")

    for cell in code_cells:
        source = "".join(cell.get("source", []))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise EvidenceVerificationError(f"Error output found: {path}")
            text = _output_text(output)
            if (
                SECRET_TEXT.search(text)
                or LOCAL_PATH_TEXT.search(text)
                or PROTECTED_OUTPUT_TEXT.search(text)
            ):
                raise EvidenceVerificationError(f"Unsafe output text found: {path}")
            if PRIVATE_VISUAL_SOURCE.search(source):
                mime_types = output.get("data", {})
                if any(
                    str(mime_type).startswith(VISUAL_MIME_PREFIXES)
                    for mime_type in mime_types
                ):
                    raise EvidenceVerificationError(
                        f"Private visual output was retained: {path}"
                    )


def _verify_manifest() -> int:
    manifest_path = EVIDENCE_ROOT / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise EvidenceVerificationError("Evidence manifest schema differs")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceVerificationError("Evidence manifest has no files")
    seen: set[str] = set()
    for row in files:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise EvidenceVerificationError("Evidence manifest path is missing or repeated")
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            raise EvidenceVerificationError(f"Manifest file is missing: {relative}")
        if path.stat().st_size != row.get("size_bytes"):
            raise EvidenceVerificationError(f"Manifest size differs: {relative}")
        if file_sha256(path) != row.get("sha256"):
            raise EvidenceVerificationError(f"Manifest hash differs: {relative}")
    return len(files)


def _registered_publications() -> list[Path]:
    # Later experiment sets have their own inventories and verification commands.
    # Keep this check tied to its original evidence set, not a repository-wide count.
    manifest = json.loads((EVIDENCE_ROOT / "evidence_manifest.json").read_text())
    relative_paths = [
        row["path"] for row in manifest["files"]
        if row["path"].endswith("-executed.ipynb")
    ]
    if len(relative_paths) != len(set(relative_paths)):
        raise EvidenceVerificationError("Publication notebook inventory contains duplicates")
    paths = []
    for relative in relative_paths:
        path = ROOT / relative
        if not path.resolve().is_relative_to((ROOT / "notebooks").resolve()):
            raise EvidenceVerificationError("Publication notebook is outside the notebook directory")
        paths.append(path)
    return sorted(paths)


def verify() -> dict[str, int]:
    publication_notebooks = _registered_publications()
    if len(publication_notebooks) != EXPECTED_PUBLICATION_NOTEBOOKS:
        raise EvidenceVerificationError(
            "Publication notebook count differs: "
            f"expected {EXPECTED_PUBLICATION_NOTEBOOKS}, found {len(publication_notebooks)}"
        )
    for path in publication_notebooks:
        _verify_publication_notebook(path)

    logs = sorted((EVIDENCE_ROOT / "logs").glob("*.log"))
    receipts = sorted((EVIDENCE_ROOT / "receipts").glob("*.json"))
    if len(logs) != EXPECTED_PUBLICATION_NOTEBOOKS:
        raise EvidenceVerificationError("Execution transcript count differs")
    if len(receipts) != EXPECTED_PUBLICATION_NOTEBOOKS:
        raise EvidenceVerificationError("Publication receipt count differs")
    for path in logs:
        text = path.read_text(encoding="utf-8")
        if (
            SECRET_TEXT.search(text)
            or LOCAL_PATH_TEXT.search(text)
            or PROTECTED_OUTPUT_TEXT.search(text)
        ):
            raise EvidenceVerificationError(f"Unsafe execution transcript: {path}")

    evidence_manifest = json.loads((EVIDENCE_ROOT / "evidence_manifest.json").read_text())
    clearml_receipts = sorted(
        ROOT / row["path"]
        for row in evidence_manifest["files"]
        if row["path"].startswith("reports/run-evidence/clearml/")
    )
    if len(clearml_receipts) != EXPECTED_CLEARML_RECEIPTS:
        raise EvidenceVerificationError("ClearML receipt count differs")
    for path in clearml_receipts:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("mode") != "online" or receipt.get("status") != "completed":
            raise EvidenceVerificationError(f"ClearML receipt is incomplete: {path}")
        if not re.fullmatch(r"[a-f0-9]{32}", str(receipt.get("task_id", ""))):
            raise EvidenceVerificationError(f"ClearML task ID is invalid: {path}")

    manifest_files = _verify_manifest()
    return {
        "publication_notebooks": len(publication_notebooks),
        "execution_transcripts": len(logs),
        "publication_receipts": len(receipts),
        "clearml_receipts": len(clearml_receipts),
        "manifest_files": manifest_files,
    }


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
