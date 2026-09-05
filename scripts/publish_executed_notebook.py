#!/usr/bin/env python3
"""Create a repository-safe evidence copy of an executed notebook.

The clean notebook remains the runnable source. This tool verifies that a
downloaded saved-output notebook has exactly the same cell sources, keeps safe
aggregate outputs, removes private scene visuals, redacts protected text, and
writes a compact execution transcript for review.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


PRIVATE_VISUAL_MARKERS = (
    '"private"',
    "'private'",
    "private/",
    "first_scene",
    "prepare_masked_crop",
)
PRIVATE_WORD = re.compile(r"\bprivate\b", re.IGNORECASE)
VISUAL_MIME_TYPES = frozenset(
    {
        "application/vnd.plotly.v1+json",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/svg+xml",
    }
)
SECRET_TEXT = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\bhf_[A-Za-z0-9]{20,}\b|"
    r"\bKGAT_[A-Za-z0-9_-]{20,}\b|"
    r"(?i:(?:api[_-]?(?:secret|access)?[_-]?key|password|token)"
    r"\s*[:=]\s*[\"'][^\"'${}\s]{8,}[\"'])"
)
LOCAL_PATH_TEXT = re.compile(r"/(?:Users|home)/[^\s\"'<>]+")
PROTECTED_TEXT = re.compile(
    r"\bS\d{6}\b|\bO\d{2}\b|opaque_scene_id|source_scene_id|"
    r"labels_by_object|mask_rle|box_xyxy",
    re.IGNORECASE,
)


class PublicationError(ValueError):
    """Raised when notebook evidence cannot be published safely."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_signature(notebook: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(cell.get("cell_type", "")), "".join(cell.get("source", [])))
        for cell in notebook.get("cells", [])
    ]


def _replace_text(text: str) -> str:
    value = SECRET_TEXT.sub("[secret removed]", text)
    value = LOCAL_PATH_TEXT.sub("[local path removed]", value)
    value = PROTECTED_TEXT.sub("[protected identifier removed]", value)
    return "\n".join(line.rstrip(" \t") for line in value.split("\n"))


def _sanitize_text_value(value: Any) -> Any:
    if isinstance(value, str):
        return _replace_text(value)
    if isinstance(value, list):
        return [_sanitize_text_value(item) for item in value]
    return value


def _private_visual_cell(source: str) -> bool:
    lowered = source.lower()
    return bool(PRIVATE_WORD.search(source)) or any(
        marker in lowered for marker in PRIVATE_VISUAL_MARKERS
    )


def _sanitize_output(
    output: Mapping[str, Any], *, remove_visuals: bool
) -> tuple[dict[str, Any] | None, int, int]:
    if output.get("output_type") == "error":
        raise PublicationError("The executed notebook contains an error output")

    result = deepcopy(dict(output))
    removed_visuals = 0
    retained_visuals = 0

    if "text" in result:
        result["text"] = _sanitize_text_value(result["text"])

    data = result.get("data")
    if isinstance(data, dict):
        for mime_type in list(data):
            if mime_type in VISUAL_MIME_TYPES:
                if remove_visuals:
                    del data[mime_type]
                    removed_visuals += 1
                else:
                    retained_visuals += 1
            elif mime_type.startswith("text/"):
                data[mime_type] = _sanitize_text_value(data[mime_type])
        if not data:
            return None, removed_visuals, retained_visuals

    return result, removed_visuals, retained_visuals


def _publication_note(
    *, label: str, clean_sha256: str, executed_sha256: str
) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": ["repository-publication-note"]},
        "source": [
            "## Saved-output evidence copy\n",
            "\n",
            "This notebook records the completed run for repository review. "
            "Its code and markdown match the clean runnable notebook. Private "
            "scene visuals and identifier-bearing output were removed. Safe "
            "aggregate tables and charts were retained.\n",
            "\n",
            f"Run stage: `{label}`  \n",
            f"Clean notebook SHA-256: `{clean_sha256}`  \n",
            f"Downloaded saved-output SHA-256: `{executed_sha256}`\n",
        ],
    }


def publish_notebook(
    *,
    clean_path: Path,
    executed_path: Path,
    output_path: Path,
    log_path: Path,
    label: str,
) -> dict[str, Any]:
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    executed = json.loads(executed_path.read_text(encoding="utf-8"))
    if _cell_signature(clean) != _cell_signature(executed):
        raise PublicationError("Clean and executed notebook cell sources differ")

    code_cells = [
        cell for cell in executed.get("cells", []) if cell.get("cell_type") == "code"
    ]
    incomplete = [
        index
        for index, cell in enumerate(code_cells)
        if cell.get("execution_count") is None
    ]
    if incomplete:
        raise PublicationError(f"Unexecuted code cells found: {incomplete}")

    published = deepcopy(executed)
    removed_visuals = 0
    retained_visuals = 0
    output_blocks = 0
    transcript: list[str] = []
    for cell_index, cell in enumerate(published.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        remove_visuals = _private_visual_cell(source)
        safe_outputs: list[dict[str, Any]] = []
        for output in cell.get("outputs", []):
            safe_output, removed, retained = _sanitize_output(
                output, remove_visuals=remove_visuals
            )
            removed_visuals += removed
            retained_visuals += retained
            if safe_output is None:
                continue
            safe_outputs.append(safe_output)
            output_blocks += 1
            text_parts: list[str] = []
            if "text" in safe_output:
                text_value = safe_output["text"]
                text_parts.extend(text_value if isinstance(text_value, list) else [text_value])
            for mime_type, value in safe_output.get("data", {}).items():
                if not mime_type.startswith("text/"):
                    continue
                text_parts.extend(value if isinstance(value, list) else [value])
            text = "".join(str(part) for part in text_parts).strip()
            if text:
                transcript.append(f"[cell {cell_index}]\n{text}\n")
        cell["outputs"] = safe_outputs

    clean_hash = file_sha256(clean_path)
    executed_hash = file_sha256(executed_path)
    published["cells"].insert(
        1,
        _publication_note(
            label=label,
            clean_sha256=clean_hash,
            executed_sha256=executed_hash,
        ),
    )
    metadata = published.setdefault("metadata", {})
    metadata["visual_peer_inspector_publication"] = {
        "schema_version": 1,
        "label": label,
        "clean_notebook_sha256": clean_hash,
        "downloaded_saved_output_sha256": executed_hash,
        "private_visuals_removed": removed_visuals,
        "safe_visuals_retained": retained_visuals,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(published, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_header = [
        f"Stage: {label}",
        "Status: completed",
        f"Clean notebook: {clean_path.as_posix()}",
        f"Clean notebook SHA-256: {clean_hash}",
        f"Downloaded saved-output SHA-256: {executed_hash}",
        f"Executed code cells: {len(code_cells)}",
        f"Retained output blocks: {output_blocks}",
        f"Retained safe visuals: {retained_visuals}",
        f"Removed private visuals: {removed_visuals}",
        "",
        "Sanitised execution transcript",
        "==============================",
        "",
    ]
    log_path.write_text(
        "\n".join(log_header + transcript).rstrip() + "\n",
        encoding="utf-8",
    )

    serialized = output_path.read_text(encoding="utf-8") + log_path.read_text(
        encoding="utf-8"
    )
    if SECRET_TEXT.search(serialized) or LOCAL_PATH_TEXT.search(serialized):
        raise PublicationError("Published evidence still contains a secret or local path")

    return {
        "schema_version": 1,
        "label": label,
        "status": "completed",
        "clean_notebook_sha256": clean_hash,
        "downloaded_saved_output_sha256": executed_hash,
        "published_notebook_sha256": file_sha256(output_path),
        "published_log_sha256": file_sha256(log_path),
        "executed_code_cells": len(code_cells),
        "retained_output_blocks": output_blocks,
        "retained_safe_visuals": retained_visuals,
        "removed_private_visuals": removed_visuals,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--executed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = publish_notebook(
        clean_path=args.clean,
        executed_path=args.executed,
        output_path=args.output,
        log_path=args.log,
        label=args.label,
    )
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
