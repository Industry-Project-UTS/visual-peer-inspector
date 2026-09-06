"""Strict provenance binding for generated Kaggle notebooks and stage runners."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from visual_peer_inspector.verification import verify_code_snapshot_record

ENVIRONMENT_KEY = "VPI_CODE_PROVENANCE_JSON"
PROVENANCE_FIELDS = (
    "snapshot_sha256",
    "git_commit",
    "git_dirty",
    "notebook_sha256",
)


def verify_notebook_code_snapshot(
    code_root: Path,
    notebook_relative_path: str,
    *,
    export_environment: bool = True,
) -> dict[str, Any]:
    """Verify one complete, clean snapshot and bind its notebook identity."""

    root = Path(code_root).expanduser().resolve()
    snapshot = verify_code_snapshot_record(root, root / "snapshot_record.json")
    if snapshot.get("git_dirty") is not False:
        raise RuntimeError("Kaggle stages require a clean code snapshot")
    if snapshot.get("precommit_candidate") is not False:
        raise RuntimeError("Kaggle stages reject pre-commit candidate snapshots")

    git_commit = snapshot.get("git_commit")
    snapshot_sha256 = snapshot.get("snapshot_sha256")
    if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("Code snapshot git commit is not a pinned SHA-1")
    if (
        not isinstance(snapshot_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256) is None
    ):
        raise ValueError("Code snapshot digest is not a SHA-256")

    relative = Path(notebook_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Notebook snapshot path must be safe and relative")
    relative_name = relative.as_posix()
    files = snapshot.get("files")
    notebook_sha256 = files.get(relative_name) if isinstance(files, dict) else None
    if (
        not isinstance(notebook_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", notebook_sha256) is None
    ):
        raise ValueError("Notebook is absent from the verified code snapshot")

    provenance = {
        "snapshot_sha256": snapshot_sha256,
        "git_commit": git_commit,
        "git_dirty": False,
        "notebook_sha256": notebook_sha256,
    }
    if export_environment:
        os.environ[ENVIRONMENT_KEY] = json.dumps(
            provenance, sort_keys=True, separators=(",", ":")
        )
    return provenance


def runtime_code_provenance(
    environment: Mapping[str, str] | None = None,
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Load and validate the notebook-exported provenance for a stage process."""

    source = os.environ if environment is None else environment
    encoded = source.get(ENVIRONMENT_KEY)
    if encoded is None:
        if required:
            raise RuntimeError("Verified notebook code provenance is not bound")
        return {}
    value = json.loads(encoded)
    if not isinstance(value, dict) or set(value) != set(PROVENANCE_FIELDS):
        raise ValueError("Notebook code provenance has an invalid field contract")
    if value.get("git_dirty") is not False:
        raise RuntimeError("Stage process requires clean notebook provenance")
    for name, length in (
        ("snapshot_sha256", 64),
        ("git_commit", 40),
        ("notebook_sha256", 64),
    ):
        if re.fullmatch(rf"[0-9a-f]{{{length}}}", str(value.get(name, ""))) is None:
            raise ValueError(f"Notebook code provenance field is invalid: {name}")
    return {name: value[name] for name in PROVENANCE_FIELDS}


def with_runtime_code_provenance(
    record: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Return a receipt or run record with its safe code provenance fields."""

    return {
        **dict(record),
        **runtime_code_provenance(environment, required=required),
    }


def bind_runtime_code_provenance_to_json(
    path: Path,
    environment: Mapping[str, str] | None = None,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Atomically add runtime code provenance to an existing JSON receipt."""

    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Provenance can be bound only to a JSON object")
    bound = with_runtime_code_provenance(
        value, environment, required=required
    )
    temporary = target.with_suffix(target.suffix + ".provenance.incomplete")
    temporary.write_text(
        json.dumps(bound, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return bound


__all__ = [
    "ENVIRONMENT_KEY",
    "PROVENANCE_FIELDS",
    "bind_runtime_code_provenance_to_json",
    "runtime_code_provenance",
    "verify_notebook_code_snapshot",
    "with_runtime_code_provenance",
]
