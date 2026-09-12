"""Verify the published development-stage notebooks and artifact inventory."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath

from verify_repository_run_evidence import _verify_publication_notebook

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(part in {".", ".."} for part in parts.parts):
        raise ValueError("Evidence path is outside the repository")
    path = root
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Evidence cannot use symbolic links")
    if not path.is_file():
        raise ValueError(f"Evidence file is missing: {relative}")
    return path


def verify(root: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "reports/development-freeze/evidence_manifest.json").read_text())
    if manifest.get("status") != "verified_complete":
        raise ValueError("The combined development evidence is not complete")
    runs = manifest["runs"]
    if len(runs) != 4 or {row["stage"] for row in runs} != {"08d", "08e", "08e1", "08f"}:
        raise ValueError("The combined development evidence requires four completed stages")
    declared = set()
    for row in manifest["files"]:
        relative = row["path"]
        if relative in declared:
            raise ValueError("The evidence inventory contains a duplicate file")
        declared.add(relative)
        path = contained_file(root, relative)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Evidence checksum differs: {relative}")
    for run in runs:
        if run["clean_notebook"] != run["executed_notebook"].replace("-executed.ipynb", ".ipynb"):
            raise ValueError("An executed notebook is not paired with its exact clean source")
        for field in ("clean_notebook", "executed_notebook", "execution_log"):
            if run[field] not in declared:
                raise ValueError(f"Run evidence is not in the checksum inventory: {field}")
        _verify_publication_notebook(contained_file(root, run["executed_notebook"]))
    return {"status": "verified_complete", "runs": len(runs), "files": len(declared)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
