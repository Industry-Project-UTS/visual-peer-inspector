"""Build the code-only Kaggle input for the staged DS0 notebooks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_TARGETS = {
    Path(
        "notebooks/detection-and-segmentation/toysad8k-ds0-sam21/"
        "05-toysad8k-ds0-sam21-segmentation.ipynb"
    ): Path("notebooks/05_toysad8k_ds0_sam21_segmentation.ipynb"),
    Path(
        "notebooks/detection-and-segmentation/toysad8k-ds0-mask-evaluation/"
        "06-toysad8k-ds0-mask-evaluation.ipynb"
    ): Path("notebooks/06-toysad8k-ds0-mask-evaluation.ipynb"),
    Path(
        "notebooks/representation-and-scoring/toysad8k-ds0-peer-scores/"
        "07-toysad8k-ds0-representation-and-peer-scores.ipynb"
    ): Path("notebooks/07-toysad8k-ds0-representation-and-peer-scores.ipynb"),
}

SOURCE_ROOTS = (
    Path("src/visual_peer_inspector"),
    Path("scripts"),
    Path("requirements"),
    Path("tests"),
)

ALLOWED_SUFFIXES = {
    Path("src/visual_peer_inspector"): {".py"},
    Path("scripts"): {".py"},
    Path("requirements"): {".txt"},
    Path("tests"): {".py"},
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "__pycache__" not in path.parts
    )


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
            text=True,
        ).strip()
    )
    return commit, dirty


def build_snapshot(
    *,
    output_root: Path,
    allow_dirty: bool,
    code_dataset_version: str,
    notebook_version_05: str,
    notebook_version_06: str,
    notebook_version_07: str,
) -> Path:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Snapshot output already exists and will not be overwritten: {output_root}"
        )

    commit, dirty = _git_state()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "The repository is dirty. Commit the exact source or pass --allow-dirty "
            "to build a candidate-only snapshot."
        )

    payload_root = output_root / "code"
    payload_root.mkdir(parents=True)
    copied: list[Path] = []

    for source_relative in SOURCE_ROOTS:
        source_root = REPOSITORY_ROOT / source_relative
        if not source_root.is_dir():
            raise FileNotFoundError(f"Snapshot source is missing: {source_relative}")
        for source_file in _source_files(
            source_root, ALLOWED_SUFFIXES[source_relative]
        ):
            relative = source_file.relative_to(REPOSITORY_ROOT)
            destination = payload_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            copied.append(relative)

    for source_relative, target_relative in NOTEBOOK_TARGETS.items():
        source = REPOSITORY_ROOT / source_relative
        if not source.is_file():
            raise FileNotFoundError(f"Snapshot notebook is missing: {source_relative}")
        destination = payload_root / target_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(target_relative)

    file_hashes = {
        relative.as_posix(): file_sha256(payload_root / relative)
        for relative in sorted(set(copied), key=lambda item: item.as_posix())
    }
    snapshot_payload = "".join(
        f"{digest}  {name}\n" for name, digest in file_hashes.items()
    ).encode("utf-8")
    snapshot_record = {
        "schema_version": 1,
        "purpose": "Code-only input for Visual Peer Inspector Kaggle notebooks",
        "git_commit": commit,
        "git_dirty": dirty,
        "precommit_candidate": dirty,
        "snapshot_sha256": sha256(snapshot_payload).hexdigest(),
        "file_count": len(file_hashes),
        "files": file_hashes,
        "platform_versions": {
            "kaggle_notebook_version_03": "5",
            "kaggle_notebook_version_04": "13",
            "kaggle_notebook_version_05": str(notebook_version_05),
            "kaggle_notebook_version_06": str(notebook_version_06),
            "kaggle_notebook_version_07": str(notebook_version_07),
            "code_dataset_version": str(code_dataset_version),
            "model_input_dataset_version": "1",
            "prompt_dataset_version": "1",
            "mask_evaluator_dataset_version": "1",
        },
    }
    (payload_root / "snapshot_record.json").write_text(
        json.dumps(snapshot_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    actual_files = {
        path.relative_to(payload_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file()
    }
    expected_files = set(file_hashes) | {"snapshot_record.json"}
    if actual_files != expected_files:
        raise RuntimeError("Snapshot contains undeclared or missing payload files")

    metadata = {
        "title": "Visual Peer Inspector Code Snapshot",
        "id": "esreekarreddy/vpi-code-snapshot",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-dataset-version", required=True)
    parser.add_argument("--notebook-version-05", required=True)
    parser.add_argument("--notebook-version-06", required=True)
    parser.add_argument("--notebook-version-07", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        build_snapshot(
            output_root=arguments.output_root,
            allow_dirty=arguments.allow_dirty,
            code_dataset_version=arguments.code_dataset_version,
            notebook_version_05=arguments.notebook_version_05,
            notebook_version_06=arguments.notebook_version_06,
            notebook_version_07=arguments.notebook_version_07,
        )
    )
