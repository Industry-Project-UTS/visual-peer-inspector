#!/usr/bin/env python3
"""Build deterministic Kaggle source and kernel upload folders for 08L/08M."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_ROOT = ROOT.parent / "identity-association"
SOURCE_DATASET = "esreekarreddy/vpi-learned-peer-view-source"
SCORING_KERNEL = "esreekarreddy/toysad-8k-development-scoring-comparison-full"
MODEL_INPUT_DATASET = "esreekarreddy/vpi-development-scoring-model-input"
EVALUATOR_DATASET = "esreekarreddy/vpi-development-scoring-evaluator"
FEATURE_KERNEL = "esreekarreddy/toysad-8k-development-learned-peer-features"
EVALUATION_KERNEL = "esreekarreddy/toysad-8k-development-learned-peer-ranker"
ARCHIVE_PREFIX = Path("learned-peer-view-source")
EXCLUDED_PARTS = {".local", ".pytest_cache", "__pycache__", ".DS_Store"}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def copy_source_tree(staging: Path) -> None:
    """Copy the established package, then overlay only this stage's files."""

    shutil.copytree(
        IDENTITY_ROOT / "src",
        staging / "src",
        ignore=shutil.ignore_patterns(*EXCLUDED_PARTS),
    )
    overlay = ROOT / "src/visual_peer_inspector/workflows"
    target = staging / "src/visual_peer_inspector/workflows"
    for name in ("learned_peer_features.py", "learned_peer_evaluation.py"):
        shutil.copy2(overlay / name, target / name)
    for folder in ("scripts", "tests", "requirements", "notebooks"):
        shutil.copytree(
            ROOT / folder,
            staging / folder,
            ignore=shutil.ignore_patterns(*EXCLUDED_PARTS),
        )
    shutil.copy2(IDENTITY_ROOT / "pyproject.toml", staging / "pyproject.toml")
    shutil.copy2(ROOT / "README.md", staging / "README.md")


def staged_files(staging: Path) -> list[Path]:
    allowed_suffixes = {".py", ".txt", ".toml", ".ipynb", ".md"}
    files = [
        path
        for path in staging.rglob("*")
        if path.is_file()
        and path.suffix in allowed_suffixes
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(staging).parts)
    ]
    if not files:
        raise ValueError("The Kaggle source inventory is empty")
    return sorted(files)


def write_source_archive(staging: Path, output: Path) -> tuple[Path, dict[str, str]]:
    files = staged_files(staging)
    inventory = {path.relative_to(staging).as_posix(): digest(path) for path in files}
    archive = output / "learned-peer-view-source-v1.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as handle:
        for path in files:
            relative = ARCHIVE_PREFIX / path.relative_to(staging)
            info = ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes())
        if handle.testzip() is not None:
            raise ValueError("The Kaggle source archive failed its CRC check")
    with ZipFile(archive) as handle:
        names = handle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("The Kaggle source archive contains duplicate members")
        required = {
            f"{ARCHIVE_PREFIX}/scripts/run_metric_feature_preparation.py",
            f"{ARCHIVE_PREFIX}/scripts/run_metric_evaluation.py",
            f"{ARCHIVE_PREFIX}/src/visual_peer_inspector/workflows/learned_peer_features.py",
            f"{ARCHIVE_PREFIX}/src/visual_peer_inspector/workflows/learned_peer_evaluation.py",
        }
        if not required.issubset(names):
            raise ValueError("The Kaggle source archive is incomplete")
    return archive, inventory


def kernel_metadata(*, identifier: str, title: str, code_file: str, datasets: list[str], kernels: list[str]) -> dict[str, object]:
    return {
        "id": identifier,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["computer-vision", "anomaly-ranking", "development-evaluation"],
        "dataset_sources": datasets,
        "kernel_sources": kernels,
        "competition_sources": [],
        "model_sources": [],
    }


def write_kernel(directory: Path, notebook_source: Path, notebook_name: str, metadata: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_source, directory / notebook_name)
    atomic_json(directory / "kernel-metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".local/kaggle-upload/metric-v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    source_upload = output / "source"
    staging = output / "staging"
    source_upload.mkdir(parents=True)
    staging.mkdir(parents=True)

    copy_source_tree(staging)
    archive, inventory = write_source_archive(staging, source_upload)
    atomic_json(
        source_upload / "source-manifest.json",
        {
            "archive": archive.name,
            "archive_sha256": digest(archive),
            "file_count": len(inventory),
            "schema_version": 1,
        },
    )
    atomic_json(
        source_upload / "dataset-metadata.json",
        {
            "title": "VPI Learned Peer and View Source",
            "id": SOURCE_DATASET,
            "licenses": [{"name": "other"}],
        },
    )

    notebook_08l = "toysad-8k-development-learned-peer-features.ipynb"
    notebook_08m = "toysad-8k-development-learned-peer-ranker.ipynb"
    write_kernel(
        output / "kernel-08l",
        ROOT / "notebooks/08l-development-learned-peer-view-features.ipynb",
        notebook_08l,
        kernel_metadata(
            identifier=FEATURE_KERNEL,
            title="ToysAD-8K Development Learned Peer Features",
            code_file=notebook_08l,
            datasets=[SOURCE_DATASET],
            kernels=[SCORING_KERNEL],
        ),
    )
    write_kernel(
        output / "kernel-08m",
        ROOT / "notebooks/08m-development-learned-peer-view-ranker.ipynb",
        notebook_08m,
        kernel_metadata(
            identifier=EVALUATION_KERNEL,
            title="ToysAD-8K Development Learned Peer Ranker",
            code_file=notebook_08m,
            datasets=[SOURCE_DATASET, MODEL_INPUT_DATASET, EVALUATOR_DATASET],
            kernels=[SCORING_KERNEL, FEATURE_KERNEL],
        ),
    )

    receipt = {
        "schema_version": "learned-peer-kaggle-package/v1",
        "source_dataset": SOURCE_DATASET,
        "source_archive": archive.name,
        "source_archive_sha256": digest(archive),
        "source_archive_bytes": archive.stat().st_size,
        "source_file_count": len(inventory),
        "source_inventory": inventory,
        "kernels": [FEATURE_KERNEL, EVALUATION_KERNEL],
        "sequence": ["08L", "08M"],
        "gpu_required": False,
    }
    atomic_json(output / "package_receipt.json", receipt)
    shutil.rmtree(staging)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
