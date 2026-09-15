"""Build deterministic Kaggle upload folders for Notebook 08K."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / ".local" / "kaggle-upload" / "patch-v1"
SOURCE_UPLOAD = UPLOAD_ROOT / "source"
KERNEL_UPLOAD = UPLOAD_ROOT / "kernel"
ARCHIVE_NAME = "identity-patch-association-source-v1.zip"
ARCHIVE_PREFIX = "identity-patch-association"
KERNEL_SLUG = "toysad-8k-development-patch-correspondence"
SOURCE_DATASET = "esreekarreddy/vpi-identity-patch-association-source"
MODEL_INPUT_DATASET = "esreekarreddy/vpi-development-scoring-model-input"
EVALUATOR_DATASET = "esreekarreddy/vpi-development-scoring-evaluator"
SCORING_KERNEL = "esreekarreddy/toysad-8k-development-scoring-comparison-full"
INCLUDE_ROOTS = (Path("notebooks"), Path("src"), Path("scripts"), Path("tests"), Path("requirements"))
INCLUDE_FILES = (Path("pyproject.toml"),)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store", ".local"}


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> list[Path]:
    files = []
    for relative_root in INCLUDE_ROOTS:
        files.extend(path for path in (ROOT / relative_root).rglob("*") if path.is_file())
    files.extend(ROOT / relative for relative in INCLUDE_FILES)
    selected = []
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix not in {".py", ".txt", ".toml", ".ipynb"}:
            continue
        selected.append(path)
    return selected


def reset_upload_root() -> None:
    if UPLOAD_ROOT.exists():
        shutil.rmtree(UPLOAD_ROOT)
    SOURCE_UPLOAD.mkdir(parents=True)
    KERNEL_UPLOAD.mkdir(parents=True)


def write_archive(files: list[Path]) -> Path:
    archive = SOURCE_UPLOAD / ARCHIVE_NAME
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as handle:
        for path in files:
            relative = Path(ARCHIVE_PREFIX) / path.relative_to(ROOT)
            info = ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes())
    with ZipFile(archive) as handle:
        names = handle.namelist()
        expected = f"{ARCHIVE_PREFIX}/notebooks/08k-development-patch-correspondence.ipynb"
        if expected not in names or len(names) != len(set(names)) or handle.testzip() is not None:
            raise ValueError("The source archive failed verification")
    return archive


def write_metadata(archive: Path, files: list[Path]) -> None:
    dataset_metadata = {
        "title": "VPI Identity Patch Association Source",
        "id": SOURCE_DATASET,
        "licenses": [{"name": "other"}],
    }
    (SOURCE_UPLOAD / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    notebook_target = KERNEL_UPLOAD / f"{KERNEL_SLUG}.ipynb"
    shutil.copy2(ROOT / "notebooks" / "08k-development-patch-correspondence.ipynb", notebook_target)
    kernel_metadata = {
        "id": f"esreekarreddy/{KERNEL_SLUG}",
        "title": "ToysAD-8K Development Patch Correspondence",
        "code_file": notebook_target.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "dinov2", "computer-vision", "identity"],
        "dataset_sources": [SOURCE_DATASET, MODEL_INPUT_DATASET, EVALUATOR_DATASET],
        "kernel_sources": [SCORING_KERNEL],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (KERNEL_UPLOAD / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2) + "\n")
    receipt = {
        "archive": archive.name,
        "archive_sha256": file_digest(archive),
        "archive_size_bytes": archive.stat().st_size,
        "file_count": len(files),
        "files": {path.relative_to(ROOT).as_posix(): file_digest(path) for path in files},
        "kernel": kernel_metadata["id"],
        "inputs": {
            "source": SOURCE_DATASET,
            "model_input": MODEL_INPUT_DATASET,
            "evaluator": EVALUATOR_DATASET,
            "scoring_output": SCORING_KERNEL,
        },
    }
    (UPLOAD_ROOT / "package_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main() -> None:
    reset_upload_root()
    files = source_files()
    if not files:
        raise ValueError("The source package inventory is empty")
    archive = write_archive(files)
    write_metadata(archive, files)
    print(f"Built {archive} with {len(files)} files")
    print(f"SHA-256: {file_digest(archive)}")
    print(f"Kernel upload: {KERNEL_UPLOAD}")


if __name__ == "__main__":
    main()
