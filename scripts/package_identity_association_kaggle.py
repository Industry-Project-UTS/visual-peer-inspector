"""Create the local source archive and Kaggle metadata for Notebook 08J."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local" / "kaggle-upload" / "v1"


def main() -> None:
    from build_identity_association_notebook import build

    build()
    source_root = OUTPUT / "source"
    kernel_root = OUTPUT / "kernel"
    source_root.mkdir(parents=True, exist_ok=True)
    kernel_root.mkdir(parents=True, exist_ok=True)
    files = []
    for folder in ("src", "scripts", "requirements", "tests"):
        files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.append(ROOT / "pyproject.toml")
    archive = source_root / "identity-association-source-v1.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(set(files)):
            handle.write(path, path.relative_to(ROOT).as_posix())
    (source_root / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "VPI Identity Association Source",
                "id": "esreekarreddy/vpi-identity-association-source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    notebook_name = "toysad-8k-development-identity-association.ipynb"
    notebook_target = kernel_root / notebook_name
    notebook_target.write_bytes((ROOT / "notebooks" / "08j-development-identity-association.ipynb").read_bytes())
    (kernel_root / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "esreekarreddy/toysad-8k-development-identity-association",
                "title": "ToysAD-8K Development Identity Association",
                "code_file": notebook_name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": True,
                "keywords": ["computer-vision", "identity-association", "evaluation"],
                "dataset_sources": [
                    "esreekarreddy/vpi-identity-association-source",
                    "esreekarreddy/vpi-development-scoring-model-input",
                    "esreekarreddy/vpi-development-scoring-evaluator",
                ],
                "kernel_sources": ["esreekarreddy/toysad-8k-development-scoring-comparison-full"],
                "competition_sources": [],
                "model_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    receipt = {
        "archive": archive.name,
        "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
        "archive_size_bytes": archive.stat().st_size,
        "file_count": len(set(files)),
        "kernel": "esreekarreddy/toysad-8k-development-identity-association",
        "inputs": {
            "source": "esreekarreddy/vpi-identity-association-source",
            "model_input": "esreekarreddy/vpi-development-scoring-model-input",
            "evaluator": "esreekarreddy/vpi-development-scoring-evaluator",
            "scoring_output": "esreekarreddy/toysad-8k-development-scoring-comparison-full",
        },
    }
    (OUTPUT / "package_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
