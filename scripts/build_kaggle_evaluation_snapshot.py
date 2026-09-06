"""Build the code-only Kaggle input for the staged evaluation notebooks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

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
    Path(
        "notebooks/data-preparation/toysad8k-ds1-opaque-package/"
        "03.1-toysad8k-ds1-opaque-package.ipynb"
    ): Path(
        "notebooks/data-preparation/toysad8k-ds1-opaque-package/"
        "03.1-toysad8k-ds1-opaque-package.ipynb"
    ),
    Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/"
        "04.1-toysad8k-ds1-locateanything-screen.ipynb"
    ): Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/"
        "04.1-toysad8k-ds1-locateanything-screen.ipynb"
    ),
    Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-sam21/"
        "05.1-toysad8k-ds1-sam21-segmentation.ipynb"
    ): Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-sam21/"
        "05.1-toysad8k-ds1-sam21-segmentation.ipynb"
    ),
    Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-mask-evaluation/"
        "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb"
    ): Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-mask-evaluation/"
        "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb"
    ),
    Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-proposal-refinement/"
        "04.2-toysad8k-ds1-proposal-refinement.ipynb"
    ): Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-proposal-refinement/"
        "04.2-toysad8k-ds1-proposal-refinement.ipynb"
    ),
    Path(
        "notebooks/representation-and-scoring/toysad8k-ds1-peer-scores/"
        "07.1-toysad8k-ds1-representation-and-peer-scores.ipynb"
    ): Path(
        "notebooks/representation-and-scoring/toysad8k-ds1-peer-scores/"
        "07.1-toysad8k-ds1-representation-and-peer-scores.ipynb"
    ),
    Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything-prompt-freeze/"
        "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb"
    ): Path(
        "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything-prompt-freeze/"
        "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb"
    ),
    Path(
        "notebooks/confirmation-and-multiview/"
        "08a-toysad8k-ds2-confirmation-release.ipynb"
    ): Path(
        "notebooks/confirmation-and-multiview/"
        "08a-toysad8k-ds2-confirmation-release.ipynb"
    ),
    Path(
        "notebooks/confirmation-and-multiview/"
        "08b-toysad8k-ds2-frozen-pipeline.ipynb"
    ): Path(
        "notebooks/confirmation-and-multiview/"
        "08b-toysad8k-ds2-frozen-pipeline.ipynb"
    ),
    Path(
        "notebooks/confirmation-and-multiview/"
        "08c-toysad8k-ds2-confirmation-and-multiview.ipynb"
    ): Path(
        "notebooks/confirmation-and-multiview/"
        "08c-toysad8k-ds2-confirmation-and-multiview.ipynb"
    ),
    Path(
        "notebooks/development-freeze/"
        "08d-toysad8k-primary-protocol-release.ipynb"
    ): Path(
        "notebooks/development-freeze/"
        "08d-toysad8k-primary-protocol-release.ipynb"
    ),
    Path(
        "notebooks/development-freeze/"
        "08e-toysad8k-frozen-development-pipeline.ipynb"
    ): Path(
        "notebooks/development-freeze/"
        "08e-toysad8k-frozen-development-pipeline.ipynb"
    ),
    Path(
        "notebooks/development-freeze/"
        "08e1-toysad8k-development-identity-evaluator-release.ipynb"
    ): Path(
        "notebooks/development-freeze/"
        "08e1-toysad8k-development-identity-evaluator-release.ipynb"
    ),
    Path(
        "notebooks/development-freeze/"
        "08f-toysad8k-cross-view-identity-freeze.ipynb"
    ): Path(
        "notebooks/development-freeze/"
        "08f-toysad8k-cross-view-identity-freeze.ipynb"
    ),
    Path(
        "notebooks/protected-evaluation/"
        "08f1-toysad8k-calibration-resource-release.ipynb"
    ): Path(
        "notebooks/protected-evaluation/"
        "08f1-toysad8k-calibration-resource-release.ipynb"
    ),
    Path(
        "notebooks/protected-evaluation/"
        "09a-toysad8k-protected-calibration-inference.ipynb"
    ): Path(
        "notebooks/protected-evaluation/"
        "09a-toysad8k-protected-calibration-inference.ipynb"
    ),
    Path(
        "notebooks/protected-evaluation/"
        "09b-toysad8k-calibration-review-policy.ipynb"
    ): Path(
        "notebooks/protected-evaluation/"
        "09b-toysad8k-calibration-review-policy.ipynb"
    ),
    Path(
        "notebooks/protected-evaluation/"
        "09b1-toysad8k-internal-lock-resource-release.ipynb"
    ): Path(
        "notebooks/protected-evaluation/"
        "09b1-toysad8k-internal-lock-resource-release.ipynb"
    ),
    Path(
        "notebooks/protected-evaluation/"
        "09c-toysad8k-protected-internal-lock-inference.ipynb"
    ): Path(
        "notebooks/protected-evaluation/"
        "09c-toysad8k-protected-internal-lock-inference.ipynb"
    ),
    Path(
        "notebooks/protected-evaluation/"
        "09d-toysad8k-internal-lock-one-shot-evaluation.ipynb"
    ): Path(
        "notebooks/protected-evaluation/"
        "09d-toysad8k-internal-lock-one-shot-evaluation.ipynb"
    ),
    Path(
        "notebooks/official-evaluation/"
        "10r-toysad8k-official-release.ipynb"
    ): Path(
        "notebooks/official-evaluation/"
        "10r-toysad8k-official-release.ipynb"
    ),
    Path(
        "notebooks/official-evaluation/"
        "10a-toysad8k-official-seen-frozen-inference.ipynb"
    ): Path(
        "notebooks/official-evaluation/"
        "10a-toysad8k-official-seen-frozen-inference.ipynb"
    ),
    Path(
        "notebooks/official-evaluation/"
        "10a1-toysad8k-official-unseen-frozen-inference.ipynb"
    ): Path(
        "notebooks/official-evaluation/"
        "10a1-toysad8k-official-unseen-frozen-inference.ipynb"
    ),
    Path(
        "notebooks/official-evaluation/"
        "10a-finalize-toysad8k-official-prediction-summary.ipynb"
    ): Path(
        "notebooks/official-evaluation/"
        "10a-finalize-toysad8k-official-prediction-summary.ipynb"
    ),
    Path(
        "notebooks/official-evaluation/"
        "10b-toysad8k-official-seen-unseen-evaluation.ipynb"
    ): Path(
        "notebooks/official-evaluation/"
        "10b-toysad8k-official-seen-unseen-evaluation.ipynb"
    ),
    Path(
        "notebooks/post-freeze-sensitivity/"
        "11r-toysad8k-full-pool-release.ipynb"
    ): Path(
        "notebooks/post-freeze-sensitivity/"
        "11r-toysad8k-full-pool-release.ipynb"
    ),
    Path(
        "notebooks/post-freeze-sensitivity/"
        "11a-toysad8k-full-pool-frozen-inference.ipynb"
    ): Path(
        "notebooks/post-freeze-sensitivity/"
        "11a-toysad8k-full-pool-frozen-inference.ipynb"
    ),
    Path(
        "notebooks/post-freeze-sensitivity/"
        "11b-toysad8k-full-pool-sensitivity-evaluation.ipynb"
    ): Path(
        "notebooks/post-freeze-sensitivity/"
        "11b-toysad8k-full-pool-sensitivity-evaluation.ipynb"
    ),
}

DS1_NOTEBOOK_PATHS = frozenset(
    {
        Path(
            "notebooks/data-preparation/toysad8k-ds1-opaque-package/"
            "03.1-toysad8k-ds1-opaque-package.ipynb"
        ),
        Path(
            "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/"
            "04.1-toysad8k-ds1-locateanything-screen.ipynb"
        ),
        Path(
            "notebooks/detection-and-segmentation/toysad8k-ds1-sam21/"
            "05.1-toysad8k-ds1-sam21-segmentation.ipynb"
        ),
        Path(
            "notebooks/detection-and-segmentation/toysad8k-ds1-mask-evaluation/"
            "06.1-toysad8k-ds1-proposal-and-mask-evaluation.ipynb"
        ),
        Path(
            "notebooks/detection-and-segmentation/toysad8k-ds1-proposal-refinement/"
            "04.2-toysad8k-ds1-proposal-refinement.ipynb"
        ),
        Path(
            "notebooks/representation-and-scoring/toysad8k-ds1-peer-scores/"
            "07.1-toysad8k-ds1-representation-and-peer-scores.ipynb"
        ),
        Path(
            "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything-prompt-freeze/"
            "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb"
        ),
    }
)

DS2_NOTEBOOK_PATHS = frozenset(
    {
        Path(
            "notebooks/confirmation-and-multiview/"
            "08a-toysad8k-ds2-confirmation-release.ipynb"
        ),
        Path(
            "notebooks/confirmation-and-multiview/"
            "08b-toysad8k-ds2-frozen-pipeline.ipynb"
        ),
        Path(
            "notebooks/confirmation-and-multiview/"
            "08c-toysad8k-ds2-confirmation-and-multiview.ipynb"
        ),
    }
)

REMAINING_NOTEBOOK_PATHS = frozenset(
    {
        Path(
            "notebooks/development-freeze/"
            "08d-toysad8k-primary-protocol-release.ipynb"
        ),
        Path(
            "notebooks/development-freeze/"
            "08e-toysad8k-frozen-development-pipeline.ipynb"
        ),
        Path(
            "notebooks/development-freeze/"
            "08e1-toysad8k-development-identity-evaluator-release.ipynb"
        ),
        Path(
            "notebooks/development-freeze/"
            "08f-toysad8k-cross-view-identity-freeze.ipynb"
        ),
        Path(
            "notebooks/protected-evaluation/"
            "08f1-toysad8k-calibration-resource-release.ipynb"
        ),
        Path(
            "notebooks/protected-evaluation/"
            "09a-toysad8k-protected-calibration-inference.ipynb"
        ),
        Path(
            "notebooks/protected-evaluation/"
            "09b-toysad8k-calibration-review-policy.ipynb"
        ),
        Path(
            "notebooks/protected-evaluation/"
            "09b1-toysad8k-internal-lock-resource-release.ipynb"
        ),
        Path(
            "notebooks/protected-evaluation/"
            "09c-toysad8k-protected-internal-lock-inference.ipynb"
        ),
        Path(
            "notebooks/protected-evaluation/"
            "09d-toysad8k-internal-lock-one-shot-evaluation.ipynb"
        ),
        Path(
            "notebooks/official-evaluation/"
            "10r-toysad8k-official-release.ipynb"
        ),
        Path(
            "notebooks/official-evaluation/"
            "10a-toysad8k-official-seen-frozen-inference.ipynb"
        ),
        Path(
            "notebooks/official-evaluation/"
            "10a1-toysad8k-official-unseen-frozen-inference.ipynb"
        ),
        Path(
            "notebooks/official-evaluation/"
            "10a-finalize-toysad8k-official-prediction-summary.ipynb"
        ),
        Path(
            "notebooks/official-evaluation/"
            "10b-toysad8k-official-seen-unseen-evaluation.ipynb"
        ),
        Path(
            "notebooks/post-freeze-sensitivity/"
            "11r-toysad8k-full-pool-release.ipynb"
        ),
        Path(
            "notebooks/post-freeze-sensitivity/"
            "11a-toysad8k-full-pool-frozen-inference.ipynb"
        ),
        Path(
            "notebooks/post-freeze-sensitivity/"
            "11b-toysad8k-full-pool-sensitivity-evaluation.ipynb"
        ),
    }
)

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

FORBIDDEN_FILE_NAMES = frozenset(
    {
        ".env",
        "clearml.conf",
        "credentials.json",
        "kaggle.json",
        "secrets.json",
    }
)
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "__pycache__",
        ".git",
        ".local",
        ".venv",
        "outputs",
        "private-vault",
        "transfers",
    }
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Snapshot source cannot contain a symlink: {path}")
        if not path.is_file() or path.suffix not in suffixes:
            continue
        relative = path.relative_to(REPOSITORY_ROOT)
        if path.name.lower() in FORBIDDEN_FILE_NAMES or any(
            part.lower() in FORBIDDEN_PATH_PARTS for part in relative.parts
        ):
            continue
        files.append(path)
    return files


def verify_snapshot_record(payload_root: Path) -> dict[str, object]:
    """Verify the complete snapshot inventory, hashes and aggregate digest."""

    payload_root = payload_root.expanduser().resolve()
    record_path = payload_root / "snapshot_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Snapshot record does not contain file hashes")
    if int(record.get("file_count", -1)) != len(files):
        raise ValueError("Snapshot record file count differs")
    verified_lines = []
    for name, digest in sorted(files.items()):
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Snapshot record contains an unsafe path")
        path = payload_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Snapshot record file is missing or unsafe: {name}")
        if file_sha256(path) != digest:
            raise ValueError(f"Snapshot record hash mismatch: {name}")
        verified_lines.append(f"{digest}  {name}\n")
    expected_files = set(files) | {"snapshot_record.json"}
    actual_files = {
        path.relative_to(payload_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Snapshot payload contains an undeclared or missing file")
    aggregate = sha256("".join(verified_lines).encode("utf-8")).hexdigest()
    if aggregate != record.get("snapshot_sha256"):
        raise ValueError("Snapshot aggregate SHA-256 differs")
    return record


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
    notebook_version_03_1: str = "pending",
    notebook_version_04_1: str = "pending",
    notebook_version_05_1: str = "pending",
    notebook_version_06_1: str = "pending",
    notebook_version_04_2: str = "pending",
    notebook_version_07_1: str = "pending",
    notebook_version_04_3: str = "pending",
    notebook_version_08a: str = "pending",
    notebook_version_08b: str = "pending",
    notebook_version_08c: str = "pending",
    notebook_version_08d: str = "pending",
    notebook_version_08e: str = "pending",
    notebook_version_08e1: str = "pending",
    notebook_version_08f: str = "pending",
    notebook_version_08f1: str = "pending",
    notebook_version_09a: str = "pending",
    notebook_version_09b: str = "pending",
    notebook_version_09b1: str = "pending",
    notebook_version_09c: str = "pending",
    notebook_version_09d: str = "pending",
    notebook_version_10r: str = "pending",
    notebook_version_10a: str = "pending",
    notebook_version_10a1: str = "pending",
    notebook_version_10a_finalize: str = "pending",
    notebook_version_10b: str = "pending",
    notebook_version_11r: str = "pending",
    notebook_version_11a: str = "pending",
    notebook_version_11b: str = "pending",
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
        if source_root.is_symlink() or not source_root.is_dir():
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
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Snapshot notebook is missing: {source_relative}")
        if target_relative in copied:
            raise ValueError(f"Snapshot target is duplicated: {target_relative}")
        destination = payload_root / target_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(target_relative)

    missing_ds1_notebooks = DS1_NOTEBOOK_PATHS.difference(copied)
    if missing_ds1_notebooks:
        raise RuntimeError(
            f"Snapshot omits DS1 notebook paths: {sorted(map(str, missing_ds1_notebooks))}"
        )
    missing_ds2_notebooks = DS2_NOTEBOOK_PATHS.difference(copied)
    if missing_ds2_notebooks:
        raise RuntimeError(
            f"Snapshot omits DS2 notebook paths: {sorted(map(str, missing_ds2_notebooks))}"
        )
    missing_remaining_notebooks = REMAINING_NOTEBOOK_PATHS.difference(copied)
    if missing_remaining_notebooks:
        raise RuntimeError(
            "Snapshot omits remaining notebook paths: "
            f"{sorted(map(str, missing_remaining_notebooks))}"
        )

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
            "kaggle_notebook_version_03_1": str(notebook_version_03_1),
            "kaggle_notebook_version_04_1": str(notebook_version_04_1),
            "kaggle_notebook_version_05_1": str(notebook_version_05_1),
            "kaggle_notebook_version_06_1": str(notebook_version_06_1),
            "kaggle_notebook_version_04_2": str(notebook_version_04_2),
            "kaggle_notebook_version_07_1": str(notebook_version_07_1),
            "kaggle_notebook_version_04_3": str(notebook_version_04_3),
            "kaggle_notebook_version_08a": str(notebook_version_08a),
            "kaggle_notebook_version_08b": str(notebook_version_08b),
            "kaggle_notebook_version_08c": str(notebook_version_08c),
            "kaggle_notebook_version_08d": str(notebook_version_08d),
            "kaggle_notebook_version_08e": str(notebook_version_08e),
            "kaggle_notebook_version_08e1": str(notebook_version_08e1),
            "kaggle_notebook_version_08f": str(notebook_version_08f),
            "kaggle_notebook_version_08f1": str(notebook_version_08f1),
            "kaggle_notebook_version_09a": str(notebook_version_09a),
            "kaggle_notebook_version_09b": str(notebook_version_09b),
            "kaggle_notebook_version_09b1": str(notebook_version_09b1),
            "kaggle_notebook_version_09c": str(notebook_version_09c),
            "kaggle_notebook_version_09d": str(notebook_version_09d),
            "kaggle_notebook_version_10r": str(notebook_version_10r),
            "kaggle_notebook_version_10a": str(notebook_version_10a),
            "kaggle_notebook_version_10a1": str(notebook_version_10a1),
            "kaggle_notebook_version_10a_finalize": str(
                notebook_version_10a_finalize
            ),
            "kaggle_notebook_version_10b": str(notebook_version_10b),
            "kaggle_notebook_version_11r": str(notebook_version_11r),
            "kaggle_notebook_version_11a": str(notebook_version_11a),
            "kaggle_notebook_version_11b": str(notebook_version_11b),
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

    verify_snapshot_record(payload_root)

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
    parser.add_argument("--notebook-version-03-1", default="pending")
    parser.add_argument("--notebook-version-04-1", default="pending")
    parser.add_argument("--notebook-version-05-1", default="pending")
    parser.add_argument("--notebook-version-06-1", default="pending")
    parser.add_argument("--notebook-version-04-2", default="pending")
    parser.add_argument("--notebook-version-07-1", default="pending")
    parser.add_argument("--notebook-version-04-3", default="pending")
    parser.add_argument("--notebook-version-08a", default="pending")
    parser.add_argument("--notebook-version-08b", default="pending")
    parser.add_argument("--notebook-version-08c", default="pending")
    parser.add_argument("--notebook-version-08d", default="pending")
    parser.add_argument("--notebook-version-08e", default="pending")
    parser.add_argument("--notebook-version-08e1", default="pending")
    parser.add_argument("--notebook-version-08f", default="pending")
    parser.add_argument("--notebook-version-08f1", default="pending")
    parser.add_argument("--notebook-version-09a", default="pending")
    parser.add_argument("--notebook-version-09b", default="pending")
    parser.add_argument("--notebook-version-09b1", default="pending")
    parser.add_argument("--notebook-version-09c", default="pending")
    parser.add_argument("--notebook-version-09d", default="pending")
    parser.add_argument("--notebook-version-10r", default="pending")
    parser.add_argument("--notebook-version-10a", default="pending")
    parser.add_argument("--notebook-version-10a1", default="pending")
    parser.add_argument("--notebook-version-10a-finalize", default="pending")
    parser.add_argument("--notebook-version-10b", default="pending")
    parser.add_argument("--notebook-version-11r", default="pending")
    parser.add_argument("--notebook-version-11a", default="pending")
    parser.add_argument("--notebook-version-11b", default="pending")
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
            notebook_version_03_1=arguments.notebook_version_03_1,
            notebook_version_04_1=arguments.notebook_version_04_1,
            notebook_version_05_1=arguments.notebook_version_05_1,
            notebook_version_06_1=arguments.notebook_version_06_1,
            notebook_version_04_2=arguments.notebook_version_04_2,
            notebook_version_07_1=arguments.notebook_version_07_1,
            notebook_version_04_3=arguments.notebook_version_04_3,
            notebook_version_08a=arguments.notebook_version_08a,
            notebook_version_08b=arguments.notebook_version_08b,
            notebook_version_08c=arguments.notebook_version_08c,
            notebook_version_08d=arguments.notebook_version_08d,
            notebook_version_08e=arguments.notebook_version_08e,
            notebook_version_08e1=arguments.notebook_version_08e1,
            notebook_version_08f=arguments.notebook_version_08f,
            notebook_version_08f1=arguments.notebook_version_08f1,
            notebook_version_09a=arguments.notebook_version_09a,
            notebook_version_09b=arguments.notebook_version_09b,
            notebook_version_09b1=arguments.notebook_version_09b1,
            notebook_version_09c=arguments.notebook_version_09c,
            notebook_version_09d=arguments.notebook_version_09d,
            notebook_version_10r=arguments.notebook_version_10r,
            notebook_version_10a=arguments.notebook_version_10a,
            notebook_version_10a1=arguments.notebook_version_10a1,
            notebook_version_10a_finalize=arguments.notebook_version_10a_finalize,
            notebook_version_10b=arguments.notebook_version_10b,
            notebook_version_11r=arguments.notebook_version_11r,
            notebook_version_11a=arguments.notebook_version_11a,
            notebook_version_11b=arguments.notebook_version_11b,
        )
    )
