"""Create three separate, checksummed upload bundles from an explicit allowlist."""

import argparse
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visual_peer_inspector.workflows.comparison_io import atomic_json, digest_file, verify_seal


def bundle(path, files):
    if path.exists():
        raise FileExistsError("Preserve the existing upload bundle; use a versioned directory")
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source, relative in files:
            if source.is_symlink():
                raise ValueError("Upload sources cannot contain symlinks")
            archive.write(source, relative)
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Upload ZIP failed integrity validation")
    return {"file": path.name, "sha256": digest_file(path), "bytes": path.stat().st_size, "files": len(files)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    if args.version < 1:
        raise ValueError("Version must be positive")
    destination = ROOT / f".local/upload-bundles-v{args.version}"
    destination.mkdir(parents=True, exist_ok=True)
    runtime_scripts = [
        "run_scoring_comparison.py", "scoring_comparison_runtime.py", "log_scoring_comparison.py",
        "prepare_scoring_comparison.py", "build_scoring_comparison_notebooks.py",
        "package_scoring_comparison.py", "validate_scoring_comparison_local.py", "kaggle_mounts.py",
    ]
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "scripts" / name for name in runtime_scripts]
    paths += sorted((ROOT / "notebooks").glob("08[gh]_*.ipynb"))
    paths += sorted((ROOT / "requirements").glob("scoring-comparison*.txt"))
    paths += [ROOT / "tests/test_scoring_comparison.py", ROOT / "tests/test_peer_support_and_context.py", ROOT / "comparison_base.json", ROOT / "README.md", ROOT / "KAGGLE_RUN_HANDOVER.md", ROOT / "KAGGLE_ISSUES.md", ROOT / "LOCAL_VALIDATION.md"]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("A required source artifact is missing")
    # No environment files, raw data, local test results or cache directories enter source.
    receipts = [bundle(destination / f"scoring-comparison-source-v{args.version}.zip", [(p, "scoring-comparison/" + str(p.relative_to(ROOT))) for p in paths])]
    reused = []
    for folder in ("model-input", "evaluator-only"):
        package = ROOT / ".local/inputs-v1" / folder
        verify_seal(package)
        existing = ROOT / ".local/upload-bundles" / f"scoring-comparison-{folder}.zip"
        if not existing.is_file():
            raise FileNotFoundError(f"The unchanged {folder} upload bundle is missing")
        reused.append({"file": str(existing.relative_to(ROOT)), "sha256": digest_file(existing), "bytes": existing.stat().st_size, "reason": "input lock unchanged"})
    atomic_json(destination / "bundle_checksums.json", {"schema": 2, "purpose": "updated source plus unchanged private Kaggle input resources, not a GitHub transfer", "new_bundles": receipts, "reused_bundles": reused})
    print(receipts)


if __name__ == "__main__":
    main()
