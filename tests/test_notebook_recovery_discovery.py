from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest


def recovery_finder(roots: list[Path]):
    notebook = Path(__file__).resolve().parents[1] / (
        "notebooks/development-freeze/"
        "08e-toysad8k-frozen-development-pipeline.ipynb"
    )
    cells = json.loads(notebook.read_text())["cells"]
    source = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
        and "def find_recovery_pair(" in "".join(cell["source"])
    )
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "find_recovery_pair"
    )
    namespace = {"Path": Path, "os": os, "kaggle_roots": lambda: roots}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)  # noqa: S102
    return namespace["find_recovery_pair"]


@pytest.mark.parametrize("suffix", [".zip", ".zip.bin"])
def test_finds_archive_without_unpacking_and_deduplicates_roots(tmp_path, monkeypatch, suffix):
    monkeypatch.delenv("VPI_RESUME_ROOT", raising=False)
    package = tmp_path / "datasets/user/recovery"
    package.mkdir(parents=True)
    bundle = package / f"notebook08e-recovery{suffix}"
    manifest = package / "notebook08e-recovery-manifest.json"
    bundle.write_bytes(b"archive bytes")
    manifest.write_text("{}")
    assert recovery_finder([tmp_path, package])() == (bundle, manifest)


def test_unpacked_upload_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("VPI_RESUME_ROOT", raising=False)
    (tmp_path / "notebook08e-recovery-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="found 0 bundles and 1 manifests"):
        recovery_finder([tmp_path])()


def test_no_recovery_is_optional_but_multiple_archives_are_not(tmp_path, monkeypatch):
    monkeypatch.delenv("VPI_RESUME_ROOT", raising=False)
    find = recovery_finder([tmp_path])
    assert find() is None
    for directory in ("first", "second"):
        root = tmp_path / directory
        root.mkdir()
        (root / "notebook08e-recovery.zip.bin").write_bytes(b"archive bytes")
        (root / "notebook08e-recovery-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="found 2 bundles and 2 manifests"):
        find()
