"""Regression checks for notebook transport and interrupted resource releases."""

from __future__ import annotations

import ast
import errno
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from visual_peer_inspector.workflows.kaggle_recovery import restore_recovery_bundle
from visual_peer_inspector.workflows.runtime_failures import (
    raise_if_infrastructure_failure,
)

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = script("notebook_runtime_cells")
gate = script("run_primary_resource_gate")


def test_nested_mounts_exclude_container_and_duplicate_versions(tmp_path):
    children = ["flat-data", "datasets/owner/data/versions/3", "notebooks/owner/run/versions/5"]
    for child in children:
        (tmp_path / child).mkdir(parents=True)
    source = runtime.ROOT_DISCOVERY.format(name="kaggle_roots").replace(
        'Path("/kaggle/input")', f"Path({str(tmp_path)!r})"
    )
    namespace = {"Path": Path}
    exec(source, namespace)  # noqa: S102 - execute the code-owned notebook helper
    assert set(namespace["kaggle_roots"]()) == {tmp_path / child for child in children}


def test_inline_archive_discovery_accepts_byte_preserved_zip(tmp_path):
    expected = tmp_path / "full-pool-release-recovery.zip.bin"
    expected.write_bytes(b"fixture")
    source = (
        "from __future__ import annotations\n"
        "from pathlib import Path\n"
        f"source = Path({str(tmp_path)!r})\n"
        'found = list(source.rglob("full-pool-release-recovery.zip"))\n'
    )
    hardened = runtime.harden_source(source)
    namespace = {}
    exec(hardened, namespace)  # noqa: S102 - execute the code-owned notebook helper
    assert namespace["found"] == [expected]
    assert runtime.harden_source(hardened) == hardened


def test_hardening_does_not_change_model_cells():
    source = 'result = encoder.encode(crop)\nprint(result.shape)\n'
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    changed = runtime.harden_notebook(notebook)
    assert changed.cells[0].source == source


def test_resource_discovery_is_bounded_and_does_not_scan_raw_media(tmp_path, monkeypatch):
    builder = script("build_primary_resource_notebooks")
    source = runtime.harden_source(builder.HELPERS)
    namespace = {"Path": Path, "os": os, "json": json}
    exec(source, namespace)  # noqa: S102 - execute generated input-discovery helpers
    raw = tmp_path / "source"
    raw.mkdir()
    (raw / "archive_index.sqlite3").write_bytes(b"index")
    released = tmp_path / "predictions"
    nested = released / "output" / "screen" / "model-safe"
    nested.mkdir(parents=True)
    expected = nested / "prediction_lock.json"
    expected.write_text('{}')
    media = released / "model-input" / "media"
    media.mkdir(parents=True)
    (media / "prediction_lock.json").write_text('{"unrelated": true}')
    namespace["kaggle_roots"] = lambda: [raw, released, released]
    original_iterdir = Path.iterdir

    def guarded_iterdir(path):
        assert path != raw and media not in (path, *path.parents)
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr(Path, "rglob", lambda *_: pytest.fail("Unbounded traversal"))
    assert namespace["mounted_files"]("prediction_lock.json") == [expected]
    assert namespace["find_recovery_pair"]("development-identity") is None
    bundle = released / "development-identity-resource-recovery.zip.bin"
    manifest = released / "development-identity-resource-recovery-manifest.json"
    bundle.write_bytes(b"checkpoint")
    manifest.write_text('{}')
    assert namespace["find_recovery_pair"]("development-identity") == (bundle, manifest)
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "rglob"
        for node in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize("error", [
    MemoryError("allocation failed"),
    OSError(errno.ENOSPC, "disk full"),
    OSError(errno.EACCES, "permission denied"),
    RuntimeError("CUDA out of memory"),
    RuntimeError("CUDA error: an illegal memory access was encountered"),
])
def test_infrastructure_failure_is_not_an_image_outcome(error):
    with pytest.raises(type(error)):
        raise_if_infrastructure_failure(error)


def test_wrapped_cuda_failure_is_not_swallowed():
    cause = RuntimeError("CUDA out of memory")
    error = ValueError("feature extraction failed")
    error.__cause__ = cause
    with pytest.raises(ValueError, match="feature extraction"):
        raise_if_infrastructure_failure(error)
    raise_if_infrastructure_failure(ValueError("empty mask"))


def arguments(tmp_path):
    return SimpleNamespace(
        mode="development-identity", prediction_root=[tmp_path / "predictions"],
        output_root=tmp_path / "out", recovery_bundle=None, recovery_manifest=None,
        recovery_output_directory=tmp_path / "recovery",
    )


def test_failed_release_exports_completed_scenes_and_preserves_original_error(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    binding = {"stage": "fixture", "input_sha256": "a" * 64}
    monkeypatch.setattr(gate, "_binding", lambda _: binding)

    def interrupted(*_):
        state = args.output_root / "trusted-release-state/development-identity/S1.json"
        state.parent.mkdir(parents=True)
        state.write_text('{"input_signature": "fixture"}\n')
        raise RuntimeError("interrupted after one scene")

    monkeypatch.setattr(gate, "_build_and_publish", interrupted)
    with pytest.raises(RuntimeError, match="after one scene"):
        gate.run(args)
    record = json.loads((args.output_root / "resource_run_result.json").read_text())
    assert record["status"] == "interrupted"
    restore_recovery_bundle(
        bundle_path=Path(record["recovery"]["bundle"]),
        manifest_path=Path(record["recovery"]["manifest"]),
        destination_root=tmp_path / "restored", expected_binding=binding,
        allowed_top_levels=gate._top_levels(args.mode),
    )
    assert (tmp_path / "restored/trusted-release-state/development-identity/S1.json").is_file()


def test_failed_recovery_keeps_original_release_exception(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(gate, "_binding", lambda _: {"stage": "fixture"})
    def interrupted(*_):
        raise RuntimeError("original failure")
    def export_failed(*_):
        raise OSError(errno.ENOSPC, "disk full")
    monkeypatch.setattr(gate, "_build_and_publish", interrupted)
    monkeypatch.setattr(gate, "_save_recovery", export_failed)
    with pytest.raises(RuntimeError, match="original failure"):
        gate.run(args)
